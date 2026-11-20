"""Modul Algoritma Aliran Daya Newton-Raphson Polar From-Scratch.

Modul ini mengimplementasikan penyelesaian persamaan aliran daya AC non-linear
dalam koordinat polar menggunakan metode Newton-Raphson (NR), mencakup:
  - Perhitungan daya injeksi terhitung (P_calc, Q_calc) dan vektor mismatch
    (Delta_P, Delta_Q).
  - Penyusunan 4 sub-matriks Jacobian (J1 = dP/dtheta, J2 = dP/d|V|,
    J3 = dQ/dtheta, J4 = dQ/d|V|) baik secara eksplisit (rumus trigonometri
    buku teks untuk pedagogi) maupun vektorisasi sparse.
  - Penegakan batas daya reaktif generator (Q_min, Q_max) tingkat industri:
    mendukung multi-generator per bus, seleksi pelanggaran terbesar per siklus
    (largest-violation-first), serta mekanisme back-switching (PQ -> PV) berbasis
    syarat komplementaritas AVR.
  - Deteksi dini divergensi, kolaps tegangan, dan singularitas matriks Jacobian.
  - Perhitungan aliran daya saluran (P, Q, S, loading) dan rugi-rugi transmisi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ybus import (
    BUS_I,
    BUS_TYPE,
    PD,
    QD,
    VM,
    VA,
    PQ,
    PV,
    REF,
    NONE,
    GEN_BUS,
    PG,
    QG,
    QMAX,
    QMIN,
    VG,
    GEN_STATUS,
    F_BUS,
    T_BUS,
    RATE_A,
    BR_STATUS,
    PowerSystemCase,
    build_ybus,
    get_ieee5_case,
    get_ieee14_case,
)


@dataclass
class BranchFlowRecord:
    """Data aliran daya pada satu cabang (saluran/transformator)."""

    branch_id: int
    from_bus: int
    to_bus: int
    p_from_mw: float
    q_from_mvar: float
    p_to_mw: float
    q_to_mvar: float
    p_loss_mw: float
    q_loss_mvar: float
    loading_percent: float


@dataclass
class NewtonRaphsonResult:
    """Struktur hasil eksekusi solver aliran daya Newton-Raphson."""

    converged: bool
    iterations: int
    mismatch_history: List[float]
    vm_pu: NDArray[np.float64]
    va_deg: NDArray[np.float64]
    v_complex: NDArray[np.complex128]
    p_calc_mw: NDArray[np.float64]
    q_calc_mvar: NDArray[np.float64]
    branch_flows: List[BranchFlowRecord]
    total_p_loss_mw: float
    total_q_loss_mvar: float
    switched_pv_buses: List[int] = field(default_factory=list)
    bus_types_final: NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )


def identify_bus_types(
    bus: NDArray[np.float64],
    gen: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Mengelompokkan indeks bus (0-based) ke dalam himpunan Slack (REF), PV, dan PQ.

    Apabila matriks `gen` diberikan, bus bertipe PV yang tidak memiliki satu pun
    generator aktif (`GEN_STATUS > 0`) secara otomatis didemosi menjadi bus PQ
    sesuai standar industri (MATPOWER `bustypes`). Bus terisolasi (`BUS_TYPE == NONE`)
    dikecualikan dari ketiga himpunan aktif.
    """
    types = bus[:, BUS_TYPE].astype(int).copy()
    if gen is not None:
        bus_map: Dict[int, int] = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
        on_gens = gen[gen[:, GEN_STATUS] > 0]
        active_gen_buses = {bus_map[int(g_bus)] for g_bus in on_gens[:, GEN_BUS]}
        for idx in range(bus.shape[0]):
            if types[idx] == PV and idx not in active_gen_buses:
                types[idx] = PQ

    unknown = set(types.tolist()) - {PQ, PV, REF, NONE}
    if unknown:
        raise ValueError(f"Ditemukan kode tipe bus tidak valid: {sorted(unknown)}")

    ref_idx = np.where(types == REF)[0]
    pv_idx = np.where(types == PV)[0]
    pq_idx = np.where(types == PQ)[0]
    if ref_idx.size == 0:
        raise ValueError("Sistem wajib memiliki minimal satu Slack/Reference bus.")
    return ref_idx, pv_idx, pq_idx


def compute_scheduled_injections(
    base_mva: float,
    bus: NDArray[np.float64],
    gen: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Menghitung vektor injeksi daya terjadwal P_sch dan Q_sch dalam per-unit.

    Mendukung banyak generator paralel pada satu bus melalui akumulasi `np.add.at`:
        P_sch,i = (sum_{g in i} P_G,g - P_D,i) / base_mva
        Q_sch,i = (sum_{g in i} Q_G,g - Q_D,i) / base_mva
    """
    n_bus = bus.shape[0]
    bus_map: Dict[int, int] = {int(b): i for i, b in enumerate(bus[:, BUS_I])}

    p_gen = np.zeros(n_bus, dtype=np.float64)
    q_gen = np.zeros(n_bus, dtype=np.float64)

    on_mask = gen[:, GEN_STATUS] > 0
    if np.any(on_mask):
        gen_idx = np.array(
            [bus_map[int(b)] for b in gen[on_mask, GEN_BUS]], dtype=int
        )
        np.add.at(p_gen, gen_idx, gen[on_mask, PG])
        np.add.at(q_gen, gen_idx, gen[on_mask, QG])

    p_sch = (p_gen - bus[:, PD]) / base_mva
    q_sch = (q_gen - bus[:, QD]) / base_mva
    return p_sch, q_sch


def compute_power_injections(
    y_bus: sp.csr_matrix,
    v_complex: NDArray[np.complex128],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Menghitung daya aktif dan reaktif terhitung (P_calc, Q_calc) dalam per-unit.

    Menggunakan persamaan aliran daya polar:
        P_i = |V_i| * sum_k |V_k| * (G_ik cos(theta_ik) + B_ik sin(theta_ik))
        Q_i = |V_i| * sum_k |V_k| * (G_ik sin(theta_ik) - B_ik cos(theta_ik))
    yang ekuivalen dengan bentuk kompleks S = V * conj(Y_bus @ V).
    """
    i_bus = y_bus @ v_complex
    s_calc = v_complex * np.conj(i_bus)
    return s_calc.real.astype(np.float64), s_calc.imag.astype(np.float64)


def compute_mismatch_vector(
    y_bus: sp.csr_matrix,
    v_complex: NDArray[np.complex128],
    p_sch: NDArray[np.float64],
    q_sch: NDArray[np.float64],
    pv_idx: NDArray[np.int64],
    pq_idx: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Menghitung vektor mismatch daya (Delta_P, Delta_Q) untuk iterasi NR.

    Konvensi tanda yang digunakan adalah:
        Delta_P = P_sch - P_calc   (untuk bus i in PV U PQ)
        Delta_Q = Q_sch - Q_calc   (untuk bus i in PQ)
    sehingga sistem persamaan linear Newton-Raphson berbentuk:
        J * [Delta_theta; Delta_|V|] = [Delta_P; Delta_Q].
    """
    p_calc, q_calc = compute_power_injections(y_bus, v_complex)
    pvpq_idx = np.concatenate([pv_idx, pq_idx])
    delta_p = p_sch[pvpq_idx] - p_calc[pvpq_idx]
    delta_q = q_sch[pq_idx] - q_calc[pq_idx]
    mismatch_vec = np.concatenate([delta_p, delta_q])
    return delta_p, delta_q, mismatch_vec


def build_jacobian_explicit(
    y_bus: sp.csr_matrix,
    v_complex: NDArray[np.complex128],
    pv_idx: NDArray[np.int64],
    pq_idx: NDArray[np.int64],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Menyusun 4 sub-matriks Jacobian secara eksplisit per-elemen (referensi pedagogis).

    Fungsi ini menerjemahkan langsung rumus trigonometri buku teks untuk membantu
    peserta memverifikasi kesetaraan matematis dengan versi vektorisasi sparse:
      Off-diagonal (i != k):
        J1[r, c] =  |Vi||Vk| (G_ik sin(theta_ik) - B_ik cos(theta_ik))
        J2[r, c] =  |Vi|     (G_ik cos(theta_ik) + B_ik sin(theta_ik))
        J3[r, c] = -|Vi||Vk| (G_ik cos(theta_ik) + B_ik sin(theta_ik))
        J4[r, c] =  |Vi|     (G_ik sin(theta_ik) - B_ik cos(theta_ik))
      Diagonal (i == k):
        J1[r, r] = -Q_i - B_ii |Vi|^2
        J2[r, c] =  P_i / |Vi| + G_ii |Vi|
        J3[r, c] =  P_i - G_ii |Vi|^2
        J4[r, r] =  Q_i / |Vi| - B_ii |Vi|
    """
    y_dense = y_bus.toarray()
    g_mat, b_mat = y_dense.real, y_dense.imag
    v_mag = np.maximum(np.abs(v_complex), 1e-12)
    v_ang = np.angle(v_complex)
    p_calc, q_calc = compute_power_injections(y_bus, v_complex)

    pvpq_idx = np.concatenate([pv_idx, pq_idx])
    n_pvpq = pvpq_idx.size
    n_pq = pq_idx.size

    j1 = np.zeros((n_pvpq, n_pvpq), dtype=np.float64)
    j2 = np.zeros((n_pvpq, n_pq), dtype=np.float64)
    j3 = np.zeros((n_pq, n_pvpq), dtype=np.float64)
    j4 = np.zeros((n_pq, n_pq), dtype=np.float64)

    for r, i in enumerate(pvpq_idx):
        for c, k in enumerate(pvpq_idx):
            if i == k:
                j1[r, c] = -q_calc[i] - b_mat[i, i] * (v_mag[i] ** 2)
            else:
                th_ik = v_ang[i] - v_ang[k]
                j1[r, c] = (
                    v_mag[i]
                    * v_mag[k]
                    * (g_mat[i, k] * np.sin(th_ik) - b_mat[i, k] * np.cos(th_ik))
                )
        for c, k in enumerate(pq_idx):
            if i == k:
                j2[r, c] = p_calc[i] / v_mag[i] + g_mat[i, i] * v_mag[i]
            else:
                th_ik = v_ang[i] - v_ang[k]
                j2[r, c] = v_mag[i] * (
                    g_mat[i, k] * np.cos(th_ik) + b_mat[i, k] * np.sin(th_ik)
                )

    for r, i in enumerate(pq_idx):
        for c, k in enumerate(pvpq_idx):
            if i == k:
                j3[r, c] = p_calc[i] - g_mat[i, i] * (v_mag[i] ** 2)
            else:
                th_ik = v_ang[i] - v_ang[k]
                j3[r, c] = (
                    -v_mag[i]
                    * v_mag[k]
                    * (g_mat[i, k] * np.cos(th_ik) + b_mat[i, k] * np.sin(th_ik))
                )
        for c, k in enumerate(pq_idx):
            if i == k:
                j4[r, c] = q_calc[i] / v_mag[i] - b_mat[i, i] * v_mag[i]
            else:
                th_ik = v_ang[i] - v_ang[k]
                j4[r, c] = v_mag[i] * (
                    g_mat[i, k] * np.sin(th_ik) - b_mat[i, k] * np.cos(th_ik)
                )

    j_full = np.block([[j1, j2], [j3, j4]])
    return j1, j2, j3, j4, j_full


def build_jacobian_submatrices(
    y_bus: sp.csr_matrix,
    v_complex: NDArray[np.complex128],
    pv_idx: NDArray[np.int64],
    pq_idx: NDArray[np.int64],
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Menyusun 4 sub-matriks Jacobian (J1, J2, J3, J4) secara vektorisasi sparse.

    Definisi sub-matriks (dengan variabel keadaan x = [theta_{PV U PQ}, |V|_{PQ}]^T):
        J1 = dP / d(theta)    berukuran (N_pv + N_pq, N_pv + N_pq)
        J2 = dP / d|V|        berukuran (N_pv + N_pq, N_pq)
        J3 = dQ / d(theta)    berukuran (N_pq, N_pv + N_pq)
        J4 = dQ / d|V|        berukuran (N_pq, N_pq)

    Perumusan dilakukan melalui turunan daya kompleks:
        dS / d(theta) = j * diag(V) * conj(diag(I_bus) - Y_bus * diag(V))
        dS / d|V|     = diag(V) * conj(Y_bus * diag(V / |V|)) + conj(diag(I_bus)) * diag(V / |V|)
    """
    pvpq_idx = np.concatenate([pv_idx, pq_idx])
    i_bus = y_bus @ v_complex
    v_safe = np.maximum(np.abs(v_complex), 1e-12)
    v_norm = v_complex / v_safe

    diag_v = sp.diags(v_complex, format="csr")
    diag_i = sp.diags(i_bus, format="csr")
    diag_v_norm = sp.diags(v_norm, format="csr")

    ds_dva = 1j * diag_v @ (diag_i - y_bus @ diag_v).conj()
    ds_dvm = diag_v @ (y_bus @ diag_v_norm).conj() + diag_i.conj() @ diag_v_norm

    j1 = ds_dva[pvpq_idx][:, pvpq_idx].real.tocsr()
    j2 = ds_dvm[pvpq_idx][:, pq_idx].real.tocsr()
    j3 = ds_dva[pq_idx][:, pvpq_idx].imag.tocsr()
    j4 = ds_dvm[pq_idx][:, pq_idx].imag.tocsr()

    j_full = sp.bmat([[j1, j2], [j3, j4]], format="csr")
    return j1, j2, j3, j4, j_full


def compute_branch_flows(
    base_mva: float,
    bus: NDArray[np.float64],
    branch: NDArray[np.float64],
    v_complex: NDArray[np.complex128],
    y_f: sp.csr_matrix,
    y_t: sp.csr_matrix,
) -> Tuple[List[BranchFlowRecord], float, float]:
    """Menghitung aliran daya aktif/reaktif pada tiap cabang dan total rugi-rugi."""
    bus_map: Dict[int, int] = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    f_idx = np.array([bus_map[int(b)] for b in branch[:, F_BUS]], dtype=int)
    t_idx = np.array([bus_map[int(b)] for b in branch[:, T_BUS]], dtype=int)

    s_from = v_complex[f_idx] * np.conj(y_f @ v_complex) * base_mva
    s_to = v_complex[t_idx] * np.conj(y_t @ v_complex) * base_mva
    s_loss = s_from + s_to

    records: List[BranchFlowRecord] = []
    for k in range(branch.shape[0]):
        if branch[k, BR_STATUS] <= 0:
            continue
        s_max_end = max(abs(s_from[k]), abs(s_to[k]))
        rate_a = float(branch[k, RATE_A])
        loading = (100.0 * s_max_end / rate_a) if rate_a > 0.0 else 0.0
        records.append(
            BranchFlowRecord(
                branch_id=k + 1,
                from_bus=int(branch[k, F_BUS]),
                to_bus=int(branch[k, T_BUS]),
                p_from_mw=float(s_from[k].real),
                q_from_mvar=float(s_from[k].imag),
                p_to_mw=float(s_to[k].real),
                q_to_mvar=float(s_to[k].imag),
                p_loss_mw=float(s_loss[k].real),
                q_loss_mvar=float(s_loss[k].imag),
                loading_percent=float(loading),
            )
        )

    total_p_loss = float(np.sum(s_loss.real))
    total_q_loss = float(np.sum(s_loss.imag))
    return records, total_p_loss, total_q_loss


def solve_newton_raphson(
    case: PowerSystemCase,
    tol: float = 1e-8,
    max_iter: int = 30,
    enforce_q_limits: bool = True,
    flat_start: bool = True,
    max_dtheta_rad: float = 0.5,
    max_dvm_pu: float = 0.25,
) -> NewtonRaphsonResult:
    """Menjalankan algoritma Newton-Raphson polar lengkap dengan Q-limit switching robust.

    Fitur ketangguhan numerik & pemodelan:
      1. Bus PV tanpa generator aktif otomatis didemosi menjadi bus PQ.
      2. Kapasitas daya reaktif (Q_min, Q_max) diagregasi per bus sehingga mendukung
         banyak generator paralel pada satu bus.
      3. Seleksi satu pelanggaran daya reaktif terbesar per siklus luar
         (largest-violation-first) dipadukan dengan mekanisme back-switching (PQ -> PV)
         apabila tegangan menyimpang ke arah yang memulihkan kendali AVR:
           - Terkunci pada Q_max namun |V_i| > V_set,i  -> pulihkan ke PV
           - Terkunci pada Q_min namun |V_i| < V_set,i  -> pulihkan ke PV
      4. Deteksi dini matriks Jacobian singular (pulau tanpa referensi), nilai NaN/Inf,
         serta pembatasan langkah koreksi (step clamping) untuk mencegah kolaps numerik.

    Parameters
    ----------
    case : PowerSystemCase
        Data sistem tenaga listrik (baseMVA, bus, gen, branch).
    tol : float, default=1e-8
        Toleransi konvergensi norm-infinit vektor mismatch dalam per-unit.
    max_iter : int, default=30
        Batas maksimum iterasi Newton-Raphson per siklus Q-limit.
    enforce_q_limits : bool, default=True
        Aktifkan pengecekan batas daya reaktif generator (PV <-> PQ switching).
    flat_start : bool, default=True
        Jika True, inisialisasi tegangan bus PQ pada 1.0 < 0° dan bus generator
        aktif pada setpoint VG < 0°.
    max_dtheta_rad : float, default=0.5
        Batas maksimum perubahan sudut tegangan per iterasi (radian).
    max_dvm_pu : float, default=0.25
        Batas maksimum perubahan magnitudo tegangan per iterasi (p.u.).

    Returns
    -------
    NewtonRaphsonResult
        Objek hasil yang memuat status konvergensi, riwayat mismatch, profil V-theta,
        aliran daya tiap cabang, dan daftar bus PV yang terkunci menjadi PQ di akhir.
    """
    base_mva = float(case["baseMVA"])
    bus = case["bus"].copy()
    gen = case["gen"].copy()
    branch = case["branch"].copy()

    n_bus = bus.shape[0]
    bus_map: Dict[int, int] = {int(b): i for i, b in enumerate(bus[:, BUS_I])}

    y_bus, y_f, y_t = build_ybus(base_mva, bus, branch)

    # Inisialisasi vektor tegangan kompleks
    if flat_start:
        v_mag = np.ones(n_bus, dtype=np.float64)
        v_ang = np.zeros(n_bus, dtype=np.float64)
    else:
        v_mag = bus[:, VM].astype(np.float64).copy()
        v_ang = np.deg2rad(bus[:, VA].astype(np.float64))

    on_gens = np.where(gen[:, GEN_STATUS] > 0)[0]
    gen_bus_idx = np.array([bus_map[int(gen[g, GEN_BUS])] for g in on_gens], dtype=int)
    active_gen_buses = set(gen_bus_idx.tolist())

    for g in on_gens:
        g_idx = bus_map[int(gen[g, GEN_BUS])]
        v_mag[g_idx] = float(gen[g, VG])

    # Demosi bus PV tanpa generator aktif menjadi PQ & simpan setpoint asli V_set
    orig_pv_vset: Dict[int, float] = {}
    for i in range(n_bus):
        if int(bus[i, BUS_TYPE]) == PV:
            if i not in active_gen_buses:
                bus[i, BUS_TYPE] = PQ
            else:
                orig_pv_vset[i] = float(v_mag[i])

    # Agregasi kapasitas reaktif per bus (mendukung multi-generator di satu bus)
    qmax_bus = np.zeros(n_bus, dtype=np.float64)
    qmin_bus = np.zeros(n_bus, dtype=np.float64)
    if on_gens.size > 0:
        np.add.at(qmax_bus, gen_bus_idx, gen[on_gens, QMAX])
        np.add.at(qmin_bus, gen_bus_idx, gen[on_gens, QMIN])

    v_complex = v_mag * np.exp(1j * v_ang)

    mismatch_history: List[float] = []
    q_locked: Dict[int, Tuple[str, float]] = {}  # b_idx -> ("QMAX"|"QMIN", q_lim_mvar)
    revert_count: Dict[int, int] = {i: 0 for i in orig_pv_vset}
    total_iterations = 0
    converged = False

    max_outer_loops = 2 * len(orig_pv_vset) + 2
    for _outer in range(max_outer_loops):
        ref_idx, pv_idx, pq_idx = identify_bus_types(bus)
        pvpq_idx = np.concatenate([pv_idx, pq_idx])
        n_pvpq = pvpq_idx.size

        p_sch, q_sch = compute_scheduled_injections(base_mva, bus, gen)
        for b_idx, (_, q_lim_mvar) in q_locked.items():
            q_sch[b_idx] = (q_lim_mvar - bus[b_idx, QD]) / base_mva

        converged = False
        for it in range(max_iter + 1):
            _, _, mismatch_vec = compute_mismatch_vector(
                y_bus, v_complex, p_sch, q_sch, pv_idx, pq_idx
            )
            max_mismatch = (
                float(np.linalg.norm(mismatch_vec, ord=np.inf))
                if mismatch_vec.size > 0
                else 0.0
            )

            if it > 0 or not mismatch_history:
                mismatch_history.append(max_mismatch)

            # Deteksi dini divergensi numerik atau kolaps tegangan non-fisik
            if not np.isfinite(max_mismatch) or max_mismatch > 1e5 or np.any(v_mag < 0.2):
                converged = False
                break

            if max_mismatch < tol:
                converged = True
                break

            if it == max_iter:
                break

            _, _, _, _, j_full = build_jacobian_submatrices(
                y_bus, v_complex, pv_idx, pq_idx
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    dx = spsolve(j_full.tocsc(), mismatch_vec)
                except Exception:
                    converged = False
                    break

            if not np.all(np.isfinite(dx)):
                converged = False
                break

            # Step-size clamping untuk menjaga stabilitas iterasi pada beban berat
            max_da = float(np.max(np.abs(dx[:n_pvpq]), initial=0.0))
            max_dv = float(np.max(np.abs(dx[n_pvpq:]), initial=0.0))
            scale = min(
                1.0,
                max_dtheta_rad / max(max_da, 1e-12),
                max_dvm_pu / max(max_dv, 1e-12),
            )

            total_iterations += 1
            v_ang[pvpq_idx] += scale * dx[:n_pvpq]
            v_mag[pq_idx] += scale * dx[n_pvpq:]
            v_complex = v_mag * np.exp(1j * v_ang)

        if not converged or not enforce_q_limits:
            break

        _, q_calc_pu = compute_power_injections(y_bus, v_complex)
        state_changed = False

        # 1. Back-switching PQ -> PV jika syarat komplementaritas AVR dilanggar
        for b_idx, (mode, _) in list(q_locked.items()):
            v_set = orig_pv_vset[b_idx]
            if revert_count[b_idx] < 2 and (
                (mode == "QMAX" and v_mag[b_idx] > v_set + 1e-5)
                or (mode == "QMIN" and v_mag[b_idx] < v_set - 1e-5)
            ):
                del q_locked[b_idx]
                bus[b_idx, BUS_TYPE] = PV
                v_mag[b_idx] = v_set
                v_complex[b_idx] = v_set * np.exp(1j * v_ang[b_idx])
                revert_count[b_idx] += 1
                state_changed = True

        # 2. Konversi PV -> PQ untuk 1 pelanggaran terbesar (largest-violation-first)
        worst_bus = -1
        worst_viol = 0.0
        worst_mode = ""
        worst_lim = 0.0

        for b_idx in pv_idx:
            qg_tot_mvar = q_calc_pu[b_idx] * base_mva + bus[b_idx, QD]
            if (
                qg_tot_mvar > qmax_bus[b_idx] + 1e-6
                and (qg_tot_mvar - qmax_bus[b_idx]) > worst_viol
            ):
                worst_bus = int(b_idx)
                worst_viol = float(qg_tot_mvar - qmax_bus[b_idx])
                worst_mode = "QMAX"
                worst_lim = float(qmax_bus[b_idx])
            elif (
                qg_tot_mvar < qmin_bus[b_idx] - 1e-6
                and (qmin_bus[b_idx] - qg_tot_mvar) > worst_viol
            ):
                worst_bus = int(b_idx)
                worst_viol = float(qmin_bus[b_idx] - qg_tot_mvar)
                worst_mode = "QMIN"
                worst_lim = float(qmin_bus[b_idx])

        if worst_bus >= 0:
            bus[worst_bus, BUS_TYPE] = PQ
            q_locked[worst_bus] = (worst_mode, worst_lim)
            state_changed = True

        if not state_changed:
            break

    switched_pv_buses = [int(bus[b_idx, BUS_I]) for b_idx in sorted(q_locked.keys())]
    p_calc_pu, q_calc_pu = compute_power_injections(y_bus, v_complex)
    flows, total_p_loss, total_q_loss = compute_branch_flows(
        base_mva, bus, branch, v_complex, y_f, y_t
    )

    return NewtonRaphsonResult(
        converged=converged,
        iterations=total_iterations,
        mismatch_history=mismatch_history,
        vm_pu=np.abs(v_complex),
        va_deg=np.rad2deg(np.angle(v_complex)),
        v_complex=v_complex,
        p_calc_mw=p_calc_pu * base_mva,
        q_calc_mvar=q_calc_pu * base_mva,
        branch_flows=flows,
        total_p_loss_mw=total_p_loss,
        total_q_loss_mvar=total_q_loss,
        switched_pv_buses=switched_pv_buses,
        bus_types_final=bus[:, BUS_TYPE].astype(np.int64),
    )


def format_power_flow_report(case: PowerSystemCase, res: NewtonRaphsonResult) -> str:
    """Menyusun laporan tekstual hasil aliran daya secara rapi."""
    type_labels = {PQ: "PQ", PV: "PV", REF: "Slack", NONE: "Isol"}
    lines = [
        "======================================================================",
        f"HASIL ALIRAN DAYA NEWTON-RAPHSON — {case['name']}",
        "======================================================================",
        f"Status Konvergensi : {'KONVERGEN' if res.converged else 'DIVERGEN'} "
        f"dalam {res.iterations} iterasi (||dF||_inf akhir = {res.mismatch_history[-1]:.3e} p.u.)",
        f"Bus PV -> PQ       : {res.switched_pv_buses if res.switched_pv_buses else 'Tidak ada'}",
        f"Total Rugi-Rugi    : {res.total_p_loss_mw:.4f} MW + j{res.total_q_loss_mvar:.4f} MVAr",
        "",
        "--- Profil Tegangan & Injeksi Bus ---",
        f"{'Bus':>4} {'Tipe':>6} {'|V| (pu)':>10} {'Sudut (°)':>11} {'P_inj (MW)':>12} {'Q_inj (MVAr)':>13}",
    ]
    for i, b_row in enumerate(case["bus"]):
        b_num = int(b_row[BUS_I])
        t_lbl = type_labels.get(int(res.bus_types_final[i]), "?")
        lines.append(
            f"{b_num:4d} {t_lbl:>6} {res.vm_pu[i]:10.5f} {res.va_deg[i]:11.4f} "
            f"{res.p_calc_mw[i]:12.3f} {res.q_calc_mvar[i]:13.3f}"
        )

    lines.extend(
        [
            "",
            "--- Aliran Daya Saluran / Transformator (5 Teratas Berdasarkan |P_from|) ---",
            f"{'ID':>3} {'Dari':>4} {'Ke':>4} {'P_from(MW)':>11} {'Q_from(MVAr)':>13} "
            f"{'P_loss(MW)':>11} {'Loading(%)':>11}",
        ]
    )
    top_flows = sorted(res.branch_flows, key=lambda r: abs(r.p_from_mw), reverse=True)[:5]
    for fl in top_flows:
        lines.append(
            f"{fl.branch_id:3d} {fl.from_bus:4d} {fl.to_bus:4d} "
            f"{fl.p_from_mw:11.3f} {fl.q_from_mvar:13.3f} "
            f"{fl.p_loss_mw:11.4f} {fl.loading_percent:11.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # 0. Verifikasi kesetaraan Jacobian eksplisit (buku teks) vs vektorisasi sparse
    case14_chk = get_ieee14_case()
    y_chk, _, _ = build_ybus(
        case14_chk["baseMVA"], case14_chk["bus"], case14_chk["branch"]
    )
    _, pv_chk, pq_chk = identify_bus_types(case14_chk["bus"], case14_chk["gen"])
    rng = np.random.default_rng(42)
    v_test = (1.0 + 0.03 * rng.standard_normal(14)) * np.exp(
        1j * 0.05 * rng.standard_normal(14)
    )
    _, _, _, _, j_exp = build_jacobian_explicit(y_chk, v_test, pv_chk, pq_chk)
    _, _, _, _, j_vec = build_jacobian_submatrices(y_chk, v_test, pv_chk, pq_chk)
    print(
        f"Verifikasi Jacobian Eksplisit vs Sparse: "
        f"max|J_exp - J_sparse| = {np.max(np.abs(j_exp - j_vec.toarray())):.3e}\n"
    )

    # 1. Demo pada IEEE 5-Bus
    case5 = get_ieee5_case()
    result5 = solve_newton_raphson(case5, tol=1e-9, enforce_q_limits=True)
    print(format_power_flow_report(case5, result5))
    print()

    # 2. Demo pada IEEE 14-Bus (kondisi normal)
    case14 = get_ieee14_case()
    result14 = solve_newton_raphson(case14, tol=1e-9, enforce_q_limits=False)
    print(format_power_flow_report(case14, result14))
    print()

    # 3. Demo penegakan batas daya reaktif (PV -> PQ switching) pada IEEE 14-Bus
    case14_qlim = get_ieee14_case()
    case14_qlim["name"] = "IEEE 14-Bus (Uji Q-Limit Bus 8: Qmax = 10 MVAr)"
    case14_qlim["gen"][4, QMAX] = 10.0
    result14_qlim = solve_newton_raphson(case14_qlim, tol=1e-9, enforce_q_limits=True)
    print(format_power_flow_report(case14_qlim, result14_qlim))
