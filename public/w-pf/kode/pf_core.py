"""
pf_core.py — Pustaka inti aliran daya "from scratch" untuk pelatihan.

Isi:
  make_ybus()                : Matriks admitansi bus sparse (Modul 1)
  make_sbus()                : Injeksi daya kompleks terjadwal (p.u.)
  jacobian_polar()           : Jacobian polar tervektorisasi (Modul 2)
  jacobian_polar_explicit()  : Jacobian per-elemen (versi pedagogis, untuk verifikasi)
  newton_raphson()           : Solver NR polar (dengan deteksi singularitas & damping)
  runpf_qlim()               : NR + Q-limit switching PV <-> PQ (multi-gen & back-switching)
  fdlf_xb()                  : Fast Decoupled Load Flow (skema XB)
  gauss_seidel()             : Gauss-Seidel dengan faktor akselerasi
  branch_flows()             : Aliran daya cabang & rugi-rugi

Konvensi data mengikuti format MATPOWER (lihat ieee14_data.py).
"""
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve, splu

from ieee14_data import (BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, GEN_BUS, PG, QG,
                         QMAX, QMIN, VG, GEN_STATUS, F_BUS, T_BUS, BR_R, BR_X,
                         BR_B, TAP, SHIFT, BR_STATUS, PQ, PV, REF)

NONE = 4  # kode bus terisolasi standar MATPOWER


def _bus_map(bus):
    """Memetakan nomor bus eksternal pada kolom BUS_I ke indeks baris 0..nb-1."""
    return {int(b): i for i, b in enumerate(bus[:, BUS_I])}


# ---------------------------------------------------------------------------
# MODUL 1 — Matriks admitansi bus
# ---------------------------------------------------------------------------
def make_ybus(baseMVA, bus, branch):
    """Membangun Ybus, Yf, Yt (scipy.sparse CSR) dengan model nominal-pi + trafo bertap.

    Untuk setiap cabang:  ys = 1/(R + jX),  a = t * exp(j*phi)
        Yff = (ys + j*B/2) / t^2      Yft = -ys / conj(a)
        Ytf = -ys / a                 Ytt =  ys + j*B/2
    Ybus = Cf^T * Yf + Ct^T * Yt + diag(Ysh)
    """
    nb, nl = bus.shape[0], branch.shape[0]
    bmap = _bus_map(bus)
    stat = branch[:, BR_STATUS]                       # 1 = in-service, 0 = out
    z_ser = branch[:, BR_R] + 1j * branch[:, BR_X]
    safe_z = np.where(stat > 0, z_ser, 1.0 + 0.0j)
    ys = np.where(stat > 0, stat / safe_z, 0.0 + 0.0j)
    bc = stat * branch[:, BR_B]
    tap = np.where(branch[:, TAP] == 0, 1.0, branch[:, TAP])
    tap = tap * np.exp(1j * np.deg2rad(branch[:, SHIFT]))

    Ytt = ys + 1j * bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -ys / np.conj(tap)
    Ytf = -ys / tap

    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA     # shunt bus (MW, MVAr @ V=1 pu)

    f = np.array([bmap[int(b)] for b in branch[:, F_BUS]], dtype=int)
    t = np.array([bmap[int(b)] for b in branch[:, T_BUS]], dtype=int)
    l = np.arange(nl)
    Cf = sp.csr_matrix((np.ones(nl), (l, f)), shape=(nl, nb))
    Ct = sp.csr_matrix((np.ones(nl), (l, t)), shape=(nl, nb))

    Yf = sp.csr_matrix((np.r_[Yff, Yft], (np.r_[l, l], np.r_[f, t])), shape=(nl, nb))
    Yt = sp.csr_matrix((np.r_[Ytf, Ytt], (np.r_[l, l], np.r_[f, t])), shape=(nl, nb))
    Ybus = (Cf.T @ Yf + Ct.T @ Yt + sp.diags(Ysh)).tocsr()
    return Ybus, Yf, Yt


def make_sbus(baseMVA, bus, gen):
    """S_terjadwal = (Pg - Pd) + j(Qg - Qd)  dalam p.u. (mendukung multi-generator)."""
    nb = bus.shape[0]
    bmap = _bus_map(bus)
    on = gen[:, GEN_STATUS] > 0
    gbus = np.array([bmap[int(b)] for b in gen[on, GEN_BUS]], dtype=int)
    Sg = np.zeros(nb, dtype=complex)
    if gbus.size > 0:
        np.add.at(Sg, gbus, gen[on, PG] + 1j * gen[on, QG])
    return (Sg - (bus[:, PD] + 1j * bus[:, QD])) / baseMVA


def bus_types(bus, gen=None):
    """Mengembalikan indeks (ref, pv, pq). Jika `gen` diberikan, PV tanpa gen aktif -> PQ."""
    types = bus[:, BUS_TYPE].astype(int).copy()
    if gen is not None:
        bmap = _bus_map(bus)
        on = gen[:, GEN_STATUS] > 0
        active_gbus = {bmap[int(b)] for b in gen[on, GEN_BUS]}
        for i in range(len(types)):
            if types[i] == PV and i not in active_gbus:
                types[i] = PQ
    ref = np.where(types == REF)[0]
    pv = np.where(types == PV)[0]
    pq = np.where(types == PQ)[0]
    return ref, pv, pq


def initial_voltage(bus, gen, flat=True):
    """Tebakan awal: flat start (1∠0) dengan |V| bus generator = setpoint VG."""
    bmap = _bus_map(bus)
    V0 = np.ones(bus.shape[0], dtype=complex) if flat else \
        bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, 8]))
    on = gen[:, GEN_STATUS] > 0
    gbus = np.array([bmap[int(b)] for b in gen[on, GEN_BUS]], dtype=int)
    if gbus.size > 0:
        V0[gbus] = gen[on, VG] * V0[gbus] / np.maximum(np.abs(V0[gbus]), 1e-12)
    return V0


# ---------------------------------------------------------------------------
# MODUL 2 — Newton-Raphson polar
# ---------------------------------------------------------------------------
def dS_dV(Ybus, V):
    """Turunan injeksi daya kompleks S = diag(V) conj(Ybus V) terhadap |V| dan θ."""
    Ibus = Ybus @ V
    Vn = V / np.maximum(np.abs(V), 1e-12)
    dV, dI, dVn = sp.diags(V), sp.diags(Ibus), sp.diags(Vn)
    dS_dVm = dV @ np.conj(Ybus @ dVn) + np.conj(dI) @ dVn
    dS_dVa = 1j * dV @ np.conj(dI - Ybus @ dV)
    return dS_dVm, dS_dVa


def jacobian_polar(Ybus, V, pv, pq):
    """J = [[dP/dθ, dP/d|V|], [dQ/dθ, dQ/d|V|]]  (baris P: pv+pq, baris Q: pq)."""
    pvpq = np.r_[pv, pq]
    dS_dVm, dS_dVa = dS_dV(Ybus, V)
    J11 = dS_dVa[pvpq][:, pvpq].real
    J12 = dS_dVm[pvpq][:, pq].real
    J21 = dS_dVa[pq][:, pvpq].imag
    J22 = dS_dVm[pq][:, pq].imag
    return sp.bmat([[J11, J12], [J21, J22]], format="csr")


def jacobian_polar_explicit(Ybus, V, pv, pq):
    """Versi per-elemen (loop) dari rumus buku teks — lambat, hanya untuk pembelajaran.

    i≠k: H_ik = |Vi||Vk|(G_ik sinθ_ik − B_ik cosθ_ik)   N_ik = |Vi|(G_ik cosθ_ik + B_ik sinθ_ik)
         M_ik = −|Vi||Vk|(G_ik cosθ_ik + B_ik sinθ_ik)  L_ik = |Vi|(G_ik sinθ_ik − B_ik cosθ_ik)
    i=k: H_ii = −Q_i − B_ii|Vi|²     N_ii =  P_i/|Vi| + G_ii|Vi|
         M_ii =  P_i − G_ii|Vi|²     L_ii =  Q_i/|Vi| − B_ii|Vi|
    """
    Y = Ybus.toarray()
    G, B = Y.real, Y.imag
    Vm, th = np.maximum(np.abs(V), 1e-12), np.angle(V)
    S = V * np.conj(Y @ V)
    P, Q = S.real, S.imag
    pvpq = np.r_[pv, pq]
    n1, n2 = len(pvpq), len(pq)
    J = np.zeros((n1 + n2, n1 + n2))
    rows = [(r, i, "P") for r, i in enumerate(pvpq)] + [(n1 + r, i, "Q") for r, i in enumerate(pq)]
    cols = [(c, k, "th") for c, k in enumerate(pvpq)] + [(n1 + c, k, "V") for c, k in enumerate(pq)]
    for r, i, eq in rows:
        for c, k, var in cols:
            if i != k:
                tik = th[i] - th[k]
                s, co = np.sin(tik), np.cos(tik)
                if eq == "P" and var == "th":
                    J[r, c] = Vm[i] * Vm[k] * (G[i, k] * s - B[i, k] * co)
                elif eq == "P" and var == "V":
                    J[r, c] = Vm[i] * (G[i, k] * co + B[i, k] * s)
                elif eq == "Q" and var == "th":
                    J[r, c] = -Vm[i] * Vm[k] * (G[i, k] * co + B[i, k] * s)
                else:
                    J[r, c] = Vm[i] * (G[i, k] * s - B[i, k] * co)
            else:
                if eq == "P" and var == "th":
                    J[r, c] = -Q[i] - B[i, i] * Vm[i] ** 2
                elif eq == "P" and var == "V":
                    J[r, c] = P[i] / Vm[i] + G[i, i] * Vm[i]
                elif eq == "Q" and var == "th":
                    J[r, c] = P[i] - G[i, i] * Vm[i] ** 2
                else:
                    J[r, c] = Q[i] / Vm[i] - B[i, i] * Vm[i]
    return J


def mismatch(Ybus, V, Sbus, pv, pq):
    mis = V * np.conj(Ybus @ V) - Sbus
    return np.r_[mis[np.r_[pv, pq]].real, mis[pq].imag]


def newton_raphson(Ybus, Sbus, V0, ref, pv, pq, tol=1e-8, max_it=20, verbose=False,
                   max_dtheta=0.5, max_dvm=0.25):
    """Solver NR polar dengan deteksi singularitas/divergensi & step clamping."""
    V = V0.copy()
    Va, Vm = np.angle(V), np.abs(V)
    pvpq = np.r_[pv, pq]
    n1 = len(pvpq)
    F = mismatch(Ybus, V, Sbus, pv, pq)
    hist = [float(np.linalg.norm(F, np.inf)) if F.size else 0.0]
    if verbose:
        print(f"  it {0:2d}  max|ΔF| = {hist[-1]:.3e}")
    for it in range(1, max_it + 1):
        if not np.isfinite(hist[-1]) or hist[-1] > 1e5 or np.any(Vm < 0.2):
            return V, False, it - 1, hist
        if hist[-1] < tol:
            return V, True, it - 1, hist
        J = jacobian_polar(Ybus, V, pv, pq)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                dx = spsolve(J.tocsc(), -F)
            except Exception:
                return V, False, it - 1, hist
        if not np.all(np.isfinite(dx)):
            return V, False, it - 1, hist
        scale = min(1.0,
                    max_dtheta / max(float(np.max(np.abs(dx[:n1]), initial=0.0)), 1e-12),
                    max_dvm / max(float(np.max(np.abs(dx[n1:]), initial=0.0)), 1e-12))
        Va[pvpq] += scale * dx[:n1]
        Vm[pq] += scale * dx[n1:]
        V = Vm * np.exp(1j * Va)
        F = mismatch(Ybus, V, Sbus, pv, pq)
        hist.append(float(np.linalg.norm(F, np.inf)) if F.size else 0.0)
        if verbose:
            print(f"  it {it:2d}  max|ΔF| = {hist[-1]:.3e}")
    return V, hist[-1] < tol, max_it, hist


def runpf_qlim(case, tol=1e-8, max_it=20, enforce_q_lims=True, verbose=True):
    """NR + Q-limit switching (PV <-> PQ): mendukung multi-gen & back-switching."""
    baseMVA = case["baseMVA"]
    bus, gen, branch = case["bus"].copy(), case["gen"].copy(), case["branch"]
    nb = bus.shape[0]
    bmap = _bus_map(bus)
    Ybus, Yf, Yt = make_ybus(baseMVA, bus, branch)
    V = initial_voltage(bus, gen)

    on = np.where(gen[:, GEN_STATUS] > 0)[0]
    gbus_idx = np.array([bmap[int(gen[g, GEN_BUS])] for g in on], dtype=int)
    active_gbus = set(gbus_idx.tolist())

    orig_pv_vset = {}
    for i in range(nb):
        if int(bus[i, BUS_TYPE]) == PV:
            if i not in active_gbus:
                bus[i, BUS_TYPE] = PQ
            else:
                orig_pv_vset[i] = float(abs(V[i]))

    qmax_bus, qmin_bus = np.zeros(nb), np.zeros(nb)
    if on.size > 0:
        np.add.at(qmax_bus, gbus_idx, gen[on, QMAX])
        np.add.at(qmin_bus, gbus_idx, gen[on, QMIN])

    q_locked = {}  # b_idx -> (mode, q_lim_mvar)
    revert_count = {i: 0 for i in orig_pv_vset}
    all_hist = []
    ok = False

    for outer in range(1, 2 * len(orig_pv_vset) + 3):
        ref, pv, pq = bus_types(bus)
        Sbus = make_sbus(baseMVA, bus, gen)
        for b_i, (_, qlim) in q_locked.items():
            Sbus[b_i] = Sbus[b_i].real + 1j * (qlim - bus[b_i, QD]) / baseMVA
        V, ok, nit, hist = newton_raphson(Ybus, Sbus, V, ref, pv, pq, tol, max_it)
        all_hist += hist if outer == 1 else hist[1:]
        if verbose:
            print(f"[Loop Q-lim {outer}] konvergen={ok}, iterasi NR={nit}")
        if not ok or not enforce_q_lims:
            break

        Sinj = V * np.conj(Ybus @ V) * baseMVA
        changed = False

        # 1) Back-switching PQ -> PV jika syarat komplementaritas AVR dilanggar
        for b_i, (mode, _) in list(q_locked.items()):
            vset = orig_pv_vset[b_i]
            if revert_count[b_i] < 2 and (
                (mode == "QMAX" and abs(V[b_i]) > vset + 1e-5)
                or (mode == "QMIN" and abs(V[b_i]) < vset - 1e-5)
            ):
                del q_locked[b_i]
                bus[b_i, BUS_TYPE] = PV
                V[b_i] = vset * V[b_i] / max(abs(V[b_i]), 1e-12)
                revert_count[b_i] += 1
                changed = True
                if verbose:
                    print(f"   <- Bus {int(bus[b_i, BUS_I])}: kembali ke kendali PV (V={vset:.3f})")

        # 2) Konversi PV -> PQ untuk 1 pelanggaran terbesar (largest-violation-first)
        worst_b, worst_dv, worst_mode, worst_lim = -1, 0.0, "", 0.0
        for b_i in pv:
            qg_tot = Sinj[b_i].imag + bus[b_i, QD]
            if qg_tot > qmax_bus[b_i] + 1e-6 and (qg_tot - qmax_bus[b_i]) > worst_dv:
                worst_b, worst_dv, worst_mode, worst_lim = b_i, qg_tot - qmax_bus[b_i], "QMAX", qmax_bus[b_i]
            elif qg_tot < qmin_bus[b_i] - 1e-6 and (qmin_bus[b_i] - qg_tot) > worst_dv:
                worst_b, worst_dv, worst_mode, worst_lim = b_i, qmin_bus[b_i] - qg_tot, "QMIN", qmin_bus[b_i]

        if worst_b >= 0:
            qg_val = Sinj[worst_b].imag + bus[worst_b, QD]
            if verbose:
                print(f"   -> Gen bus {int(bus[worst_b, BUS_I])}: Qg={qg_val:.2f} MVAr melanggar batas, "
                      f"dikunci pada {worst_lim:.2f} MVAr (PV -> PQ)")
            bus[worst_b, BUS_TYPE] = PQ
            q_locked[worst_b] = (worst_mode, worst_lim)
            changed = True

        if not changed:
            break

    # Perbarui kolom QG dan PG Slack pada matriks gen secara aman (mendukung multi-gen)
    Sinj = V * np.conj(Ybus @ V) * baseMVA
    for b_i in active_gbus:
        gens_at_b = [g for g in on if bmap[int(gen[g, GEN_BUS])] == b_i]
        qg_tot = q_locked[b_i][1] if b_i in q_locked else (Sinj[b_i].imag + bus[b_i, QD])
        span = sum(max(gen[g, QMAX] - gen[g, QMIN], 0.0) for g in gens_at_b)
        for g in gens_at_b:
            w = (max(gen[g, QMAX] - gen[g, QMIN], 0.0) / span) if span > 0 else (1.0 / len(gens_at_b))
            gen[g, QG] = qg_tot * w
    for r_i in bus_types(bus)[0]:
        gens_at_r = [g for g in on if bmap[int(gen[g, GEN_BUS])] == r_i]
        if gens_at_r:
            pg_tot = Sinj[r_i].real + bus[r_i, PD]
            other_pg = sum(gen[g, PG] for g in gens_at_r[1:])
            gen[gens_at_r[0], PG] = pg_tot - other_pg

    switched = [int(bus[b_i, BUS_I]) for b_i in sorted(q_locked.keys())]
    return {"V": V, "converged": ok, "bus": bus, "gen": gen, "Ybus": Ybus,
            "Yf": Yf, "Yt": Yt, "history": all_hist, "switched": switched}


# ---------------------------------------------------------------------------
# Fast Decoupled (XB) & Gauss-Seidel
# ---------------------------------------------------------------------------
def fdlf_xb(case, tol=1e-8, max_it=50):
    """FDLF versi XB: B' tanpa R, shunt, dan tap; B'' dari Ybus tanpa phase-shift."""
    baseMVA, bus, gen, branch = case["baseMVA"], case["bus"], case["gen"], case["branch"]
    Ybus, _, _ = make_ybus(baseMVA, bus, branch)
    ref, pv, pq = bus_types(bus, gen)
    pvpq = np.r_[pv, pq]
    Sbus = make_sbus(baseMVA, bus, gen)

    br1 = branch.copy(); br1[:, BR_R] = 0; br1[:, BR_B] = 0; br1[:, TAP] = 1; br1[:, SHIFT] = 0
    bus1 = bus.copy(); bus1[:, BS] = 0
    Bp = -make_ybus(baseMVA, bus1, br1)[0].imag
    br2 = branch.copy(); br2[:, SHIFT] = 0
    Bpp = -make_ybus(baseMVA, bus, br2)[0].imag
    lu_p = splu(Bp[pvpq][:, pvpq].tocsc())
    lu_q = splu(Bpp[pq][:, pq].tocsc())

    V = initial_voltage(bus, gen)
    Va, Vm = np.angle(V), np.abs(V)
    hist = []
    for it in range(1, max_it + 1):
        mis = (V * np.conj(Ybus @ V) - Sbus) / np.maximum(Vm, 1e-12)
        hist.append(max(np.abs(mis[pvpq].real).max(), np.abs(mis[pq].imag).max()))
        if hist[-1] < tol:
            return V, True, it - 1, hist
        Va[pvpq] -= lu_p.solve(mis[pvpq].real)          # P-θ half-iteration
        V = Vm * np.exp(1j * Va)
        mis = (V * np.conj(Ybus @ V) - Sbus) / np.maximum(Vm, 1e-12)
        Vm[pq] -= lu_q.solve(mis[pq].imag)              # Q-V half-iteration
        V = Vm * np.exp(1j * Va)
    return V, False, max_it, hist


def gauss_seidel(case, tol=1e-8, max_it=2000, alpha=1.0):
    """Gauss-Seidel klasik; alpha = faktor akselerasi (≈1.4–1.6 umum dipakai)."""
    baseMVA, bus, gen, branch = case["baseMVA"], case["bus"], case["gen"], case["branch"]
    Y_csr = make_ybus(baseMVA, bus, branch)[0]
    Y = Y_csr.toarray()
    ref, pv, pq = bus_types(bus, gen)
    pv_set = set(pv.tolist())
    order = np.r_[pv, pq]
    Sbus = make_sbus(baseMVA, bus, gen)
    V = initial_voltage(bus, gen)
    Vset = np.abs(V)
    hist = []
    for it in range(1, max_it + 1):
        for i in order:
            S_i = Sbus[i]
            if i in pv_set:
                Q_i = -np.imag(np.conj(V[i]) * (Y[i] @ V))
                S_i = Sbus[i].real + 1j * Q_i
            Vnew = (np.conj(S_i / V[i]) - (Y[i] @ V - Y[i, i] * V[i])) / Y[i, i]
            if i in pv_set:
                Vnew = Vset[i] * Vnew / max(abs(Vnew), 1e-12)
            else:
                Vnew = V[i] + alpha * (Vnew - V[i])
            V[i] = Vnew
        F = mismatch(Y_csr, V, Sbus, pv, pq)
        hist.append(float(np.linalg.norm(F, np.inf)))
        if hist[-1] < tol:
            return V, True, it, hist
    return V, False, max_it, hist


# ---------------------------------------------------------------------------
# Aliran cabang & rugi-rugi
# ---------------------------------------------------------------------------
def branch_flows(case, V, Yf, Yt):
    """Sf, St (MVA kompleks) dan rugi-rugi tiap cabang Sloss = Sf + St."""
    bmap = _bus_map(case["bus"])
    br = case["branch"]
    f = np.array([bmap[int(b)] for b in br[:, F_BUS]], dtype=int)
    t = np.array([bmap[int(b)] for b in br[:, T_BUS]], dtype=int)
    Sf = V[f] * np.conj(Yf @ V) * case["baseMVA"]
    St = V[t] * np.conj(Yt @ V) * case["baseMVA"]
    return Sf, St, Sf + St
