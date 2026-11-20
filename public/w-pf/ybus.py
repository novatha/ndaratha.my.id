"""Modul Pembentukan Matriks Admitansi Bus (Ybus) Sparse dan Data Sistem Uji.

Modul ini mengimplementasikan perumusan matriks admitansi bus (Ybus) serta
matriks admitansi cabang sisi pengirim (Yf) dan penerima (Yt) dalam format
sparse Compressed Sparse Row (scipy.sparse.csr_matrix). Pemodelan mencakup:
  - Saluran transmisi model nominal-pi (impedansi seri R + jX dan charging B)
  - Transformator dengan off-nominal tap ratio (t) dan phase shift angle (theta_shift)
  - Admitansi shunt bus (Gs + jBs)

Modul juga menyediakan data sistem uji standar IEEE 5-bus dan IEEE 14-bus
yang siap digunakan secara mandiri.
"""

from __future__ import annotations

from typing import Dict, Tuple, TypedDict
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

# Indeks kolom matriks bus (konvensi MATPOWER v2, 0-based)
BUS_I: int = 0
BUS_TYPE: int = 1
PD: int = 2
QD: int = 3
GS: int = 4
BS: int = 5
AREA: int = 6
VM: int = 7
VA: int = 8
BASE_KV: int = 9
ZONE: int = 10
VMAX: int = 11
VMIN: int = 12

# Kode tipe bus (konvensi MATPOWER: 4 = bus terisolasi / tidak aktif)
PQ: int = 1
PV: int = 2
REF: int = 3
NONE: int = 4

# Indeks kolom matriks generator
GEN_BUS: int = 0
PG: int = 1
QG: int = 2
QMAX: int = 3
QMIN: int = 4
VG: int = 5
MBASE: int = 6
GEN_STATUS: int = 7
PMAX: int = 8
PMIN: int = 9

# Indeks kolom matriks cabang (branch)
F_BUS: int = 0
T_BUS: int = 1
BR_R: int = 2
BR_X: int = 3
BR_B: int = 4
RATE_A: int = 5
RATE_B: int = 6
RATE_C: int = 7
TAP: int = 8
SHIFT: int = 9
BR_STATUS: int = 10


class PowerSystemCase(TypedDict):
    """Struktur data kamus untuk kasus sistem tenaga listrik."""

    name: str
    baseMVA: float
    bus: NDArray[np.float64]
    gen: NDArray[np.float64]
    branch: NDArray[np.float64]


def build_ybus(
    base_mva: float,
    bus: NDArray[np.float64],
    branch: NDArray[np.float64],
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Menghitung matriks admitansi bus (Ybus) dan cabang (Yf, Yt) dalam format CSR.

    Setiap cabang ke-l antara bus f (from) dan bus t (to) dimodelkan sebagai
    jaringan nominal-pi yang terhubung seri dengan transformator ideal ber-rasio
    kompleks a = tap * exp(j * shift_rad) di sisi bus f:

        [ If ]   [ (ys + j*Bc/2) / |a|^2     -ys / conj(a) ] [ Vf ]
        [    ] = [                                         ] [    ]
        [ It ]   [       -ys / a              ys + j*Bc/2  ] [ Vt ]

    Parameters
    ----------
    base_mva : float
        Daya dasar sistem tiga fasa dalam MVA (contoh: 100.0).
    bus : NDArray[np.float64]
        Matriks data bus berdimensi (N_bus, 13). Nomor bus eksternal pada kolom
        BUS_I boleh tak-berurutan (mis. 101, 205, 9001) namun harus unik.
    branch : NDArray[np.float64]
        Matriks data saluran/transformator berdimensi (N_branch, 11).

    Returns
    -------
    y_bus : scipy.sparse.csr_matrix
        Matriks admitansi bus kompleks berukuran (N_bus, N_bus).
    y_f : scipy.sparse.csr_matrix
        Matriks admitansi cabang sisi pengirim (from) berukuran (N_branch, N_bus)
        sedemikian sehingga vektor arus pengirim If = y_f @ V.
    y_t : scipy.sparse.csr_matrix
        Matriks admitansi cabang sisi penerima (to) berukuran (N_branch, N_bus)
        sedemikian sehingga vektor arus penerima It = y_t @ V.

    Raises
    ------
    ValueError
        Jika ditemukan cabang aktif dengan impedansi seri R = X = 0.
    """
    n_bus = bus.shape[0]
    n_branch = branch.shape[0]

    # Pemetaan nomor bus eksternal ke indeks internal 0 .. n_bus-1
    bus_numbers = bus[:, BUS_I].astype(int)
    if len(set(bus_numbers.tolist())) != n_bus:
        raise ValueError("Terdapat nomor bus duplikat pada kolom BUS_I.")
    bus_map: Dict[int, int] = {num: idx for idx, num in enumerate(bus_numbers)}

    try:
        f_idx = np.array([bus_map[int(b)] for b in branch[:, F_BUS]], dtype=int)
        t_idx = np.array([bus_map[int(b)] for b in branch[:, T_BUS]], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Cabang merujuk ke nomor bus {exc.args[0]} yang tidak ada pada matriks bus."
        ) from exc

    status = branch[:, BR_STATUS].astype(np.float64)
    z_series = branch[:, BR_R] + 1j * branch[:, BR_X]
    if np.any((status > 0) & (np.abs(z_series) == 0.0)):
        raise ValueError("Cabang aktif tidak boleh memiliki impedansi R + jX = 0.")

    # Admitansi seri dan suseptansi charging saluran (hindari warning 0/0 pada cabang off)
    safe_z = np.where(status > 0, z_series, 1.0 + 0.0j)
    y_series = np.where(status > 0, status / safe_z, 0.0 + 0.0j)
    b_charging = status * branch[:, BR_B]

    # Rasio tap kompleks: jika TAP == 0 maka cabang adalah saluran (tap = 1.0)
    tap_mag = np.where(branch[:, TAP] == 0.0, 1.0, branch[:, TAP])
    shift_rad = np.deg2rad(branch[:, SHIFT])
    tap_complex = tap_mag * np.exp(1j * shift_rad)

    # Elemen matriks 2x2 per cabang
    y_tt = y_series + 1j * (b_charging / 2.0)
    y_ff = y_tt / (tap_complex * np.conj(tap_complex))
    y_ft = -y_series / np.conj(tap_complex)
    y_tf = -y_series / tap_complex

    branch_idx = np.arange(n_branch, dtype=int)

    # Matriks insidensi cabang-bus Cf dan Ct
    c_f = sp.csr_matrix(
        (np.ones(n_branch, dtype=np.float64), (branch_idx, f_idx)),
        shape=(n_branch, n_bus),
    )
    c_t = sp.csr_matrix(
        (np.ones(n_branch, dtype=np.float64), (branch_idx, t_idx)),
        shape=(n_branch, n_bus),
    )

    # Matriks admitansi cabang Yf dan Yt
    row_indices = np.concatenate([branch_idx, branch_idx])
    col_indices = np.concatenate([f_idx, t_idx])
    y_f = sp.csr_matrix(
        (np.concatenate([y_ff, y_ft]), (row_indices, col_indices)),
        shape=(n_branch, n_bus),
        dtype=np.complex128,
    )
    y_t = sp.csr_matrix(
        (np.concatenate([y_tf, y_tt]), (row_indices, col_indices)),
        shape=(n_branch, n_bus),
        dtype=np.complex128,
    )

    # Admitansi shunt pada tiap bus dalam per-unit
    y_shunt = (bus[:, GS] + 1j * bus[:, BS]) / base_mva

    # Perakitan Ybus = Cf^T * Yf + Ct^T * Yt + diag(Ysh)
    y_bus = (c_f.T @ y_f + c_t.T @ y_t + sp.diags(y_shunt, format="csr")).tocsr()
    y_bus.sum_duplicates()
    y_f.sum_duplicates()
    y_t.sum_duplicates()

    return y_bus, y_f, y_t


def get_ieee5_case() -> PowerSystemCase:
    """Mengembalikan data sistem uji klasik 5-bus (Stagg & El-Abiad / Stevenson)."""
    base_mva = 100.0
    bus = np.array(
        [
            [1, REF, 0.0, 0.0, 0.0, 0.0, 1, 1.06, 0.0, 138.0, 1, 1.06, 0.94],
            [2, PV, 20.0, 10.0, 0.0, 0.0, 1, 1.045, 0.0, 138.0, 1, 1.06, 0.94],
            [3, PQ, 45.0, 15.0, 0.0, 0.0, 1, 1.00, 0.0, 138.0, 1, 1.06, 0.94],
            [4, PQ, 40.0, 5.0, 0.0, 0.0, 1, 1.00, 0.0, 138.0, 1, 1.06, 0.94],
            [5, PQ, 60.0, 10.0, 0.0, 0.0, 1, 1.00, 0.0, 138.0, 1, 1.06, 0.94],
        ],
        dtype=np.float64,
    )
    gen = np.array(
        [
            [1, 0.0, 0.0, 150.0, -100.0, 1.060, 100.0, 1, 250.0, 0.0],
            [2, 40.0, 30.0, 50.0, -40.0, 1.045, 100.0, 1, 150.0, 0.0],
        ],
        dtype=np.float64,
    )
    branch = np.array(
        [
            [1, 2, 0.02, 0.06, 0.06, 150.0, 150.0, 150.0, 0.0, 0.0, 1],
            [1, 3, 0.08, 0.24, 0.05, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [2, 3, 0.06, 0.18, 0.04, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [2, 4, 0.06, 0.18, 0.04, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [2, 5, 0.04, 0.12, 0.03, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [3, 4, 0.01, 0.03, 0.02, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [4, 5, 0.08, 0.24, 0.05, 100.0, 100.0, 100.0, 0.98, 0.0, 1],
        ],
        dtype=np.float64,
    )
    return {
        "name": "IEEE 5-Bus",
        "baseMVA": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }


def get_ieee14_case() -> PowerSystemCase:
    """Mengembalikan data sistem uji standar IEEE 14-bus (format MATPOWER v2)."""
    base_mva = 100.0
    bus = np.array(
        [
            [1, REF, 0.0, 0.0, 0.0, 0.0, 1, 1.060, 0.00, 135.0, 1, 1.06, 0.94],
            [2, PV, 21.7, 12.7, 0.0, 0.0, 1, 1.045, -4.98, 135.0, 1, 1.06, 0.94],
            [3, PV, 94.2, 19.0, 0.0, 0.0, 1, 1.010, -12.72, 135.0, 1, 1.06, 0.94],
            [4, PQ, 47.8, -3.9, 0.0, 0.0, 1, 1.019, -10.33, 135.0, 1, 1.06, 0.94],
            [5, PQ, 7.6, 1.6, 0.0, 0.0, 1, 1.020, -8.78, 135.0, 1, 1.06, 0.94],
            [6, PV, 11.2, 7.5, 0.0, 0.0, 1, 1.070, -14.22, 135.0, 1, 1.06, 0.94],
            [7, PQ, 0.0, 0.0, 0.0, 0.0, 1, 1.062, -13.37, 135.0, 1, 1.06, 0.94],
            [8, PV, 0.0, 0.0, 0.0, 0.0, 1, 1.090, -13.36, 135.0, 1, 1.06, 0.94],
            [9, PQ, 29.5, 16.6, 0.0, 19.0, 1, 1.056, -14.94, 135.0, 1, 1.06, 0.94],
            [10, PQ, 9.0, 5.8, 0.0, 0.0, 1, 1.051, -15.10, 135.0, 1, 1.06, 0.94],
            [11, PQ, 3.5, 1.8, 0.0, 0.0, 1, 1.057, -14.79, 135.0, 1, 1.06, 0.94],
            [12, PQ, 6.1, 1.6, 0.0, 0.0, 1, 1.055, -15.07, 135.0, 1, 1.06, 0.94],
            [13, PQ, 13.5, 5.8, 0.0, 0.0, 1, 1.050, -15.16, 135.0, 1, 1.06, 0.94],
            [14, PQ, 14.9, 5.0, 0.0, 0.0, 1, 1.036, -16.04, 135.0, 1, 1.06, 0.94],
        ],
        dtype=np.float64,
    )
    gen = np.array(
        [
            [1, 232.4, -16.9, 10.0, 0.0, 1.060, 100.0, 1, 332.4, 0.0],
            [2, 40.0, 42.4, 50.0, -40.0, 1.045, 100.0, 1, 140.0, 0.0],
            [3, 0.0, 23.4, 40.0, 0.0, 1.010, 100.0, 1, 100.0, 0.0],
            [6, 0.0, 12.2, 24.0, -6.0, 1.070, 100.0, 1, 100.0, 0.0],
            [8, 0.0, 17.4, 24.0, -6.0, 1.090, 100.0, 1, 100.0, 0.0],
        ],
        dtype=np.float64,
    )
    branch = np.array(
        [
            [1, 2, 0.01938, 0.05917, 0.0528, 200.0, 200.0, 200.0, 0.0, 0.0, 1],
            [1, 5, 0.05403, 0.22304, 0.0492, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [2, 3, 0.04699, 0.19797, 0.0438, 100.0, 100.0, 100.0, 0.0, 0.0, 1],
            [2, 4, 0.05811, 0.17632, 0.0340, 80.0, 80.0, 80.0, 0.0, 0.0, 1],
            [2, 5, 0.05695, 0.17388, 0.0346, 60.0, 60.0, 60.0, 0.0, 0.0, 1],
            [3, 4, 0.06701, 0.17103, 0.0128, 40.0, 40.0, 40.0, 0.0, 0.0, 1],
            [4, 5, 0.01335, 0.04211, 0.0, 90.0, 90.0, 90.0, 0.0, 0.0, 1],
            [4, 7, 0.0, 0.20912, 0.0, 50.0, 50.0, 50.0, 0.978, 0.0, 1],
            [4, 9, 0.0, 0.55618, 0.0, 30.0, 30.0, 30.0, 0.969, 0.0, 1],
            [5, 6, 0.0, 0.25202, 0.0, 65.0, 65.0, 65.0, 0.932, 0.0, 1],
            [6, 11, 0.09498, 0.19890, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [6, 12, 0.12291, 0.25581, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [6, 13, 0.06615, 0.13027, 0.0, 30.0, 30.0, 30.0, 0.0, 0.0, 1],
            [7, 8, 0.0, 0.17615, 0.0, 30.0, 30.0, 30.0, 0.0, 0.0, 1],
            [7, 9, 0.0, 0.11001, 0.0, 45.0, 45.0, 45.0, 0.0, 0.0, 1],
            [9, 10, 0.03181, 0.08450, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [9, 14, 0.12711, 0.27038, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [10, 11, 0.08205, 0.19207, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [12, 13, 0.22092, 0.19988, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
            [13, 14, 0.17093, 0.34802, 0.0, 25.0, 25.0, 25.0, 0.0, 0.0, 1],
        ],
        dtype=np.float64,
    )
    return {
        "name": "IEEE 14-Bus",
        "baseMVA": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }


def summarize_ybus(y_bus: sp.csr_matrix, name: str = "") -> str:
    """Menyusun ringkasan properti numerik dari matriks Ybus sparse."""
    n_rows, n_cols = y_bus.shape
    density = 100.0 * y_bus.nnz / float(n_rows * n_cols)
    dense = y_bus.toarray()
    is_symmetric = bool(np.allclose(dense, dense.T, atol=1e-12))
    header = f"=== Ringkasan Ybus {name} ===" if name else "=== Ringkasan Ybus ==="
    lines = [
        header,
        f"  Format matriks         : {type(y_bus).__name__} ({y_bus.dtype})",
        f"  Dimensi                : {n_rows} x {n_cols}",
        f"  Elemen tak-nol (nnz)   : {y_bus.nnz} ({density:.2f}% densitas)",
        f"  Simetris (Y == Y^T)    : {is_symmetric}",
        f"  Y[0, 0] (bus 1 self)   : {y_bus[0, 0]:.6f} p.u.",
        f"  Y[0, 1] (mutual 1-2)   : {y_bus[0, 1]:.6f} p.u.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for getter in (get_ieee5_case, get_ieee14_case):
        case_data = getter()
        y_bus_mat, y_f_mat, y_t_mat = build_ybus(
            case_data["baseMVA"],
            case_data["bus"],
            case_data["branch"],
        )
        print(summarize_ybus(y_bus_mat, case_data["name"]))

        # Uji dampak phase-shifting transformer pada simetri Ybus
        branch_pst = case_data["branch"].copy()
        branch_pst[-1, SHIFT] = 5.0  # geser fasa 5 derajat pada cabang terakhir
        y_bus_pst, _, _ = build_ybus(case_data["baseMVA"], case_data["bus"], branch_pst)
        sym_pst = np.allclose(y_bus_pst.toarray(), y_bus_pst.toarray().T)
        print(f"  Simetris dengan PST 5° : {sym_pst}\n")
