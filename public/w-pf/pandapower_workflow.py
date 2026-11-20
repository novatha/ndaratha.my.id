"""Modul Pemodelan Jaringan pandapower, Otomatisasi Skrining Kontinjensi N-1, dan Visualisasi.

Modul ini mengimplementasikan alur kerja skala industri pada sistem uji IEEE 14-bus:
  - Pembangunan model jaringan pandapower lengkap dengan batas arus/daya termal
    cabang yang konsisten dengan kapasitas MVA (`RATE_A`).
  - Otomatisasi skrining kontinjensi N-1 cepat (tanpa `copy.deepcopy` berulang di
    dalam loop) untuk seluruh saluran transmisi dan transformator, dengan pencatatan
    persentase pembebanan termal (% loading) dan deteksi bus undervoltage (< 0.95 p.u.).
  - Visualisasi publikasi ilmiah menggunakan Matplotlib (dengan dukungan opsional
    Seaborn bila terinstal):
      1. Kurva konvergensi mismatch vs iterasi (perbandingan IEEE 5-bus & 14-bus).
      2. Heatmap matriks loading cabang (%) di seluruh skenario kontinjensi N-1.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.converter as pc

from ybus import get_ieee5_case, get_ieee14_case
from newton_raphson import solve_newton_raphson


def build_ieee14_pandapower_net() -> pp.pandapowerNet:
    """Membangun jaringan IEEE 14-bus di pandapower dengan batas termal konsisten.

    Konversi dilakukan dari struktur MATPOWER/PYPOWER (`get_ieee14_case`) agar
    parameter impedansi, tap transformator, dan rating MVA cabang (`RATE_A`) selaras
    antara solver from-scratch dan pandapower.

    Returns
    -------
    net : pp.pandapowerNet
        Objek jaringan pandapower IEEE 14-bus siap simulasi.
    """
    case = get_ieee14_case()
    ppc_dict: Dict[str, object] = {
        "version": "2",
        "baseMVA": case["baseMVA"],
        "bus": case["bus"].copy(),
        "gen": case["gen"].copy(),
        "branch": case["branch"].copy(),
    }
    net = pc.from_ppc(ppc_dict, f_hz=60.0, validate_conversion=False)
    net.bus["name"] = [f"Bus {int(b)}" for b in net.bus.index]

    net.line["name"] = [
        f"L{int(row['from_bus'])}-{int(row['to_bus'])}"
        for _, row in net.line.iterrows()
    ]
    net.trafo["name"] = [
        f"T{int(row['hv_bus'])}-{int(row['lv_bus'])}"
        for _, row in net.trafo.iterrows()
    ]
    return net


def _extract_branch_loading_series(
    net: pp.pandapowerNet,
    include_trafo: bool,
) -> pd.Series:
    """Mengekstrak seri persentase loading cabang (saluran, dan opsional trafo)."""
    line_s = pd.Series(
        net.res_line["loading_percent"].values,
        index=list(net.line["name"]),
        dtype=float,
    )
    if not include_trafo or net.trafo.empty:
        return line_s
    trafo_s = pd.Series(
        net.res_trafo["loading_percent"].values,
        index=list(net.trafo["name"]),
        dtype=float,
    )
    return pd.concat([line_s, trafo_s])


def run_n1_line_contingency_screening(
    net: pp.pandapowerNet,
    undervoltage_threshold_pu: float = 0.95,
    thermal_limit_percent: float = 100.0,
    include_trafo_outages: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Menjalankan otomatisasi skrining kontinjensi N-1 secara efisien (in-place toggle).

    Alih-alih menyalin objek jaringan dengan `copy.deepcopy` di setiap iterasi,
    fungsi ini hanya menyalin jaringan satu kali di awal, menogel status `in_service`
    di dalam blok `try ... finally`, serta memulihkan tebakan awal tegangan ke solusi
    Base Case sebelum setiap eksekusi aliran daya hangat (`init='results'`).

    Parameters
    ----------
    net : pp.pandapowerNet
        Model jaringan pandapower kondisi dasar (base case).
    undervoltage_threshold_pu : float, default=0.95
        Ambang batas bawah tegangan operasi normal/darurat dalam per-unit.
    thermal_limit_percent : float, default=100.0
        Ambang batas pembebanan termal cabang dalam persen kapasitas pengenal.
    include_trafo_outages : bool, default=True
        Jika True, mencakup kontinjensi pelepasan transformator selain saluran transmisi.

    Returns
    -------
    summary_df : pd.DataFrame
        Tabel ringkasan tiap kontinjensi N-1 terurut berdasarkan loading tertinggi.
    loading_matrix_df : pd.DataFrame
        Matriks berdimensi (N_contingency + 1, N_branch) berisi `loading_percent`.
    voltage_matrix_df : pd.DataFrame
        Matriks berdimensi (N_contingency + 1, N_bus) berisi profil `vm_pu` bus.
    """
    work_net = copy.deepcopy(net)
    pp.runpp(work_net, algorithm="nr", init="flat", tolerance_mva=1e-8)

    base_vm = work_net.res_bus["vm_pu"].copy()
    base_va = work_net.res_bus["va_degree"].copy()
    bus_labels: List[str] = [f"Bus {int(b)}" for b in work_net.bus.index]

    loading_records: Dict[str, pd.Series] = {
        "Base Case": _extract_branch_loading_series(work_net, include_trafo_outages)
    }
    voltage_records: Dict[str, pd.Series] = {
        "Base Case": pd.Series(base_vm.values, index=bus_labels, dtype=float)
    }
    summary_rows: List[Dict[str, object]] = []

    contingencies: List[Tuple[str, int, str]] = [
        ("line", int(idx), str(work_net.line.at[idx, "name"]))
        for idx in work_net.line.index
    ]
    if include_trafo_outages:
        contingencies.extend(
            [
                ("trafo", int(idx), str(work_net.trafo.at[idx, "name"]))
                for idx in work_net.trafo.index
            ]
        )

    for elem_table, elem_idx, out_name in contingencies:
        work_net[elem_table].at[elem_idx, "in_service"] = False
        try:
            work_net.res_bus["vm_pu"] = base_vm
            work_net.res_bus["va_degree"] = base_va
            pp.runpp(work_net, algorithm="nr", init="results", max_iteration=30)
            converged = bool(work_net.converged)
        except pp.LoadflowNotConverged:
            converged = False
        finally:
            work_net[elem_table].at[elem_idx, "in_service"] = True

        if not converged:
            summary_rows.append(
                {
                    "outage_branch": out_name,
                    "status": "DIVERGEN",
                    "max_loading_percent": np.nan,
                    "critical_branch": "-",
                    "num_overloaded": 0,
                    "overloaded_branches": [],
                    "min_vm_pu": np.nan,
                    "min_vm_bus": "-",
                    "undervoltage_buses": [],
                    "isolated_buses": [],
                }
            )
            continue

        branch_loading = _extract_branch_loading_series(
            work_net, include_trafo_outages
        )
        branch_loading.at[out_name] = np.nan
        loading_records[f"Out {out_name}"] = branch_loading

        bus_vm = pd.Series(
            work_net.res_bus["vm_pu"].values, index=bus_labels, dtype=float
        )
        voltage_records[f"Out {out_name}"] = bus_vm

        isolated = bus_vm[bus_vm.isna()].index.tolist()
        valid_loading = branch_loading.dropna()
        max_load = float(valid_loading.max())
        crit_branch = str(valid_loading.idxmax())
        overloaded = valid_loading[valid_loading > thermal_limit_percent].index.tolist()

        valid_vm = bus_vm.dropna()
        min_vm = float(valid_vm.min())
        min_vm_bus = str(valid_vm.idxmin())
        uv_buses = valid_vm[valid_vm < undervoltage_threshold_pu].index.tolist()

        status_str = "ISLANDING" if isolated else "KONVERGEN"
        summary_rows.append(
            {
                "outage_branch": out_name,
                "status": status_str,
                "max_loading_percent": round(max_load, 2),
                "critical_branch": crit_branch,
                "num_overloaded": len(overloaded),
                "overloaded_branches": overloaded,
                "min_vm_pu": round(min_vm, 4),
                "min_vm_bus": min_vm_bus,
                "undervoltage_buses": uv_buses,
                "isolated_buses": isolated,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(by="max_loading_percent", ascending=False)
        .reset_index(drop=True)
    )
    loading_matrix_df = pd.DataFrame(loading_records).T
    voltage_matrix_df = pd.DataFrame(voltage_records).T
    return summary_df, loading_matrix_df, voltage_matrix_df


def plot_convergence_curve(
    histories: Dict[str, List[float]],
    output_path: str,
    tol: float = 1e-8,
) -> str:
    """Membuat plot kurva konvergensi norm-infinit mismatch terhadap iterasi NR."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    markers = ["o", "s", "^", "D"]
    for idx, (label, hist) in enumerate(histories.items()):
        iterations = np.arange(len(hist))
        ax.semilogy(
            iterations,
            hist,
            marker=markers[idx % len(markers)],
            linewidth=2.0,
            markersize=6,
            label=f"{label} ({len(hist) - 1} iterasi)",
        )

    ax.axhline(
        tol,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Ambang toleransi ({tol:.0e} p.u.)",
    )
    ax.set_xlabel("Iterasi Newton-Raphson", fontsize=10)
    ax.set_ylabel(r"Maksimum Mismatch $\|\Delta F\|_\infty$ (p.u.)", fontsize=10)
    ax.set_title("Kurva Konvergensi Kuadratik Algoritma Newton-Raphson Polar", fontsize=11)
    ax.set_xticks(range(max(len(h) for h in histories.values())))
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_line_loading_heatmap(
    loading_matrix_df: pd.DataFrame,
    output_path: str,
    vmax: float = 140.0,
) -> str:
    """Membuat visualisasi heatmap pembebanan termal cabang (%) pada kontinjensi N-1.

    Mendukung `seaborn.heatmap` (jika terinstal) maupun `matplotlib` murni dengan
    anotasi konsisten (termasuk label 'OUT' pada elemen yang sedang lepas).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = loading_matrix_df.values.astype(float)
    row_labels = list(loading_matrix_df.index)
    col_labels = list(loading_matrix_df.columns)

    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.set_facecolor("#d9d9d9")

    try:
        import seaborn as sns  # type: ignore

        sns.heatmap(
            loading_matrix_df,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Loading Cabang (%)"},
            linewidths=0.4,
            linecolor="white",
        )
        # Tambahkan label 'OUT' eksplisit pada sel NaN agar konsisten dengan judul
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "OUT",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="#555555",
                    )
    except ImportError:
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="#d9d9d9")
        masked_data = np.ma.masked_invalid(data)
        im = ax.imshow(masked_data, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8.5)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(
                        j, i, "OUT", ha="center", va="center", fontsize=6.5, color="#555555"
                    )
                else:
                    txt_color = "white" if val >= 0.7 * vmax else "black"
                    weight = "bold" if val > 100.0 else "normal"
                    ax.text(
                        j,
                        i,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=txt_color,
                        fontweight=weight,
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label("Loading Cabang (%)", fontsize=10)

    ax.set_xlabel("Cabang Transmisi (Saluran & Transformator) yang Dimonitor", fontsize=10)
    ax.set_ylabel("Skenario Operasi (Base Case & Outage N-1)", fontsize=10)
    ax.set_title(
        "Heatmap Pembebanan Termal Cabang (%) pada Skrining Kontinjensi N-1 IEEE 14-Bus\n"
        "(Kotak 'OUT' = elemen lepas; nilai > 100% menandakan overload termal)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Jalankan solver Newton-Raphson from-scratch untuk mendapatkan riwayat konvergensi
    res_ieee5 = solve_newton_raphson(get_ieee5_case(), tol=1e-9, enforce_q_limits=False)
    res_ieee14 = solve_newton_raphson(get_ieee14_case(), tol=1e-9, enforce_q_limits=False)

    conv_fig_path = plot_convergence_curve(
        {
            "IEEE 5-Bus": res_ieee5.mismatch_history,
            "IEEE 14-Bus": res_ieee14.mismatch_history,
        },
        output_path=os.path.join(output_dir, "nr_convergence_curve.png"),
        tol=1e-9,
    )

    # 2. Bangun model pandapower IEEE 14-Bus & verifikasi konsistensi dengan NR from-scratch
    net14 = build_ieee14_pandapower_net()
    pp.runpp(net14, algorithm="nr", init="flat", tolerance_mva=1e-9)
    max_dv = float(np.max(np.abs(net14.res_bus["vm_pu"].values - res_ieee14.vm_pu)))
    print("======================================================================")
    print("VALIDASI SILANG SOLVER FROM-SCRATCH vs PANDAPOWER (IEEE 14-BUS)")
    print("======================================================================")
    print(f"Maksimum selisih |V|         : {max_dv:.3e} p.u.")
    print(
        f"Total rugi aktif pandapower  : "
        f"{net14.res_line['pl_mw'].sum() + net14.res_trafo['pl_mw'].sum():.4f} MW\n"
    )

    # 3. Skrining Kontinjensi N-1 (17 saluran + 3 transformator) pada beban normal
    summary_normal, loading_mat, _ = run_n1_line_contingency_screening(
        net14,
        undervoltage_threshold_pu=0.95,
        thermal_limit_percent=100.0,
        include_trafo_outages=True,
    )
    print("======================================================================")
    print("HASIL SKRINING KONTINJENSI N-1 CABANG (IEEE 14-BUS, BEBAN NORMAL)")
    print("======================================================================")
    cols = [
        "outage_branch",
        "status",
        "max_loading_percent",
        "critical_branch",
        "num_overloaded",
        "min_vm_pu",
        "min_vm_bus",
        "undervoltage_buses",
    ]
    print(summary_normal[cols].to_string(index=False))

    # 4. Uji skenario beban berat (1.6x beban dasar) untuk deteksi undervoltage < 0.95 p.u.
    net14_heavy = copy.deepcopy(net14)
    net14_heavy.load["p_mw"] *= 1.60
    net14_heavy.load["q_mvar"] *= 1.60
    summary_heavy, _, _ = run_n1_line_contingency_screening(
        net14_heavy,
        undervoltage_threshold_pu=0.95,
        thermal_limit_percent=100.0,
        include_trafo_outages=True,
    )
    uv_cases = summary_heavy[summary_heavy["undervoltage_buses"].map(len) > 0]
    print("\n--- Kontinjensi dengan Bus Undervoltage (< 0.95 p.u.) pada Beban Tinggi (1.6x) ---")
    if not uv_cases.empty:
        print(
            uv_cases[
                ["outage_branch", "min_vm_pu", "min_vm_bus", "undervoltage_buses"]
            ].to_string(index=False)
        )

    # 5. Simpan heatmap pembebanan cabang
    heatmap_path = plot_line_loading_heatmap(
        loading_mat,
        output_path=os.path.join(output_dir, "n1_line_loading_heatmap.png"),
    )
    print("\nVisualisasi berhasil disimpan:")
    print(f"  1. {conv_fig_path}")
    print(f"  2. {heatmap_path}")
