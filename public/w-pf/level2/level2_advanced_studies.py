"""Modul Seri Lanjutan Level 2: AC-OPF, PV Hosting Capacity, dan Risk-Based N-2 Screening.

Skrip ini mendemonstrasikan tiga studi rekayasa sistem tenaga listrik tingkat lanjut
menggunakan pandapower pada sistem uji IEEE 14-bus:
  1. Optimal Power Flow (AC-OPF) vs Base Case PF: minimasi biaya pembangkitan
     kuadratik sekaligus memenuhi batas tegangan [0.95, 1.06] p.u. dan termal cabang.
  2. Studi PV Hosting Capacity: injeksi bertahap pembangkit surya (sgen) pada bus
     ujung subsistem (Bus 10, 13, 14) untuk menentukan kapasitas maksimum (MW) sebelum
     pelanggaran overvoltage (> 1.06 p.u.) atau pembebanan cabang (> 100%).
  3. Skrining Kontinjensi Ganda (N-2) Berbasis Risiko: evaluasi seluruh kombinasi
     pasangan cabang C(20, 2) = 190 skenario dengan Indeks Risiko
     RI_k = p_i * p_j * (PI_S + PI_V) serta deteksi islanding / divergensi.
"""

from __future__ import annotations

import copy
import itertools
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp

# Pastikan modul utama pada root repositori dapat diimpor
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pandapower_workflow import build_ieee14_pandapower_net  # noqa: E402


def configure_ieee14_opf_constraints(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Menyiapkan batas pembangkitan, batas tegangan, dan fungsi biaya polinomial OPF."""
    opf_net = copy.deepcopy(net)

    # Batas tegangan operasi aman dan batas termal cabang
    opf_net.bus["min_vm_pu"] = 0.95
    opf_net.bus["max_vm_pu"] = 1.06
    opf_net.line["max_loading_percent"] = 100.0
    opf_net.trafo["max_loading_percent"] = 100.0

    # Batas daya aktif & reaktif ext_grid (Slack Bus 1)
    opf_net.ext_grid["min_p_mw"] = 0.0
    opf_net.ext_grid["max_p_mw"] = 332.4
    opf_net.ext_grid["min_q_mvar"] = -40.0
    opf_net.ext_grid["max_q_mvar"] = 100.0

    # Batas daya aktif & reaktif generator sinkron/kondensor (Bus 2, 3, 6, 8)
    # Bus 2: generator utama; Bus 3, 6, 8: kondensor sinkron (diizinkan sedikit dispatch pada Bus 3 untuk studi OPF)
    p_limits = [
        (0.0, 140.0),  # Gen 0 (Bus 2)
        (0.0, 100.0),  # Gen 1 (Bus 3 - dimodelkan sbg unit termal fleksibel pada studi OPF)
        (0.0, 60.0),   # Gen 2 (Bus 6 - unit peaking lokal)
        (0.0, 0.0),    # Gen 3 (Bus 8 - kondensor sinkron murni)
    ]
    for idx, (pmin, pmax) in enumerate(p_limits):
        if idx < len(opf_net.gen):
            opf_net.gen.at[idx, "min_p_mw"] = pmin
            opf_net.gen.at[idx, "max_p_mw"] = pmax
            opf_net.gen.at[idx, "controllable"] = True

    # Fungsi biaya kuadratik C(P) = c2 * P^2 + c1 * P + c0 ($/jam)
    # Mengikuti koefisien klasik IEEE 14-bus MATPOWER gencost
    pp.create_poly_cost(
        opf_net, element=0, et="ext_grid", cp2_eur_per_mw2=0.0430, cp1_eur_per_mw=20.0, cp0_eur=0.0
    )
    gen_costs = [
        (0.2500, 20.0),  # Bus 2
        (0.0100, 40.0),  # Bus 3
        (0.0100, 40.0),  # Bus 6
        (0.0100, 40.0),  # Bus 8 (P=0)
    ]
    for idx, (c2, c1) in enumerate(gen_costs):
        if idx < len(opf_net.gen):
            pp.create_poly_cost(
                opf_net, element=idx, et="gen", cp2_eur_per_mw2=c2, cp1_eur_per_mw=c1, cp0_eur=0.0
            )

    return opf_net


def run_ac_opf_study(base_net: pp.pandapowerNet) -> Dict[str, object]:
    """Membandingkan aliran daya dasar (PF) dengan Optimal Power Flow (AC-OPF)."""
    pf_net = copy.deepcopy(base_net)
    pp.runpp(pf_net, algorithm="nr", init="flat", tolerance_mva=1e-8)

    opf_net = configure_ieee14_opf_constraints(base_net)
    pp.runopp(opf_net, verbose=False)

    # Hitung biaya pembangkitan kondisi dasar menggunakan koefisien yang sama
    p_slack_pf = float(pf_net.res_ext_grid.at[0, "p_mw"])
    p_gen_pf = pf_net.res_gen["p_mw"].to_numpy(dtype=float)
    base_cost = (
        0.0430 * p_slack_pf**2 + 20.0 * p_slack_pf
        + 0.2500 * p_gen_pf[0] ** 2 + 20.0 * p_gen_pf[0]
        + 0.0100 * p_gen_pf[1] ** 2 + 40.0 * p_gen_pf[1]
        + 0.0100 * p_gen_pf[2] ** 2 + 40.0 * p_gen_pf[2]
    )

    opf_cost = float(opf_net.res_cost)
    pf_loss = float(pf_net.res_line["pl_mw"].sum() + pf_net.res_trafo["pl_mw"].sum())
    opf_loss = float(opf_net.res_line["pl_mw"].sum() + opf_net.res_trafo["pl_mw"].sum())

    pf_dispatch = [p_slack_pf] + p_gen_pf.tolist()
    opf_dispatch = [float(opf_net.res_ext_grid.at[0, "p_mw"])] + opf_net.res_gen["p_mw"].tolist()

    return {
        "pf_net": pf_net,
        "opf_net": opf_net,
        "base_cost_usd_h": base_cost,
        "opf_cost_usd_h": opf_cost,
        "cost_saving_pct": 100.0 * (base_cost - opf_cost) / base_cost,
        "pf_loss_mw": pf_loss,
        "opf_loss_mw": opf_loss,
        "pf_dispatch_mw": pf_dispatch,
        "opf_dispatch_mw": opf_dispatch,
        "pf_vm_pu": pf_net.res_bus["vm_pu"].to_numpy(dtype=float),
        "opf_vm_pu": opf_net.res_bus["vm_pu"].to_numpy(dtype=float),
        "lmp_usd_mwh": opf_net.res_bus["lam_p"].to_numpy(dtype=float),
    }


def run_pv_hosting_capacity(
    base_net: pp.pandapowerNet,
    target_buses: List[int] | None = None,
    p_steps_mw: np.ndarray | None = None,
    v_max_limit: float = 1.06,
    loading_limit: float = 100.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Menghitung kurva PV Hosting Capacity melalui injeksi bertahap sgen (pf=1.0)."""
    if target_buses is None:
        target_buses = [10, 13, 14]
    if p_steps_mw is None:
        p_steps_mw = np.arange(0.0, 95.0, 5.0)

    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for bus_id in target_buses:
        work_net = copy.deepcopy(base_net)
        # Turunkan sedikit tegangan setpoint Bus 6/8 agar studi overvoltage di bus ujung jelas
        sgen_idx = pp.create_sgen(work_net, bus=bus_id, p_mw=0.0, q_mvar=0.0, name=f"PV_Bus{bus_id}")
        # Catat baseline V pada bus target dan max loading sistem
        hosting_limit_mw = float(p_steps_mw[-1])
        first_violation_cause = "Aman hingga batas sapuan"
        violated_found = False

        for p_pv in p_steps_mw:
            work_net.sgen.at[sgen_idx, "p_mw"] = float(p_pv)
            try:
                pp.runpp(work_net, algorithm="nr", init="flat", tolerance_mva=1e-8)
                v_target = float(work_net.res_bus.at[bus_id, "vm_pu"])
                # Pantau tegangan area beban subsistem (Bus 9-14, base < 1.057 p.u.)
                max_v_pq = float(work_net.res_bus.loc[[9, 10, 11, 12, 13, 14], "vm_pu"].max())
                max_load = float(
                    max(
                        work_net.res_line["loading_percent"].max(),
                        work_net.res_trafo["loading_percent"].max(),
                    )
                )
                p_loss = float(work_net.res_line["pl_mw"].sum() + work_net.res_trafo["pl_mw"].sum())
                over_v = v_target > v_max_limit or max_v_pq > v_max_limit
                over_th = max_load > loading_limit

                if (over_v or over_th) and not violated_found:
                    violated_found = True
                    hosting_limit_mw = max(0.0, float(p_pv) - float(p_steps_mw[1] - p_steps_mw[0]))
                    if over_v and over_th:
                        first_violation_cause = "Overvoltage & Overload Termal"
                    elif over_v:
                        first_violation_cause = f"Overvoltage (> {v_max_limit:.2f} p.u.)"
                    else:
                        first_violation_cause = f"Overload Termal (> {loading_limit:.0f}%)"

                records.append(
                    {
                        "bus_id": bus_id,
                        "pv_mw": float(p_pv),
                        "v_bus_pu": v_target,
                        "max_v_pq_pu": max_v_pq,
                        "max_loading_pct": max_load,
                        "p_loss_mw": p_loss,
                    }
                )
            except pp.LoadflowNotConverged:
                if not violated_found:
                    violated_found = True
                    hosting_limit_mw = max(0.0, float(p_pv) - float(p_steps_mw[1] - p_steps_mw[0]))
                    first_violation_cause = "Divergen"
                break

        summary_rows.append(
            {
                "bus_target": f"Bus {bus_id}",
                "hosting_capacity_mw": hosting_limit_mw,
                "faktor_pembatas": first_violation_cause,
            }
        )

    return pd.DataFrame(records), pd.DataFrame(summary_rows)


def run_n2_risk_screening(base_net: pp.pandapowerNet) -> pd.DataFrame:
    """Menjalankan skrining kontinjensi ganda (N-2) berbasis risiko untuk 190 pasangan cabang.

    Probabilitas gangguan tunggal p_i dimodelkan proporsional terhadap reaktansi seri
    (proksi panjang saluran) dengan rentang realistis 0.002 hingga 0.012 per periode,
    dan probabilitas simultan p_{ij} = p_i * p_j.
    """
    work_net = copy.deepcopy(base_net)
    pp.runpp(work_net, algorithm="nr", init="flat", tolerance_mva=1e-8)
    base_vm = work_net.res_bus["vm_pu"].copy()
    base_va = work_net.res_bus["va_degree"].copy()

    # Bangun katalog 20 cabang (17 saluran + 3 transformator) dengan probabilitas outage sintetis
    elements: List[Tuple[str, int, str, float]] = []
    for idx, row in work_net.line.iterrows():
        # Proksi panjang saluran dari x_ohm_per_km -> p_outage ~ [0.0025, 0.0110]
        p_out = float(np.clip(0.002 + 0.00018 * row["x_ohm_per_km"], 0.0025, 0.0110))
        elements.append(("line", int(idx), str(row["name"]), p_out))
    for idx, row in work_net.trafo.iterrows():
        # Transformator daya umumnya memiliki laju gangguan sedikit lebih rendah (~0.0030)
        elements.append(("trafo", int(idx), str(row["name"]), 0.0030))

    rows: List[Dict[str, object]] = []
    for (t1, i1, n1, p1), (t2, i2, n2, p2) in itertools.combinations(elements, 2):
        tbl1 = work_net.line if t1 == "line" else work_net.trafo
        tbl2 = work_net.line if t2 == "line" else work_net.trafo
        p_pair = p1 * p2
        pair_label = f"{n1} + {n2}"

        tbl1.at[i1, "in_service"] = False
        tbl2.at[i2, "in_service"] = False
        try:
            work_net.res_bus["vm_pu"] = base_vm.values
            work_net.res_bus["va_degree"] = base_va.values
            try:
                pp.runpp(work_net, algorithm="nr", init="results", max_iteration=25, tolerance_mva=1e-6)
                vm = work_net.res_bus["vm_pu"]
                isolated = [int(b) for b in vm[vm.isna()].index]

                line_load = work_net.res_line["loading_percent"].fillna(0.0).to_numpy(dtype=float)
                trafo_load = work_net.res_trafo["loading_percent"].fillna(0.0).to_numpy(dtype=float)
                all_load = np.concatenate([line_load, trafo_load])
                max_load = float(np.max(all_load)) if all_load.size else 0.0
                n_over = int(np.sum(all_load > 100.0))

                vm_valid = vm.dropna().to_numpy(dtype=float)
                v_min = float(np.min(vm_valid)) if vm_valid.size else 0.0
                n_uv = int(np.sum(vm_valid < 0.95))

                # Performance Index komposit (eksponen 2n = 4)
                pi_s = float(np.sum(0.25 * (all_load / 100.0) ** 4))
                pi_v = float(np.sum(0.25 * ((vm_valid - 1.0) / 0.05) ** 4))
                # Penalti tambahan jika terjadi islanding (hilangnya suplai bus)
                island_penalty = 250.0 * len(isolated)
                pi_total = pi_s + pi_v + island_penalty
                status = f"ISLANDING ({len(isolated)} bus)" if isolated else "KONVERGEN"
            except pp.LoadflowNotConverged:
                status = "DIVERGEN"
                max_load = float("nan")
                n_over = -1
                v_min = float("nan")
                n_uv = -1
                pi_total = 1000.0  # Penalti keparahan ekstrem untuk kolaps tegangan
        finally:
            tbl1.at[i1, "in_service"] = True
            tbl2.at[i2, "in_service"] = True

        risk_index = p_pair * pi_total
        rows.append(
            {
                "kontinjensi_n2": pair_label,
                "status": status,
                "probabilitas_1e5": p_pair * 1e5,
                "max_loading_pct": max_load,
                "n_overload": n_over,
                "v_min_pu": v_min,
                "n_undervoltage": n_uv,
                "pi_total": pi_total,
                "risk_index_1e3": risk_index * 1e3,
            }
        )

    df = pd.DataFrame(rows).sort_values("risk_index_1e3", ascending=False).reset_index(drop=True)
    return df


def plot_level2_figures(
    opf_res: Dict[str, object],
    pv_df: pd.DataFrame,
    n2_df: pd.DataFrame,
    figures_dir: str,
) -> None:
    """Menyimpan 3 grafik publikasi ilmiah Level 2 beresolusi 300 DPI."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Grafik Perbandingan AC-OPF vs Base Case PF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=300)
    gen_labels = ["Bus 1\n(Slack)", "Bus 2\n(Gen)", "Bus 3\n(Gen)", "Bus 6\n(Gen)", "Bus 8\n(Sync C)"]
    x = np.arange(len(gen_labels))
    w = 0.36
    ax1.bar(x - w / 2, opf_res["pf_dispatch_mw"], w, label="Base Case PF", color="#475569")
    ax1.bar(x + w / 2, opf_res["opf_dispatch_mw"], w, label="AC-OPF Optimal", color="#003366")
    ax1.set_xticks(x)
    ax1.set_xticklabels(gen_labels)
    ax1.set_ylabel("Daya Aktif Pembangkitan (MW)")
    ax1.set_title("Redistribusi Pembangkitan: Base Case vs AC-OPF")
    ax1.legend()

    buses = np.arange(1, 15)
    ax2.plot(buses, opf_res["pf_vm_pu"], "o--", color="#64748b", label="Base Case PF")
    ax2.plot(buses, opf_res["opf_vm_pu"], "s-", color="#006644", label="AC-OPF (0,95–1,06 p.u.)")
    ax2.axhline(1.06, color="#dc2626", linestyle=":", linewidth=1.3, label="Batas Atas 1,06 p.u.")
    ax2.axhline(0.95, color="#d97706", linestyle=":", linewidth=1.3, label="Batas Bawah 0,95 p.u.")
    ax2.set_xticks(buses)
    ax2.set_xlabel("Nomor Bus IEEE 14-Bus")
    ax2.set_ylabel("Magnitudo Tegangan |V| (p.u.)")
    ax2.set_title(
        f"Profil Tegangan (Hemat Biaya {opf_res['cost_saving_pct']:.1f}% | Rugi {opf_res['opf_loss_mw']:.2f} MW)"
    )
    ax2.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "level2_opf_comparison.png"), dpi=300)
    plt.close(fig)

    # 2. Grafik PV Hosting Capacity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=300)
    colors = {10: "#003366", 13: "#0f766e", 14: "#b45309"}
    markers = {10: "o", 13: "s", 14: "^"}
    for bus_id, grp in pv_df.groupby("bus_id"):
        ax1.plot(
            grp["pv_mw"], grp["v_bus_pu"], marker=markers[bus_id], color=colors[bus_id],
            linewidth=1.8, markersize=4.5, label=f"Injeksi PLTS di Bus {bus_id}"
        )
        ax2.plot(
            grp["pv_mw"], grp["p_loss_mw"], marker=markers[bus_id], color=colors[bus_id],
            linewidth=1.8, markersize=4.5, label=f"Injeksi PLTS di Bus {bus_id}"
        )
    ax1.axhline(1.06, color="#dc2626", linestyle="--", linewidth=1.4, label="Batas Overvoltage (1,06 p.u.)")
    ax1.set_xlabel("Kapasitas Injeksi PLTS (MW)")
    ax1.set_ylabel("Tegangan Bus Injeksi |V| (p.u.)")
    ax1.set_title("Kurva Kenaikan Tegangan & Batas PV Hosting Capacity")
    ax1.legend(fontsize=8.5)

    ax2.set_xlabel("Kapasitas Injeksi PLTS (MW)")
    ax2.set_ylabel("Rugi Daya Aktif Sistem Total (MW)")
    ax2.set_title("Kurva Berbentuk-U Rugi Transmisi vs Penetrasi PLTS")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "level2_pv_hosting_capacity.png"), dpi=300)
    plt.close(fig)

    # 3. Grafik Top 15 Risiko Kontinjensi N-2
    top15 = n2_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    bar_colors = [
        "#991b1b" if "DIVERGEN" in st else "#b45309" if "ISLAND" in st else "#003366"
        for st in top15["status"]
    ]
    ax.barh(top15["kontinjensi_n2"], top15["risk_index_1e3"], color=bar_colors)
    for y_idx, (_, row) in enumerate(top15.iterrows()):
        tag = f" {row['risk_index_1e3']:.2f} ({row['status']})"
        ax.text(row["risk_index_1e3"] + 0.15, y_idx, tag, va="center", fontsize=8)
    ax.set_xlabel("Indeks Risiko Kontinjensi Ganda RI × 10³")
    ax.set_title("15 Skenario Kontinjensi N-2 dengan Risiko Tertinggi (dari 190 Pasangan Cabang)")
    ax.set_xlim(0, top15["risk_index_1e3"].max() * 1.38)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "level2_n2_risk_top15.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    net14 = build_ieee14_pandapower_net()

    print("=== 1. STUDI AC-OPF IEEE 14-BUS ===")
    opf_res = run_ac_opf_study(net14)
    print(f"Biaya Base Case PF : ${opf_res['base_cost_usd_h']:.2f}/jam | Rugi: {opf_res['pf_loss_mw']:.3f} MW")
    print(f"Biaya AC-OPF       : ${opf_res['opf_cost_usd_h']:.2f}/jam | Rugi: {opf_res['opf_loss_mw']:.3f} MW")
    print(f"Penghematan Biaya  : {opf_res['cost_saving_pct']:.2f}%")
    print(f"Dispatch OPF (MW)  : {[round(x, 2) for x in opf_res['opf_dispatch_mw']]}")
    print(f"LMP min/max ($/MWh): {opf_res['lmp_usd_mwh'].min():.2f} / {opf_res['lmp_usd_mwh'].max():.2f}")

    print("\n=== 2. STUDI PV HOSTING CAPACITY (BUS 10, 13, 14) ===")
    pv_df, pv_summary = run_pv_hosting_capacity(net14)
    print(pv_summary.to_string(index=False))

    print("\n=== 3. SKRINING KONTINJENSI N-2 BERBASIS RISIKO (190 PASANGAN) ===")
    n2_df = run_n2_risk_screening(net14)
    out_dir = os.path.join(os.path.dirname(__file__), "hasil")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "n2_risk_ranking.csv")
    n2_df.to_csv(csv_path, index=False)
    print(f"Total kombinasi N-2: {len(n2_df)} | Divergen: {(n2_df['status'] == 'DIVERGEN').sum()} | Islanding: {n2_df['status'].str.contains('ISLANDING').sum()}")
    print(n2_df.head(10).to_string(index=False))

    figures_path = os.path.join(ROOT_DIR, "figures")
    plot_level2_figures(opf_res, pv_df, n2_df, figures_path)
    print(f"\nGrafik Level 2 disimpan di: {figures_path}")
    print(f"CSV ranking N-2 disimpan di: {csv_path}")
