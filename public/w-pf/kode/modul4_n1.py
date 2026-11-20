"""
modul4_n1.py — Analisis kontinjensi N-1 otomatis (IEEE 14-bus, pandapower)
  * Outage setiap saluran & trafo satu per satu
  * Deteksi overload cabang & undervoltage/overvoltage bus, islanding, divergensi
  * Indeks keparahan (Performance Index) & ranking
  * Visualisasi: heatmap tegangan, bar loading, diagram topologi
"""
import os
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandapower as pp

from modul3_pandapower import build_net

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "hasil")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)

V_MIN, V_MAX, LOAD_MAX = 0.95, 1.10, 100.0   # kriteria keamanan (dapat disesuaikan)
RATING_FACTOR = 1.35                          # rating sintetis = 1.35 × aliran base case
MIN_RATING = 25.0                             # rating minimum sintetis (MVA)

# Posisi skematik IEEE 14-bus untuk diagram topologi
POS = {1: (0, 4), 2: (1.2, 2.2), 3: (3.4, 1.0), 4: (3.3, 3.0), 5: (1.3, 4.0),
       6: (1.4, 5.6), 7: (4.6, 3.6), 8: (5.6, 3.6), 9: (4.6, 4.8), 10: (3.8, 5.7),
       11: (2.7, 6.0), 12: (0.6, 7.0), 13: (2.0, 7.2), 14: (4.6, 6.8)}


def branch_table(net):
    """Satu tabel cabang terpadu (saluran + trafo) dengan |S| sisi-kirim & sisi-terima."""
    ln = pd.DataFrame({"tipe": "line", "idx": net.line.index, "f": net.line.from_bus.values,
                       "t": net.line.to_bus.values,
                       "S_mva": np.maximum(np.hypot(net.res_line.p_from_mw, net.res_line.q_from_mvar),
                                           np.hypot(net.res_line.p_to_mw, net.res_line.q_to_mvar))})
    tr = pd.DataFrame({"tipe": "trafo", "idx": net.trafo.index, "f": net.trafo.hv_bus.values,
                       "t": net.trafo.lv_bus.values,
                       "S_mva": np.maximum(np.hypot(net.res_trafo.p_hv_mw, net.res_trafo.q_hv_mvar),
                                           np.hypot(net.res_trafo.p_lv_mw, net.res_trafo.q_lv_mvar))})
    df = pd.concat([ln, tr], ignore_index=True)
    df["nama"] = [f"{'L' if r.tipe == 'line' else 'T'}{r.f}-{r.t}" for r in df.itertuples()]
    return df


def run_n1(net, rating):
    results, vm_matrix = [], {}
    work_net = copy.deepcopy(net)
    pp.runpp(work_net, algorithm="nr", init="flat", tolerance_mva=1e-8)
    base_vm = work_net.res_bus["vm_pu"].copy()
    base_va = work_net.res_bus["va_degree"].copy()

    base_cases = [("line", i) for i in work_net.line.index] + [("trafo", i) for i in work_net.trafo.index]
    for tipe, idx in base_cases:
        f = work_net.line.at[idx, "from_bus"] if tipe == "line" else work_net.trafo.at[idx, "hv_bus"]
        t = work_net.line.at[idx, "to_bus"] if tipe == "line" else work_net.trafo.at[idx, "lv_bus"]
        name = f"{'L' if tipe == 'line' else 'T'}{f}-{t}"
        work_net[tipe].at[idx, "in_service"] = False
        try:
            work_net.res_bus["vm_pu"] = base_vm
            work_net.res_bus["va_degree"] = base_va
            pp.runpp(work_net, algorithm="nr", init="results", max_iteration=30)
            converged = bool(work_net.converged)
        except pp.LoadflowNotConverged:
            converged = False
        finally:
            work_net[tipe].at[idx, "in_service"] = True

        if not converged:
            results.append({"kontinjensi": name, "status": "DIVERGEN"})
            continue
        vm = work_net.res_bus.vm_pu.copy()
        island = vm.isna().any()
        bt = branch_table(work_net)
        bt["loading_%"] = 100 * bt.S_mva / bt.nama.map(rating)
        bt = bt[bt.nama != name]
        over = bt[bt["loading_%"] > LOAD_MAX]
        uv = vm[vm < V_MIN]; ov = vm[vm > V_MAX]
        # Performance Index gabungan (n = 2): PI = Σ (S/Smax)^4 + Σ (ΔV/ΔVlim)^4
        pi_s = ((bt["loading_%"] / 100) ** 4).sum()
        dv = (vm.dropna() - 1.0) / ((V_MAX - V_MIN) / 2)
        pi_v = (dv ** 4).sum()
        results.append({"kontinjensi": name, "status": "ISLANDING" if island else "OK",
                        "bus_terisolasi": list(vm[vm.isna()].index),
                        "max_loading_%": bt["loading_%"].max(),
                        "cabang_terberat": bt.loc[bt["loading_%"].idxmax(), "nama"],
                        "n_overload": len(over), "overload": list(over.nama),
                        "Vmin_pu": vm.min(), "bus_Vmin": vm.idxmin(),
                        "n_undervoltage": len(uv), "n_overvoltage": len(ov),
                        "PI": pi_s + pi_v})
        vm_matrix[name] = vm
    return pd.DataFrame(results), pd.DataFrame(vm_matrix).T


def plot_topology(net, bt, vm, title, fname):
    """Diagram topologi: warna node = |V|, warna/tebal garis = loading cabang."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap_e = plt.cm.RdYlGn_r
    for r in bt.itertuples():
        L = getattr(r, "loading")
        x = [POS[r.f][0], POS[r.t][0]]; y = [POS[r.f][1], POS[r.t][1]]
        if np.isnan(L):
            ax.plot(x, y, ls=":", c="grey", lw=1.5)
        else:
            ax.plot(x, y, c=cmap_e(min(L, 120) / 120), lw=1.5 + 3 * min(L, 120) / 120,
                    ls="--" if r.tipe == "trafo" else "-")
            ax.text(np.mean(x), np.mean(y), f"{L:.0f}%", fontsize=7, ha="center",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    nodes = list(net.bus.index)
    sc = ax.scatter([POS[b][0] for b in nodes], [POS[b][1] for b in nodes],
                    c=[vm.get(b, np.nan) for b in nodes], cmap="coolwarm_r", vmin=0.94, vmax=1.10,
                    s=420, edgecolors="k", zorder=3)
    gen_bus = set(net.gen.bus) | set(net.ext_grid.bus)
    for b in nodes:
        ax.text(*POS[b], str(b), ha="center", va="center", fontsize=9, weight="bold", zorder=4,
                color="white" if b in gen_bus else "black")
    plt.colorbar(sc, ax=ax, label="|V| (p.u.)", shrink=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap_e, norm=plt.Normalize(0, 120))
    plt.colorbar(sm, ax=ax, label="Loading cabang (%)", shrink=0.8, location="left")
    ax.set_title(title + "\n(label putih = bus generator; garis putus = trafo; titik-titik = outage)",
                 fontsize=10)
    ax.axis("off"); plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()


if __name__ == "__main__":
    net = build_net()
    pp.runpp(net, algorithm="nr")
    base = branch_table(net)
    # Data IEEE 14-bus tidak memuat rating termal -> rating sintetis (dibulatkan ke 5 MVA)
    base["rating_mva"] = np.maximum(np.ceil(RATING_FACTOR * base.S_mva / 5) * 5, MIN_RATING)
    rating = dict(zip(base.nama, base.rating_mva))
    base["loading"] = 100 * base.S_mva / base.rating_mva
    print("Rating sintetis (MVA):", rating)

    df, vm_mat = run_n1(net, rating)
    df = df.sort_values("PI", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print("\n=== Ranking kontinjensi N-1 (berdasarkan PI) ===")
    print(df[["kontinjensi", "status", "max_loading_%", "cabang_terberat", "n_overload",
              "Vmin_pu", "bus_Vmin", "n_undervoltage", "PI"]].round(3).to_string())
    df.to_csv(os.path.join(OUT, "n1_ranking.csv"), index=False)
    print("\nTabel disimpan: hasil/n1_ranking.csv")

    # --- Heatmap tegangan (kontinjensi × bus) ---
    vm_sorted = vm_mat.loc[[k for k in df.kontinjensi if k in vm_mat.index]]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(vm_sorted.values.astype(float), cmap="RdYlBu", vmin=0.94, vmax=1.10, aspect="auto")
    ax.set_xticks(range(vm_sorted.shape[1])); ax.set_xticklabels(vm_sorted.columns)
    ax.set_yticks(range(vm_sorted.shape[0])); ax.set_yticklabels(vm_sorted.index)
    for i in range(vm_sorted.shape[0]):
        for j in range(vm_sorted.shape[1]):
            v = vm_sorted.values[i, j]
            ax.text(j, i, "isl" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("Bus"); ax.set_ylabel("Kontinjensi (urut PI menurun)")
    ax.set_title("Heatmap Profil Tegangan pada Kontinjensi N-1")
    plt.colorbar(im, label="|V| (p.u.)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "heatmap_tegangan_n1.png"), dpi=200); plt.close()

    # --- Bar loading maksimum per kontinjensi ---
    ok = df[df.status != "DIVERGEN"]
    plt.figure(figsize=(10, 4))
    plt.bar(ok.kontinjensi, ok["max_loading_%"],
            color=["tab:red" if x > LOAD_MAX else "tab:green" for x in ok["max_loading_%"]])
    plt.axhline(LOAD_MAX, ls="--", c="k"); plt.xticks(rotation=60)
    plt.ylabel("Loading maksimum (%)"); plt.title("Loading Cabang Maksimum per Kontinjensi N-1")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "loading_n1.png"), dpi=200); plt.close()

    # --- Diagram topologi: base case vs kontinjensi terburuk ---
    plot_topology(net, base, net.res_bus.vm_pu, "Base Case", os.path.join(FIG, "topologi_base.png"))
    worst = df.loc[df.status == "OK"].iloc[0].kontinjensi
    n = copy.deepcopy(net)
    tipe = "line" if worst.startswith("L") else "trafo"
    f, t = map(int, worst[1:].split("-"))
    if tipe == "line":
        idx = n.line[(n.line.from_bus == f) & (n.line.to_bus == t)].index[0]
    else:
        idx = n.trafo[(n.trafo.hv_bus == f) & (n.trafo.lv_bus == t)].index[0]
    n[tipe].at[idx, "in_service"] = False
    pp.runpp(n)
    bt = branch_table(n)
    bt["loading"] = 100 * bt.S_mva / bt.nama.map(rating)
    bt.loc[bt.nama == worst, "loading"] = np.nan
    plot_topology(n, bt, n.res_bus.vm_pu, f"Kontinjensi Terburuk: outage {worst}",
                  os.path.join(FIG, "topologi_kontinjensi_terburuk.png"))
    print("Gambar disimpan: heatmap_tegangan_n1.png, loading_n1.png, topologi_*.png")
