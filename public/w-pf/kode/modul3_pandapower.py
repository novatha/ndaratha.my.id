"""
modul3_pandapower.py — Konversi data IEEE 14-bus (format MATPOWER/PYPOWER) ke
pandapower, menjalankan AC power flow, profil tegangan, dan analisis rugi-rugi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.converter as pc

from ieee14_data import case14

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def build_net():
    """Konversi dict ppc -> pandapowerNet. BASE_KV=0 pada data asli diganti 1 kV
    agar konversi impedansi konsisten (sistem tetap dianalisis dalam p.u.)."""
    ppc = case14()
    ppc["version"] = "2"
    ppc["bus"] = ppc["bus"].copy()
    ppc["bus"][:, 9] = np.where(ppc["bus"][:, 9] == 0, 135.0, ppc["bus"][:, 9])
    net = pc.from_ppc(ppc, f_hz=60, validate_conversion=False)
    net.bus["name"] = [f"Bus {i}" for i in net.bus.index]  # indeks = nomor bus asli
    return net


if __name__ == "__main__":
    net = build_net()
    print("=== Struktur DataFrame pandapower ===")
    for tbl in ["bus", "load", "gen", "ext_grid", "shunt", "line", "trafo"]:
        print(f"  net.{tbl:<8}: {len(net[tbl]):2d} baris")
    print(net.line[["from_bus", "to_bus", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km"]].head())

    pp.runpp(net, algorithm="nr", init="flat", tolerance_mva=1e-8)
    print(f"\nKonvergen: {net.converged}, iterasi: {net._ppc['iterations']}")

    # Profil tegangan
    rb = net.res_bus.copy()
    rb.index = net.bus.name
    print("\n=== Profil tegangan ===")
    print(rb[["vm_pu", "va_degree", "p_mw", "q_mvar"]].round(4))
    vmin, vmax = 0.94, 1.06   # batas VMIN/VMAX data IEEE 14-bus
    out = rb[(rb.vm_pu < vmin) | (rb.vm_pu > vmax)]
    print(f"Bus di luar pita [{vmin}, {vmax}] pu: {list(out.index)}")

    # Rugi-rugi
    loss = pd.concat([
        pd.DataFrame({"elemen": [f"L{r.from_bus}-{r.to_bus}" for r in net.line.itertuples()],
                      "P_loss_MW": net.res_line.pl_mw.values,
                      "Q_loss_MVAr": net.res_line.ql_mvar.values}),
        pd.DataFrame({"elemen": [f"T{r.hv_bus}-{r.lv_bus}" for r in net.trafo.itertuples()],
                      "P_loss_MW": net.res_trafo.pl_mw.values,
                      "Q_loss_MVAr": net.res_trafo.ql_mvar.values}),
    ]).sort_values("P_loss_MW", ascending=False).reset_index(drop=True)
    P_load = net.res_load.p_mw.sum()
    P_loss = loss.P_loss_MW.sum()
    print("\n=== 5 cabang dengan rugi-rugi aktif terbesar ===")
    print(loss.head().round(3))
    print(f"Total rugi = {P_loss:.3f} MW ({100*P_loss/P_load:.2f}% dari beban {P_load:.1f} MW)")

    # Plot profil tegangan & rugi
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    idx = np.arange(1, len(rb) + 1)
    ax[0].bar(idx, rb.vm_pu, color=["tab:red" if v > vmax or v < vmin else "tab:blue"
                                    for v in rb.vm_pu])
    ax[0].axhline(vmax, ls="--", c="k", lw=0.8); ax[0].axhline(vmin, ls="--", c="k", lw=0.8)
    ax[0].set_ylim(0.9, 1.1); ax[0].set_xticks(idx)
    ax[0].set_xlabel("Bus"); ax[0].set_ylabel("|V| (p.u.)"); ax[0].set_title("Profil Tegangan")
    top = loss.head(10)
    ax[1].barh(top.elemen[::-1], top.P_loss_MW[::-1], color="tab:orange")
    ax[1].set_xlabel("P loss (MW)"); ax[1].set_title("10 Cabang dengan Rugi Terbesar")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "profil_tegangan_rugi.png"), dpi=200)
    print("Gambar disimpan: figures/profil_tegangan_rugi.png")
