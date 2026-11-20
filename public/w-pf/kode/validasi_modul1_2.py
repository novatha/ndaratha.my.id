"""
validasi_modul1_2.py — Uji Modul 1 & 2 pada IEEE 14-bus.
  1) Ybus sparse: sifat simetri & pola sparsity
  2) Jacobian vektor vs per-elemen
  3) NR, FDLF, GS: bandingkan dengan solusi referensi & pandapower
  4) Demo Q-limit switching
  5) Plot kurva konvergensi (figures/konvergensi.png)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ieee14_data import case14, VM, VA, QMAX
import pf_core as pf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

case = case14()
bus, gen, branch, base = case["bus"], case["gen"], case["branch"], case["baseMVA"]

# 1) Ybus -------------------------------------------------------------------
Ybus, Yf, Yt = pf.make_ybus(base, bus, branch)
print("=== MODUL 1: Ybus ===")
print(f"Ukuran {Ybus.shape}, nnz = {Ybus.nnz}, densitas = {Ybus.nnz / Ybus.shape[0]**2:.1%}")
print("Simetris (tanpa phase-shifter)?", np.allclose(Ybus.toarray(), Ybus.toarray().T))
print("Y[0,0] =", np.round(Ybus[0, 0], 4))

# 2) Jacobian ---------------------------------------------------------------
ref, pv, pq = pf.bus_types(bus)
V0 = pf.initial_voltage(bus, gen)
Vtest = V0 * np.exp(1j * 0.05 * np.random.default_rng(0).standard_normal(len(V0)))
Jv = pf.jacobian_polar(Ybus, Vtest, pv, pq).toarray()
Je = pf.jacobian_polar_explicit(Ybus, Vtest, pv, pq)
print("\n=== MODUL 2: Jacobian ===")
print(f"Ukuran J = {Jv.shape}, max|J_vektor − J_elemen| = {np.abs(Jv - Je).max():.2e}")

# 3) Solver -----------------------------------------------------------------
Sbus = pf.make_sbus(base, bus, gen)
t0 = time.perf_counter()
V_nr, ok, it_nr, h_nr = pf.newton_raphson(Ybus, Sbus, V0, ref, pv, pq, verbose=True)
t_nr = time.perf_counter() - t0
t0 = time.perf_counter(); V_fd, ok_fd, it_fd, h_fd = pf.fdlf_xb(case); t_fd = time.perf_counter() - t0
t0 = time.perf_counter(); V_gs, ok_gs, it_gs, h_gs = pf.gauss_seidel(case, alpha=1.6); t_gs = time.perf_counter() - t0

Vm_ref, Va_ref = bus[:, VM], bus[:, VA]
err_m = np.abs(np.abs(V_nr) - Vm_ref).max()
err_a = np.abs(np.rad2deg(np.angle(V_nr)) - Va_ref).max()
print(f"\nNR  : konvergen={ok}, iterasi={it_nr}, waktu={t_nr*1e3:.1f} ms")
print(f"FDLF: konvergen={ok_fd}, iterasi={it_fd}, waktu={t_fd*1e3:.1f} ms, "
      f"max|V_NR − V_FD| = {np.abs(V_nr - V_fd).max():.1e}")
print(f"GS  : konvergen={ok_gs}, iterasi={it_gs}, waktu={t_gs*1e3:.1f} ms, "
      f"max|V_NR − V_GS| = {np.abs(V_nr - V_gs).max():.1e}")
print(f"Selisih vs solusi referensi (pembulatan 3 desimal): |V| {err_m:.1e} pu, θ {err_a:.1e}°")

print("\nBus   |V| (pu)   θ (deg)")
for i in range(len(V_nr)):
    print(f"{i+1:3d}   {abs(V_nr[i]):.4f}   {np.rad2deg(np.angle(V_nr[i])):8.3f}")

Sf, St, Sloss = pf.branch_flows(case, V_nr, Yf, Yt)
print(f"\nTotal rugi-rugi = {Sloss.sum().real:.3f} MW + j{Sloss.sum().imag:.3f} MVAr")

# Validasi silang dengan pandapower
try:
    import pandapower as pp
    import pandapower.networks as pn
    net = pn.case14()
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-9)
    dv = np.abs(net.res_bus.vm_pu.values - np.abs(V_nr)).max()
    da = np.abs(net.res_bus.va_degree.values - np.rad2deg(np.angle(V_nr))).max()
    print(f"Validasi pandapower: max Δ|V| = {dv:.1e} pu, max Δθ = {da:.1e}°, "
          f"rugi pandapower = {net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum():.3f} MW")
except ImportError:
    print("pandapower tidak terpasang — validasi silang dilewati.")

# 4) Q-limit ----------------------------------------------------------------
print("\n=== MODUL 2: Q-limit switching (Qmax gen bus 8 diturunkan ke 10 MVAr) ===")
case_q = case14()
case_q["gen"][4, QMAX] = 10.0
res = pf.runpf_qlim(case_q)
print(f"Bus yang berpindah PV->PQ: {res['switched']}; |V8| = {abs(res['V'][7]):.4f} pu "
      f"(setpoint 1.090), Qg8 = {res['gen'][4, 2]:.2f} MVAr")

# 5) Plot konvergensi -------------------------------------------------------
plt.figure(figsize=(7, 4.2))
plt.semilogy(h_nr, "o-", label=f"Newton-Raphson ({it_nr} it)")
plt.semilogy(h_fd, "s-", label=f"Fast Decoupled XB ({it_fd} it)")
plt.semilogy(h_gs, "-", label=f"Gauss-Seidel α=1.6 ({it_gs} it)")
plt.axhline(1e-8, color="k", ls="--", lw=0.8, label="toleransi 1e-8")
plt.xlabel("Iterasi"); plt.ylabel(r"$\|\Delta F\|_\infty$ (p.u.)")
plt.title("Kurva Konvergensi — IEEE 14-bus (flat start)")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "konvergensi.png"), dpi=200)
print("\nGambar disimpan: figures/konvergensi.png")
