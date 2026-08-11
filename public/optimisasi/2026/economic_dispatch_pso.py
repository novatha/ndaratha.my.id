#!/usr/bin/env python3
"""
Simulasi Economic Dispatch Sistem Tenaga Listrik Berbasis PSO
Mata Kuliah: Optimisasi untuk Teknik Elektro (TKE-41XX)
Semester Ganjil 2026 - Universitas Bengkulu
Penulis: Ir. Novalio Daratha S.T., M.Sc., Ph.D.
"""

import numpy as np
import matplotlib.pyplot as plt

def run_economic_dispatch_pso():
    # Set seed untuk reproduksibilitas
    np.random.seed(42)
    
    # Data Pembangkit Termal 3-Unit: [a ($/MW^2h), b ($/MWh), c ($/h), Pmin (MW), Pmax (MW)]
    gen_data = np.array([
        [0.00156, 7.92, 561, 100, 600],
        [0.00194, 7.85, 310, 100, 400],
        [0.00482, 7.97,  78,  50, 200]
    ])
    
    P_D = 850.0  # Total beban sistem (MW)
    N_g = len(gen_data)
    
    # Parameter PSO
    N_particles = 50
    Max_iter = 100
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    
    Pmin = gen_data[:, 3]
    Pmax = gen_data[:, 4]
    
    # Inisialisasi posisi dan kecepatan partikel
    X = np.zeros((N_particles, N_g))
    for i in range(N_g):
        X[:, i] = np.random.uniform(Pmin[i], Pmax[i], N_particles)
        
    V = np.zeros((N_particles, N_g))
    
    def calc_fitness(x):
        P = np.copy(x)
        # Clamping generator 1 s.d. N_g - 1
        P[:-1] = np.clip(P[:-1], Pmin[:-1], Pmax[:-1])
        # Generator terakhir (slack) menyeimbangkan daya
        P[-1] = P_D - np.sum(P[:-1])
        
        penalty = 0.0
        if P[-1] < Pmin[-1] or P[-1] > Pmax[-1]:
            penalty = 1e5 * (abs(P[-1] - np.clip(P[-1], Pmin[-1], Pmax[-1])))**2
            
        cost = np.sum(gen_data[:, 0] * P**2 + gen_data[:, 1] * P + gen_data[:, 2])
        return cost + penalty, P

    # Evaluasi awal
    pbest_X = np.copy(X)
    pbest_fit = np.zeros(N_particles)
    
    for i in range(N_particles):
        pbest_fit[i], _ = calc_fitness(X[i])
        
    gbest_idx = np.argmin(pbest_fit)
    gbest_X = np.copy(pbest_X[gbest_idx])
    gbest_fit = pbest_fit[gbest_idx]
    
    history_fit = []
    
    # Loop Utama PSO
    for t in range(Max_iter):
        w = w_max - ((w_max - w_min) / Max_iter) * t  # Linearly decreasing inertia weight
        
        for i in range(N_particles):
            r1 = np.random.rand(N_g)
            r2 = np.random.rand(N_g)
            
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
            X[i] = X[i] + V[i]
            
            fit_val, _ = calc_fitness(X[i])
            if fit_val < pbest_fit[i]:
                pbest_fit[i] = fit_val
                pbest_X[i] = np.copy(X[i])
                if fit_val < gbest_fit:
                    gbest_fit = fit_val
                    gbest_X = np.copy(X[i])
                    
        history_fit.append(gbest_fit)

    final_cost, P_opt = calc_fitness(gbest_X)
    
    print("=" * 60)
    print(" HASIL OPTIMISASI ECONOMIC DISPATCH BERBASIS PSO")
    print("=" * 60)
    for i in range(N_g):
        print(f" Generator {i+1} : {P_opt[i]:6.2f} MW  (Batas: [{Pmin[i]}, {Pmax[i]}] MW)")
    print("-" * 60)
    print(f" Total Daya Pembangkitan : {np.sum(P_opt):6.2f} MW  (Target Load: {P_D:.2f} MW)")
    print(f" Biaya Bahan Bakar Min.  : ${final_cost:10.2f} / jam")
    print("=" * 60)
    
    # Plot Profil Konvergensi
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(range(1, Max_iter + 1), history_fit, color='#003366', linewidth=2.5, label='Biaya Bahan Bakar ($/jam)')
    plt.title('Kurva Konvergensi PSO pada Economic Dispatch 3-Generator', fontsize=12, fontweight='bold', color='#003366')
    plt.xlabel('Iterasi', fontsize=10)
    plt.ylabel('Total Biaya ($/jam)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pso_ed_convergence.png')
    print("Grafik konvergensi disimpan ke 'pso_ed_convergence.png'")
    
    # Plot Alokasi Daya
    plt.figure(figsize=(7, 4.5), dpi=300)
    bars = plt.bar([f'Pembangkit {i+1}' for i in range(N_g)], P_opt, color=['#003366', '#d4af37', '#228b22'], width=0.5)
    plt.ylabel('Output Daya (MW)', fontsize=10)
    plt.title(f'Alokasi Daya Optimal (Total = {P_D} MW)', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:.2f} MW', ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, max(P_opt) + 70)
    plt.tight_layout()
    plt.savefig('pso_ed_power_allocation.png')
    print("Grafik alokasi daya disimpan ke 'pso_ed_power_allocation.png'")

if __name__ == '__main__':
    run_economic_dispatch_pso()
