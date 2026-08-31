#!/usr/bin/env python3
"""
Simulasi Combined Economic & Emission Dispatch (CEED) Multi-Objektif
Sintesis Materi EE (2023-2026) dan Teknik Mesin (BFGS / Multi-Objective)
Mata Kuliah: Optimisasi untuk Teknik Elektro (TKE-41XX)
Semester Ganjil 2026 - Universitas Bengkulu
Penulis: Ir. Novalio Daratha S.T., M.Sc., Ph.D.
"""

import numpy as np
import matplotlib.pyplot as plt

def run_multiobjective_ceed():
    np.random.seed(42)
    
    # Specs 3 Generator: [a, b, c, alpha, beta, gamma, Pmin, Pmax]
    gen_specs = np.array([
        [0.00156, 7.92, 561, 0.00419, 0.3276, 13.86, 100, 600],
        [0.00194, 7.85, 310, 0.00419, 0.3276, 13.86, 100, 400],
        [0.00482, 7.97,  78, 0.00683, -0.545, 40.26,  50, 200]
    ])
    
    P_D = 850.0  # MW
    N_g = len(gen_specs)
    
    # Weighting Trade-off Points to sweep Pareto Front
    N_points = 60
    w_cost_list = np.linspace(0.0, 1.0, N_points)
    
    pareto_costs = []
    pareto_emissions = []
    pareto_powers = []
    
    Pmin = gen_specs[:, 6]
    Pmax = gen_specs[:, 7]
    
    for w_cost in w_cost_list:
        w_emiss = 1.0 - w_cost
        
        # Single point PSO search with weighted objective
        N_particles = 40
        Max_iter = 60
        
        X = np.zeros((N_particles, N_g))
        for i in range(N_g):
            X[:, i] = np.random.uniform(Pmin[i], Pmax[i], N_particles)
            
        V = np.zeros((N_particles, N_g))
        
        def obj_func(x):
            P = np.copy(x)
            P[:-1] = np.clip(P[:-1], Pmin[:-1], Pmax[:-1])
            P[-1] = P_D - np.sum(P[:-1])
            
            penalty = 0
            if P[-1] < Pmin[-1] or P[-1] > Pmax[-1]:
                penalty = 1e5 * (abs(P[-1] - np.clip(P[-1], Pmin[-1], Pmax[-1])))**2
                
            cost = np.sum(gen_specs[:, 0]*P**2 + gen_specs[:, 1]*P + gen_specs[:, 2])
            emission = np.sum(gen_specs[:, 3]*P**2 + gen_specs[:, 4]*P + gen_specs[:, 5])
            
            # Weighted Single Objective for Pareto Sweep
            weighted = w_cost * cost + w_emiss * (emission * 18.0) + penalty
            return weighted, cost, emission, P

        pbest_X = np.copy(X)
        pbest_fit = np.array([obj_func(X[i])[0] for i in range(N_particles)])
        gbest_idx = np.argmin(pbest_fit)
        gbest_X = np.copy(pbest_X[gbest_idx])
        gbest_fit = pbest_fit[gbest_idx]
        
        for t in range(Max_iter):
            w = 0.7
            for i in range(N_particles):
                r1, r2 = np.random.rand(N_g), np.random.rand(N_g)
                V[i] = w * V[i] + 1.5 * r1 * (pbest_X[i] - X[i]) + 1.5 * r2 * (gbest_X - X[i])
                X[i] += V[i]
                
                fit_val, _, _, _ = obj_func(X[i])
                if fit_val < pbest_fit[i]:
                    pbest_fit[i] = fit_val
                    pbest_X[i] = np.copy(X[i])
                    if fit_val < gbest_fit:
                        gbest_fit = fit_val
                        gbest_X = np.copy(X[i])
                        
        _, c_final, e_final, p_final = obj_func(gbest_X)
        pareto_costs.append(c_final)
        pareto_emissions.append(e_final)
        pareto_powers.append(p_final)
        
    pareto_costs = np.array(pareto_costs)
    pareto_emissions = np.array(pareto_emissions)
    pareto_powers = np.array(pareto_powers)
    
    # Fuzzy Decision Making for Best Compromise Solution
    c_min, c_max = np.min(pareto_costs), np.max(pareto_costs)
    e_min, e_max = np.min(pareto_emissions), np.max(pareto_emissions)
    
    mu_c = (c_max - pareto_costs) / (c_max - c_min + 1e-6)
    mu_e = (e_max - pareto_emissions) / (e_max - e_min + 1e-6)
    mu_tot = (mu_c + mu_e) / np.sum(mu_c + mu_e)
    
    best_idx = np.argmax(mu_tot)
    best_cost = pareto_costs[best_idx]
    best_emiss = pareto_emissions[best_idx]
    best_power = pareto_powers[best_idx]
    
    print("=" * 65)
    print(" HASIL OPTIMISASI MULTI-OBJEKTIF CEED (PARETO FRONTIER)")
    print("=" * 65)
    print(f" Minimum Biaya Bahan Bakar  : ${np.min(pareto_costs):.2f} / jam (Emisi: {pareto_emissions[np.argmin(pareto_costs)]:.2f} kg/h)")
    print(f" Minimum Emisi Gas Polutan  : {np.min(pareto_emissions):.2f} kg / jam (Biaya: ${pareto_costs[np.argmin(pareto_emissions)]:.2f} /h)")
    print("-" * 65)
    print(" SOLUSI KOMPROMI TERBAIK (BEST COMPROMISE SOLUTION - FUZZY LOGIC):")
    print(f"  - Output Daya P1        : {best_power[0]:.2f} MW")
    print(f"  - Output Daya P2        : {best_power[1]:.2f} MW")
    print(f"  - Output Daya P3        : {best_power[2]:.2f} MW")
    print(f"  - Total Biaya Bahan Bakar: ${best_cost:.2f} / jam")
    print(f"  - Total Emisi Polutan   : {best_emiss:.2f} kg / jam")
    print("=" * 65)
    
    # Plot Pareto Front
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(pareto_costs, pareto_emissions, 'o-', color='#003366', linewidth=2, markersize=5, label='Pareto Frontier (MOPSO)')
    plt.plot(best_cost, best_emiss, '*', color='#d4af37', markersize=14, label='Best Compromise Solution (Fuzzy)')
    
    plt.title('Pareto Frontier: Combined Economic & Emission Dispatch', fontsize=12, fontweight='bold', color='#003366')
    plt.xlabel('Total Biaya Bahan Bakar ($/jam)', fontsize=10)
    plt.ylabel('Total Emisi Polutan NOx (kg/jam)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pso_ceed_pareto_front.png')
    print("Grafik Pareto Frontier disimpan ke 'pso_ceed_pareto_front.png'")

if __name__ == '__main__':
    run_multiobjective_ceed()
