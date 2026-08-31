#!/usr/bin/env python3
"""
Skrip Demo Pertemuan 1: Pengantar Optimisasi & Pemodelan Matematis
Mata Kuliah: Optimisasi untuk Teknik Elektro (TKE-41XX)
Semester Ganjil 2026 - Universitas Bengkulu
Penulis: Ir. Novalio Daratha S.T., M.Sc., Ph.D.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_intro_demo():
    print("=" * 60)
    print(" PERTEMUAN 1: DEMO PEMODELAN OPTIMISASI 2 GENERATOR")
    print("=" * 60)
    
    # Model Specs
    # F1(P1) = 0.002*P1^2 + 8.0*P1 + 100
    # F2(P2) = 0.003*P2^2 + 7.5*P2 + 150
    # P1 + P2 = 400 MW, 50 <= P1 <= 300, 50 <= P2 <= 250
    P_D = 400.0
    
    def objective(P):
        P1, P2 = P[0], P[1]
        cost1 = 0.002 * P1**2 + 8.0 * P1 + 100
        cost2 = 0.003 * P2**2 + 7.5 * P2 + 150
        return cost1 + cost2

    constraints = ({'type': 'eq', 'fun': lambda P: P[0] + P[1] - P_D})
    bounds = [(50, 300), (50, 250)]

    # Initial Guess
    P0 = [200.0, 200.0]
    res = minimize(objective, P0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    P1_opt, P2_opt = res.x[0], res.x[1]
    cost_opt = res.fun
    
    print(f" Day Generator 1 (P1*) : {P1_opt:.2f} MW")
    print(f" Day Generator 2 (P2*) : {P2_opt:.2f} MW")
    print(f" Total Biaya Minimum   : ${cost_opt:.2f} / jam")
    print("=" * 60)
    
    # Plot Feasible Region & Contour Lines
    P1_vals = np.linspace(40, 310, 200)
    P2_vals = np.linspace(40, 260, 200)
    P1_grid, P2_grid = np.meshgrid(P1_vals, P2_vals)
    
    Cost_grid = (0.002 * P1_grid**2 + 8.0 * P1_grid + 100) + (0.003 * P2_grid**2 + 7.5 * P2_grid + 150)
    
    plt.figure(figsize=(8, 6), dpi=300)
    CS = plt.contour(P1_grid, P2_grid, Cost_grid, levels=20, cmap='viridis', alpha=0.8)
    plt.clabel(CS, inline=True, fontsize=8, fmt='$%1.0f')
    
    # Constraint Line: P1 + P2 = 400
    P1_line = np.linspace(150, 300, 100)
    P2_line = P_D - P1_line
    plt.plot(P1_line, P2_line, 'r--', linewidth=2.5, label=r'Kendala Beban: $P_1 + P_2 = 400$ MW')
    
    # Optimum Point
    plt.plot(P1_opt, P2_opt, 'g*', markersize=14, label=f'Optimum: ({P1_opt:.1f}, {P2_opt:.1f}) MW')
    
    plt.axvline(x=50, color='gray', linestyle=':', label='Batas P1 (50 - 300 MW)')
    plt.axvline(x=300, color='gray', linestyle=':')
    plt.axhline(y=50, color='gray', linestyle='--', label='Batas P2 (50 - 250 MW)')
    plt.axhline(y=250, color='gray', linestyle='--')
    
    plt.title('Ruang Feasibel & Kontur Biaya: Economic Dispatch 2 Generator', fontsize=11, fontweight='bold', color='#003366')
    plt.xlabel('Daya Generator 1 ($P_1$ [MW])', fontsize=10)
    plt.ylabel('Daya Generator 2 ($P_2$ [MW])', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    
    plt.savefig('intro_optimization_feasible_region.png')
    print("Grafik Kontur & Ruang Feasibel disimpan ke 'intro_optimization_feasible_region.png'")

if __name__ == '__main__':
    run_intro_demo()
