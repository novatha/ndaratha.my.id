% =========================================================================
% SIMULASI EKSPERIMEN BERBASIS IMPEDANSI (ELECTROMECHANICAL IMPEDANCE - EMI)
% Studi Kasus: Monitoring Kuat-Kendur Baut & Kompensasi Suhu (EFS)
% Mata Kuliah: Metode Eksperimen Teknik (TIN-213) - Tingkat Magister (S2)
% Dosen: Ir. Novalio Daratha, S.T., M.Sc., Ph.D.
% =========================================================================

clear; clc; close all;

fprintf('====================================================================\n');
fprintf('  SIMULASI PENGOLAHAN DATA IMPEDANSI TINGKAT LANJUT (S2 TEKNIK MESIN)\n');
fprintf('  Integrasi Algoritma Kompensasi Suhu (Effective Frequency Shift - EFS)\n');
fprintf('====================================================================\n\n');

% 1. PENGATURAN PARAMETER AKUISISI FREKUENSI
f_min = 80e3;    % Frekuensi awal: 80 kHz
f_max = 140e3;   % Frekuensi akhir: 140 kHz
N_pts = 1200;    % Jumlah titik sampling frekuensi
f = linspace(f_min, f_max, N_pts)';
df = f(2) - f(1);

% Fungsi pembantu: Sintesis puncak resonansi mekanik struktur
function G = generate_conductance(freq, p_freqs, p_amps, p_widths, noise_lvl)
    G = 0.05 * (freq/1e5); % Garis dasar (baseline slope)
    for k = 1:length(p_freqs)
        f0 = p_freqs(k);
        A  = p_amps(k);
        gamma = p_widths(k);
        % Model resonansi Lorentzian
        G = G + A * (gamma^2 ./ ((freq - f0).^2 + gamma^2));
    end
    randn('seed', 42);
    G = G + noise_lvl * randn(size(freq));
    G = max(G, 0);
endfunction

% 2. DEFINISI KONDISI UJI STRUKTUR
% Baseline: Baut Kencang 50 Nm pada Suhu Referensi 25 deg C
p_freqs_base  = [92e3, 108e3, 125e3];
p_amps_base   = [0.85, 1.45,  1.10];
p_widths_base = [1.8e3, 2.2e3, 2.5e3];
G_base = generate_conductance(f, p_freqs_base, p_amps_base, p_widths_base, 0.015);

% Kondisi A: Baut Masih Kencang 50 Nm TETAPI Suhu Naik ke 40 deg C (+15 deg C)
% Suhu menurunkan modulus elastisitas -> pergeseran frekuensi -1.2 kHz + kenaikan permitivitas vertikal 4%
p_freqs_temp = p_freqs_base - 1.2e3;
G_temp_raw   = generate_conductance(f, p_freqs_temp, p_amps_base * 1.04, p_widths_base, 0.015);

% Kondisi B: Baut Mengendur ke 20 Nm pada Suhu Referensi 25 deg C
p_freqs_20Nm = p_freqs_base - [2.1e3, 2.7e3, 3.1e3];
G_20Nm       = generate_conductance(f, p_freqs_20Nm, p_amps_base * 0.90, p_widths_base * 1.12, 0.015);

% Kondisi C: Baut Mengendur ke 20 Nm DAN Suhu Naik ke 40 deg C
p_freqs_both = p_freqs_20Nm - 1.2e3;
G_both_raw   = generate_conductance(f, p_freqs_both, p_amps_base * 0.90 * 1.04, p_widths_base * 1.12, 0.015);

% 3. ALGORITMA KOMPENSASI SUHU (EFFECTIVE FREQUENCY SHIFT - EFS)
function [G_corr, best_shift] = apply_efs(G_target, G_ref, max_shift_pts)
    best_corr = -1;
    best_shift = 0;
    N = length(G_target);
    for s = -max_shift_pts:max_shift_pts
        if s >= 0
            t_idx = (1+s):N;
            r_idx = 1:(N-s);
        else
            t_idx = 1:(N+s);
            r_idx = (1-s):N;
        end
        c = corr(G_target(t_idx), G_ref(r_idx));
        if c > best_corr
            best_corr = c;
            best_shift = s;
        end
    end
    % Terapkan pergeseran sirkuler terkompensasi
    G_corr = circshift(G_target, -best_shift);
endfunction

max_shift_pts = 40; % Batas pencarian geseran frekuensi
[G_temp_comp, shift_temp] = apply_efs(G_temp_raw, G_base, max_shift_pts);
[G_both_comp, shift_both] = apply_efs(G_both_raw, G_base, max_shift_pts);

% 4. PENGOLAHAN INFORMASI: MENGHITUNG METRIK RMSD SEBELUM & SESUDAH EFS
function val = calc_rmsd(G_curr, G_ref)
    val = sqrt(sum((G_curr - G_ref).^2) / sum(G_ref.^2)) * 100;
endfunction

rmsd_base      = calc_rmsd(G_base, G_base);
rmsd_temp_raw  = calc_rmsd(G_temp_raw, G_base);
rmsd_temp_comp = calc_rmsd(G_temp_comp, G_base);
rmsd_20Nm      = calc_rmsd(G_20Nm, G_base);
rmsd_both_raw  = calc_rmsd(G_both_raw, G_base);
rmsd_both_comp = calc_rmsd(G_both_comp, G_base);

% 5. MENAMPILKAN TABEL EVALUASI KOMPARATIF
fprintf('%-32s | %-15s | %-16s | %-22s\n', 'Kondisi Operasional Uji', 'RMSD Asli (%)', 'RMSD EFS Comp(%)', 'Status Keputusan');
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf('%-32s | %13.2f %% | %14.2f %% | %-22s\n', 'Baseline (50 Nm, 25 C)', rmsd_base, rmsd_base, 'Normal (Kondisi Desain)');
fprintf('%-32s | %13.2f %% | %14.2f %% | %-22s\n', '50 Nm, Suhu 40 C (Hanya Suhu)', rmsd_temp_raw, rmsd_temp_comp, 'Toleransi (Alarm Dieliminasi!)');
fprintf('%-32s | %13.2f %% | %14.2f %% | %-22s\n', '20 Nm, Suhu 25 C (Baut Kendor)', rmsd_20Nm, rmsd_20Nm, 'BAHAYA: Kencangkan Baut!');
fprintf('%-32s | %13.2f %% | %14.2f %% | %-22s\n', '20 Nm, Suhu 40 C (Kombinasi)', rmsd_both_raw, rmsd_both_comp, 'BAHAYA: Kencangkan Baut!');
fprintf('----------------------------------------------------------------------------------------------------\n\n');

% 6. VISUALISASI HASIL PENGOLAHAN DATA
figure('Position', [100, 100, 1050, 750], 'Visible', 'off');

% Subplot 1: Perbandingan Spektrum Mentah vs Terkompensasi
subplot(2, 1, 1);
plot(f/1e3, G_base, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Baseline (50 Nm, 25^\circC)'); hold on;
plot(f/1e3, G_temp_raw, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Efek Suhu 40^\circC (Sebelum EFS)');
plot(f/1e3, G_temp_comp, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Efek Suhu 40^\circC (Setelah EFS)');
grid on;
xlabel('Frekuensi Eksitasi (kHz)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Konduktansi G (mS)', 'FontSize', 10, 'FontWeight', 'bold');
title('1. Pengolahan Data Lanjut: Spektrum Konduktansi & Efek Kompensasi Suhu (EFS)', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northeast');
xlim([85, 135]);

% Subplot 2: Perbandingan RMSD Mentah vs EFS
subplot(2, 1, 2);
test_categories = {'Suhu 40 C Murni', 'Baut 20 Nm (25 C)', 'Baut 20 Nm + Suhu 40 C'};
raw_rmsd_plot = [rmsd_temp_raw, rmsd_20Nm, rmsd_both_raw];
comp_rmsd_plot = [rmsd_temp_comp, rmsd_20Nm, rmsd_both_comp];

x_pos = 1:3;
w = 0.35;
bar(x_pos - w/2, raw_rmsd_plot, w, 'FaceColor', [0.8, 0.2, 0.2], 'DisplayName', 'Sebelum EFS (False Alarm)'); hold on;
bar(x_pos + w/2, comp_rmsd_plot, w, 'FaceColor', [0.15, 0.6, 0.3], 'DisplayName', 'Setelah EFS (True Damage)');
set(gca, 'XTick', x_pos, 'XTickLabel', test_categories, 'FontSize', 10);
grid on;
ylabel('Indeks Kerusakan RMSD (%)', 'FontSize', 10, 'FontWeight', 'bold');
title('2. Pengolahan Informasi: Eliminasi False-Alarm Termal Menggunakan Algoritma EFS', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northwest');
yline(15, 'k--', 'Ambang Batas Peringatan Baut Longgar (15%)', 'LineWidth', 1.2);
ylim([0, 80]);

print('plot_impedansi_baut.png', '-dpng', '-r300');
fprintf('Grafik hasil analisis telah diperbarui sebagai "plot_impedansi_baut.png".\n');
