% =========================================================================
% SIMULASI EKSPERIMEN BERBASIS AKUSTIK (ACOUSTIC EMISSION - AE)
% Studi Kasus: Deteksi Kebocoran Pipa Bertekanan & Analisis Dispersi TDOA
% Mata Kuliah: Metode Eksperimen Teknik (TIN-213) - Tingkat Magister (S2)
% Dosen: Ir. Novalio Daratha, S.T., M.Sc., Ph.D.
% =========================================================================

clear; clc; close all;

fprintf('====================================================================\n');
fprintf('  SIMULASI PENGOLAHAN DATA AKUSTIK TINGKAT LANJUT (S2 TEKNIK MESIN)  \n');
fprintf('  Karakterisasi Sinyal Kontinu (Kurtosis, ASL) & Efek Dispersi TDOA \n');
fprintf('====================================================================\n\n');

% 1. PARAMETER AKUISISI DATA FREKUENSI TINGGI
Fs = 2e6;              % Frekuensi sampling: 2 MHz (2 MS/s)
dt = 1 / Fs;           % Interval waktu
T_total = 0.005;       % Durasi rekam sinyal: 5 ms
t = (0:dt:T_total-dt)';
N = length(t);

f_res = 150e3;         % Frekuensi resonansi sensor AE (R15alpha: 150 kHz)

% 2. SINTESIS TIGA KONDISI TEKANAN & KEBOCORAN PIPA
% Kasus 1: Pipa Normal (Rapat, Tekanan Stabil, hanya derau latar mekanis Gaussian)
randn('seed', 101);
v_normal = 0.012 * randn(size(t));

% Kasus 2: Kebocoran Halus / Pinhole (Tekanan 15 Bar, celah 0.2 mm) - Continuous AE
v_micro = 0.035 * randn(size(t)) + ...
          0.045 * sin(2*pi*f_res*t + rand(size(t))*0.5);

% Kasus 3: Kebocoran Besar / Major Leak (Tekanan 15 Bar, celah 1 mm) - High Energy Continuous AE
v_major = 0.120 * randn(size(t)) + ...
          0.200 * sin(2*pi*f_res*t + rand(size(t))*0.8);

% 3. PENGOLAHAN INFORMASI: EKSTRAKSI FITUR STATISTIK SINYAL KONTINU (S2 METRICS)
signals = [v_normal, v_micro, v_major];
labels = {'Pipa Rapat (Normal)', 'Bocor Halus (Pinhole)', 'Bocor Kritis (Major)'};
num_cases = length(labels);

v_rms    = zeros(num_cases, 1);
asl_db   = zeros(num_cases, 1); % Average Signal Level (dB ref 1 uV)
kurt_val = zeros(num_cases, 1); % Kurtosis (derajat kepekaan letupan)
crest_f  = zeros(num_cases, 1); % Crest Factor (Peak / RMS)

V_ref = 1e-6; % 1 uV tegangan acuan standar AE

for i = 1:num_cases
    sig = signals(:, i);
    rms_i = sqrt(mean(sig.^2));
    v_rms(i) = rms_i;
    % ASL dalam skala dBAE
    asl_db(i) = 20 * log10(rms_i / V_ref);
    % Kurtosis
    mu = mean(sig);
    sigma = std(sig);
    kurt_val(i) = mean((sig - mu).^4) / (sigma^4);
    % Crest Factor
    crest_f(i) = max(abs(sig)) / rms_i;
end

% 4. ANALISIS DISPERSI GELOMBANG LAMB PADA LOKALISASI TDOA
L_pipe = 4.0;          % Jarak antar dua sensor AE: 4.0 meter
x_actual = 1.2;        % Lokasi kebocoran sebenarnya: 1.2 m dari Sensor 1
vg_S0 = 5200;          % Kecepatan grup mode Simetris S0 (m/s)
vg_A0 = 3100;          % Kecepatan grup mode Asimetris A0 (m/s)

% Selisih waktu tiba gelombang sebenarnya berdasarkan mode S0 tercepat:
delta_t_actual = ( (L_pipe - x_actual) - x_actual ) / vg_S0; 

% Estimasi lokasi jika insinyur keliru mengasumsikan mode A0 yang memicu sensor:
x_est_S0 = 0.5 * (L_pipe - vg_S0 * delta_t_actual);
x_est_A0 = 0.5 * (L_pipe - vg_A0 * delta_t_actual);
error_A0 = abs(x_est_A0 - x_actual);

% 5. MENAMPILKAN TABEL EVALUASI REKAYASA
fprintf('%-24s | %-12s | %-12s | %-10s | %-12s\n', 'Kondisi Pipa Fluida', 'RMS (mV)', 'ASL (dBAE)', 'Kurtosis', 'Crest Factor');
fprintf('------------------------------------------------------------------------------------\n');
for i = 1:num_cases
    fprintf('%-24s | %10.2f mV | %10.1f dB | %10.2f | %10.2f\n', ...
            labels{i}, v_rms(i)*1e3, asl_db(i), kurt_val(i), crest_f(i));
end
fprintf('------------------------------------------------------------------------------------\n\n');

fprintf('--- EVALUASI GALAT LOKALISASI TDOA AKIBAT EFEK DISPERSI GELOMBANG LAMB ---\n');
fprintf('Jarak Sensor (L)          : %.1f meter\n', L_pipe);
fprintf('Lokasi Kebocoran Riil     : %.2f meter dari Sensor 1\n', x_actual);
fprintf('Selisih Waktu Tiba (dt)   : %.2f mikro-detik\n', delta_t_actual * 1e6);
fprintf('Estimasi dengan Mode S0   : %.2f meter (Galat: 0.00 cm - SANGAT AKURAT)\n', x_est_S0);
fprintf('Estimasi dengan Mode A0   : %.2f meter (Galat: %.2f meter / %.1f %%)\n', ...
        x_est_A0, error_A0, (error_A0/L_pipe)*100);
fprintf('--------------------------------------------------------------------------\n\n');

% 6. VISUALISASI GRAFIK
figure('Position', [100, 100, 1050, 750], 'Visible', 'off');

% Subplot 1: Sinyal Waktu & ASL
subplot(2, 2, 1);
plot(t*1e3, v_normal*1e3, 'b-', 'LineWidth', 0.8, 'DisplayName', 'Normal'); hold on;
plot(t*1e3, v_micro*1e3, 'g-', 'LineWidth', 0.9, 'DisplayName', 'Bocor Halus');
plot(t*1e3, v_major*1e3, 'r-', 'LineWidth', 1.0, 'DisplayName', 'Bocor Kritis');
grid on;
xlabel('Waktu (ms)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Tegangan AE (mV)', 'FontSize', 10, 'FontWeight', 'bold');
title('1. Sinyal Kontinu Kebocoran Fluida', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast');
xlim([0, 3]);

% Subplot 2: Spektrum Frekuensi FFT
subplot(2, 2, 2);
f_axis = linspace(0, Fs/2, N/2+1)';
function P1 = calc_fft(sig, N)
    Y = fft(sig); P2 = abs(Y/N); P1 = P2(1:N/2+1); P1(2:end-1) = 2*P1(2:end-1);
endfunction
plot(f_axis/1e3, calc_fft(v_normal, N)*1e3, 'b-', 'LineWidth', 1.0, 'DisplayName', 'Normal'); hold on;
plot(f_axis/1e3, calc_fft(v_micro, N)*1e3, 'g-', 'LineWidth', 1.2, 'DisplayName', 'Bocor Halus');
plot(f_axis/1e3, calc_fft(v_major, N)*1e3, 'r-', 'LineWidth', 1.4, 'DisplayName', 'Bocor Kritis');
grid on;
xlabel('Frekuensi (kHz)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Amplitudo (mV)', 'FontSize', 10, 'FontWeight', 'bold');
title('2. Resonansi Spektrum Pipa (Puncak 150 kHz)', 'FontSize', 10, 'FontWeight', 'bold');
xlim([50, 350]);

% Subplot 3: Bar Chart ASL (dBAE)
subplot(2, 2, 3);
bar(asl_db, 0.45, 'FaceColor', [0.2, 0.5, 0.8]);
set(gca, 'XTickLabel', labels, 'FontSize', 9);
grid on;
ylabel('Tingkat Sinyal ASL (dBAE)', 'FontSize', 10, 'FontWeight', 'bold');
title('3. Metrik Energi Kontinu (Average Signal Level)', 'FontSize', 10, 'FontWeight', 'bold');
yline(90, 'r--', 'Ambang Bahaya (90 dBAE)', 'LineWidth', 1.1);

% Subplot 4: Perbandingan Lokalisasi TDOA (Dispersi S0 vs A0)
subplot(2, 2, 4);
modes_label = {'Posisi Sebenarnya', 'Estimasi Mode S0', 'Estimasi Salah Mode A0'};
pos_values = [x_actual, x_est_S0, x_est_A0];
b = barh(pos_values, 0.5);
set(gca, 'YTickLabel', modes_label, 'FontSize', 9);
grid on;
xlabel('Posisi dari Sensor 1 (Meter)', 'FontSize', 10, 'FontWeight', 'bold');
title('4. Pengaruh Dispersi Gelombang pada Lokalisasi TDOA', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0, L_pipe]);
xline(x_actual, 'k--', 'Titik Bocor Nyata', 'LineWidth', 1.2);

print('plot_akustik_kebocoran.png', '-dpng', '-r300');
fprintf('Grafik hasil analisis telah diperbarui sebagai "plot_akustik_kebocoran.png".\n');
