import os
import subprocess

def create_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

materials = {
    "06": {
        "title": "Konsep Konveksitas",
        "desc": "Himpunan cembung, fungsi cembung, dan masalah optimisasi cembung. Pentingnya konveksitas.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\definecolor{UNIBYellow}{RGB}{255, 204, 0}
\setbeamercolor{palette primary}{bg=UNIBBlue,fg=white}
\setbeamercolor{title}{fg=UNIBBlue}
\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}
\subtitle{Minggu 6: Konsep Konveksitas}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Semester Ganjil 2026}
\begin{document}
\begin{frame}\titlepage\end{frame}
\begin{frame}{Tujuan Pembelajaran}
\begin{itemize}
    \item Mendefinisikan himpunan cembung dan fungsi cembung.
    \item Mengidentifikasi apakah suatu masalah optimisasi bersifat cembung.
    \item Menjelaskan mengapa optimisasi cembung sangat penting dalam keteknikan.
\end{itemize}
\end{frame}
\begin{frame}{Himpunan dan Fungsi Cembung}
\textbf{Himpunan Cembung:} Garis yang menghubungkan dua titik mana pun dalam himpunan tersebut sepenuhnya berada di dalam himpunan.\\
\textbf{Fungsi Cembung:} Garis (secant) yang menghubungkan dua titik pada kurva selalu berada di atas atau pada kurva fungsi tersebut.
\end{frame}
\begin{frame}{Pentingnya Konveksitas}
\begin{tcolorbox}[colback=yellow!5,title=Sifat Emas Konveksitas]
Setiap minimum lokal pada fungsi cembung dalam himpunan cembung adalah \textbf{minimum global}.
\end{tcolorbox}
Dalam teknik elektro, mendesain sistem komunikasi atau filter sinyal seringkali dapat diformulasikan sebagai masalah konveks (seperti Convex Optimization, SOCP). Jika model kita konveks, kita dijamin menemukan solusi optimal global secara efisien.
\end{frame}
\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{xcolor}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Catatan Kuliah - Minggu 6}
\cfoot{\thepage}
\title{\vspace{-1cm}\color{UNIBBlue}\textbf{Konsep Konveksitas}\vspace{-0.5cm}}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Minggu 6}
\begin{document}
\maketitle
\section{Pengantar Konveksitas}
Dalam optimisasi, pemisahan yang paling penting bukan antara masalah Linier dan Non-Linier, melainkan antara masalah \textbf{Cembung (Convex)} dan \textbf{Tidak Cembung (Non-Convex)}. Optimisasi cembung menjamin bahwa setiap minimum lokal adalah minimum global.
\section{Definisi Formal}
\textbf{Himpunan Cembung:} Suatu himpunan $C$ disebut cembung jika untuk setiap $x, y \in C$ dan setiap $\theta$ dengan $0 \le \theta \le 1$, berlaku:
$\theta x + (1 - \theta)y \in C$.
\textbf{Fungsi Cembung:} Fungsi $f(x)$ disebut cembung jika:
$f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$.
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{xcolor}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Lembar Kerja Mahasiswa - Minggu 6}
\cfoot{\thepage}
\begin{document}
\begin{center}\Large\color{UNIBBlue}\textbf{LKM Minggu 6: Menguji Konveksitas}\end{center}
\textbf{Nama :} \rule{6cm}{0.4pt} \\
\textbf{NPM :} \rule{6.8cm}{0.4pt}
\section*{Aktivitas 1: Uji Turunan Kedua (C3)}
Buktikan apakah fungsi $f(x) = x^4 + 2x^2$ cembung menggunakan turunan kedua (Hessian/Derivatif ke-2).
Syarat: $f''(x) \ge 0$.
\section*{Aktivitas 2: Identifikasi Visual dengan Python (C4)}
Gunakan plot pada Python untuk melihat bentuk fungsi $g(x) = x \sin(x)$ di rentang $[0, \pi]$. Apakah fungsi ini cembung?
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{xcolor}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Problem Set - Minggu 6}
\cfoot{\thepage}
\begin{document}
\begin{center}\Large\color{UNIBBlue}\textbf{Problem Set 6: Konveksitas}\end{center}
\section*{Soal 1 (C2)}
Jelaskan mengapa masalah optimisasi linier (LP) secara otomatis merupakan masalah optimisasi cembung!
\section*{Soal 2 (C4)}
Buktikan secara matematis apakah himpunan $S = \{ (x_1, x_2) \mid x_1^2 + x_2^2 \le 4 \}$ adalah himpunan cembung.
\section*{Soal 3 (C5)}
Diberikan sebuah masalah optimisasi untuk merancang antena. Anda menemukan bahwa fungsi biayanya non-cembung. Jelaskan risiko apa yang mungkin terjadi saat menggunakan algoritma \textit{Gradient Descent} biasa, dan apa solusinya?
\end{document}"""
    },
    "07": {
        "title": "Optimisasi Cembung Terkendala (KKT)",
        "desc": "Kondisi Karush-Kuhn-Tucker (KKT). Pengantar Interior-Point Methods.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 7: Optimisasi Cembung Terkendala (KKT)}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Kondisi Karush-Kuhn-Tucker (KKT)}
Generalisasi dari metode Pengali Lagrange untuk menangani kendala pertidaksamaan.
Syarat perlu (dan cukup untuk masalah konveks) agar suatu solusi menjadi optimal.
\begin{itemize}
\item Stationarity: Gradien Lagrangia bernilai 0.
\item Primal Feasibility: Memenuhi batasan asli.
\item Dual Feasibility: Pengali Lagrange $\ge 0$ untuk pertidaksamaan.
\item Complementary Slackness: $\lambda_i \cdot g_i(x) = 0$.
\end{itemize}
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 7: KKT dan Interior-Point}\maketitle
\section{Kondisi KKT} Kondisi KKT adalah fondasi teoritis untuk optimisasi non-linier terkendala. Ia mengombinasikan fungsi objektif dan kendala ke dalam satu fungsi yang disebut Lagrangian. \section{Interior Point Methods} Metode komputasi modern yang sangat cepat untuk menyelesaikan LP dan NLP cembung. Ia berjalan di *dalam* daerah fisibel, tidak menyusuri tepi seperti Simpleks. \end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 7: Menyelesaikan Masalah QP}\maketitle
\section*{Aktivitas: Quadratic Programming dengan Python}
Gunakan \texttt{scipy.optimize.minimize} dengan metode 'SLSQP' atau \texttt{cvxpy} untuk meminimalkan $f(x,y) = x^2 + y^2$ dengan kendala $x + y \ge 2$. Bandingkan hasilnya dengan perhitungan manual KKT.
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 7: KKT}\maketitle
\section*{Soal 1 (C3)} Rumuskan fungsi Lagrangian dan cari kondisi KKT untuk: Minimalkan $x_1^2 + x_2^2$ subject to $x_1 + x_2 - 4 \le 0$ dan $x_1 \ge 1$.
\section*{Soal 2 (C4)} Selesaikan sistem persamaan KKT dari Soal 1 secara analitik.
\end{document}"""
    },
    "09": {
        "title": "Optimisasi Integer & Campuran (MILP)",
        "desc": "Variabel biner untuk keputusan 'ya/tidak'. Masalah Knapsack, Penjadwalan.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 9: Mixed Integer Linear Programming (MILP)}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Variabel Diskrit dalam Keputusan}
Dalam teknik elektro, banyak keputusan bersifat \textit{diskrit} atau logika \textit{Ya/Tidak}.
\begin{itemize}
\item Apakah pembangkit dihidupkan (1) atau dimatikan (0)? (Unit Commitment).
\item Jumlah kabel yang harus dibentangkan (bilangan bulat).
\end{itemize}
MILP memungkinkan kita memodelkan logika sekuensial dan pemilihan ini secara eksak, sering diselesaikan dengan metode \textit{Branch-and-Bound}.
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 9: MILP}\maketitle
\section{Formulasi Variabel Biner}
Variabel biner $x \in \{0,1\}$ sangat kuat. Contoh: Kendala "Jika A aktif, B harus mati" dimodelkan sebagai $x_A + x_B \le 1$. "Biaya setup" atau \textit{fixed charge} dimodelkan dengan $C \cdot x + c \cdot y$, di mana $y$ adalah jumlah produksi.
\section{Metode Branch-and-Bound}
Algoritma utama solver MILP. Memecah masalah ke dalam sub-masalah dengan pembatasan nilai integer secara hierarkis (Tree).
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 9: Knapsack Problem}\maketitle
\section*{Aktivitas: MILP dengan JuMP/Python}
Pilih perangkat telekomunikasi untuk dipasang di sebuah menara. Kapasitas beban menara 150 kg. 
Antena A: 40 kg, cakupan 10 km. Antena B: 50 kg, cakupan 12 km. Antena C: 70 kg, cakupan 18 km.
Tuliskan model MILP-nya di Python/Julia untuk memaksimalkan cakupan.
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 9: Formulasi MILP}\maketitle
\section*{Soal 1 (C3)} Diberikan variabel kontinu $x \ge 0$ dan biner $y$. Formulasikan kendala logika: "Jika $y=0$ maka $x=0$, jika $y=1$ maka $x \le 100$" menggunakan teknik \textit{Big-M}.
\section*{Soal 2 (C4)} Modelkan permasalahan Unit Commitment sederhana untuk 3 generator dengan biaya start-up dan batas daya minimum menggunakan MILP.
\end{document}"""
    },
    "10": {
        "title": "Algoritma Genetika (GA)",
        "desc": "Pengantar Metaheuristik: Representasi kromosom, seleksi, crossover, mutasi.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 10: Algoritma Genetika (GA)}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Mengapa Metaheuristik?}
Untuk masalah kompleks (NP-Hard) atau non-cembung yang sangat sulit bagi solver tradisional.
\textbf{Algoritma Genetika:} Terinspirasi dari evolusi biologis Darwin.
Tahapan:
1. Inisialisasi Populasi (Kromosom/Gen).
2. Evaluasi Fitness.
3. Seleksi (Roulette wheel, Tournament).
4. Crossover (Kawin silang).
5. Mutasi.
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 10: Algoritma Genetika}\maketitle
\section{Mekanisme GA}
GA adalah metode optimisasi berbasis populasi. Kunci keberhasilannya ada pada keseimbangan antara \textit{Exploration} (mutasi menjelajahi area baru) dan \textit{Exploitation} (crossover dari gen-gen terbaik). Fungsi fitness menentukan probabilitas suatu individu untuk 'bertahan hidup' dan menghasilkan keturunan.
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 10: Implementasi GA Sederhana}\maketitle
\section*{Aktivitas: Memaksimalkan Fungsi Kinerja}
Gunakan \textit{library} \texttt{deap} (Python) atau \texttt{Evolutionary.jl} (Julia) untuk mencari nilai $x \in [-10, 10]$ yang memaksimalkan fungsi $f(x) = x \sin(x) + x \cos(2x)$. Evaluasi pengaruh ukuran populasi terhadap kecepatan konvergensi.
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 10: Konsep GA}\maketitle
\section*{Soal 1 (C2)} Apa dampak dari laju mutasi (mutation rate) yang terlalu tinggi terhadap pencarian solusi pada GA?
\section*{Soal 2 (C3)} Jika kita memiliki kromosom biner: P1=[1,0,1,0,1] dan P2=[0,1,1,1,0]. Tunjukkan hasil Crossover satu-titik (One-point crossover) jika titik potongnya berada setelah gen kedua!
\end{document}"""
    },
    "11": {
        "title": "Studi Kasus 1: Desain Filter Digital dengan GA",
        "desc": "Aplikasi GA untuk optimisasi filter.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 11: Desain Filter Digital dengan GA}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Aplikasi GA pada Filter Digital}
\begin{itemize}
\item Desain filter IIR/FIR sering memiliki fungsi error dengan banyak minimum lokal.
\item GA digunakan untuk mencari koefisien filter yang meminimalkan simpangan antara respons frekuensi riil dan respons ideal (magnitude error).
\end{itemize}
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 11: Desain Filter Digital}\maketitle
\section{Formulasi Masalah Filter}
Fungsi fitness dibentuk dari kebalikan nilai error (Mean Squared Error). Representasi kromosom adalah array bilangan riil (Real-coded GA) yang berisi nilai koefisien $a_k$ dan $b_k$ dari filter.
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 11: Desain Filter FIR}\maketitle
\section*{Aktivitas: Real-coded GA untuk Filter}
Berdasarkan modul \texttt{scipy.signal.freqz}, tulis fungsi fitness yang menghitung error stopband dari filter FIR berorde 10. Optimalkan menggunakan GA.
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 11: Aplikasi Filter}\maketitle
\section*{Soal 1 (C4)} Modelkan fungsi penalti jika koefisien filter $a_1$ melewati batas kestabilan filter IIR.
\section*{Soal 2 (C5)} Bandingkan secara konseptual keunggulan mendesain filter dengan algoritma Parks-McClellan (tradisional) vs Algoritma Genetika.
\end{document}"""
    },
    "12": {
        "title": "Particle Swarm Optimization (PSO)",
        "desc": "Konsep kecerdasan kawanan.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 12: Particle Swarm Optimization (PSO)}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{PSO: Konsep Dasar}
Terinspirasi dari pergerakan kawanan burung atau ikan.
Setiap kandidat solusi adalah "partikel" yang memiliki kecepatan dan posisi di ruang solusi.
Update Kecepatan dipengaruhi oleh:
\begin{itemize}
\item Inersia (mempertahankan arah saat ini).
\item \textit{Personal Best} / pbest (pengalaman memori terbaik individu).
\item \textit{Global Best} / gbest (pengalaman memori terbaik kelompok).
\end{itemize}
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 12: PSO}\maketitle
\section{Persamaan PSO}
Persamaan kecepatan: $v_{i}(t+1) = w \cdot v_{i}(t) + c_1 r_1 (pbest_i - x_i(t)) + c_2 r_2 (gbest - x_i(t))$ \\
Persamaan posisi: $x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)$. \\
Berbeda dengan GA, PSO tidak membuang individu buruk, melainkan mengarahkan ulang seluruh partikel ke arah yang lebih baik secara matematis riil.
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 12: Eksperimen PSO}\maketitle
\section*{Aktivitas: Implementasi PSO Dasar}
Buat \textit{loop} PSO sederhana di Python/Julia untuk menguji fungsi Rosenbrock. Bereksperimenlah dengan mengubah bobot kognitif ($c_1$) dan sosial ($c_2$).
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 12: Parameter PSO}\maketitle
\section*{Soal 1 (C3)} Apa yang terjadi pada pergerakan \textit{swarm} jika kita mengatur bobot inersia $w = 0$?
\section*{Soal 2 (C4)} Dalam konteks ruang solusi multidimensi kontinu, mengapa PSO sering kali lebih cepat konvergen daripada GA standar?
\end{document}"""
    },
    "13": {
        "title": "Studi Kasus 2: Economic Dispatch dengan PSO",
        "desc": "Aplikasi PSO pada sistem tenaga listrik.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 13: Economic Dispatch dengan PSO}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Economic Dispatch (ED)}
\textbf{Tujuan:} Menentukan daya output (MW) tiap generator agar biaya bahan bakar minimum dan beban sistem terpenuhi.
\textbf{Mengapa PSO?} Kurva biaya generator riil memiliki \textit{valve-point effects} (efek non-linier/non-convex ripple) yang membuat metode turunan klasik gagal menemukan optimum global. PSO sangat andal untuk mengatasi masalah ini.
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 13: ED \& PSO}\maketitle
\section{Formulasi ED dan Fungsi Penalti}
Dalam algoritma tak terkendala seperti PSO dasar, batasan (seperti $P_{total} = Beban$) dipaksa dengan menggunakan Fungsi Penalti pada \textit{fitness evaluation}. $Fitness = BiayaTotal + K \cdot |P_{total} - Beban|^2$. Partikel yang melanggar keseimbangan daya akan terkena penalti nilai fitness memburuk drastis.
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{LKM 13: Kasus ED 3 Generator}\maketitle
\section*{Aktivitas: Optimisasi ED dengan PSO}
Diberikan koefisien biaya 3 generator. Implementasikan PSO untuk memenuhi beban 400 MW. Terapkan fungsi penalti yang sesuai.
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Problem Set 13: ED Lanjutan}\maketitle
\section*{Soal 1 (C4)} Rumuskan modifikasi evaluasi fitness PSO untuk memasukkan rugi-rugi jaringan transmisi (Losses) dengan matriks B-Coefficient.
\section*{Soal 2 (C6)} Rancanglah skema "Hybrid PSO-Local Search" untuk mempercepat penemuan solusi ED yang tepat di garis batasan daya ekuivalen.
\end{document}"""
    },
    "14": {
        "title": "Pengerjaan Proyek Akhir",
        "desc": "Panduan perumusan dan pengumpulan data proyek.",
        "slides": r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{tcolorbox}
\definecolor{UNIBBlue}{RGB}{0, 51, 102}\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}
\title{Optimisasi untuk Teknik Elektro}\subtitle{Minggu 14: Pengerjaan Proyek Akhir}
\date{Semester Ganjil 2026}\begin{document}\begin{frame}\titlepage\end{frame}
\begin{frame}{Kick-off Proyek Akhir}
Tujuan: Mengaplikasikan 1 metode optimisasi pada 1 masalah riil keteknikan.
\begin{itemize}
\item Pilih Topik (dari Katalog RPS atau ajukan sendiri).
\item Formulasikan Variabel, Fungsi Objektif, dan Kendala.
\item Pilih Software/Solver yang tepat.
\item Presentasi di Minggu 15.
\end{itemize}
\end{frame}\end{document}""",
        "notes": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Catatan Kuliah 14: Panduan Proyek}\maketitle
\section{Kerangka Pelaporan Proyek}
Laporan wajib mencakup: Latar Belakang Masalah, Model Matematis (jelaskan definisi tiap simbol), Eksperimen Komputasional (Parameter setting), Hasil dan Interpretasi Teknik (bukan sekadar angka, apa maknanya secara engineering).
\end{document}""",
        "worksheet": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Lembar Kerja Proyek - Minggu 14}\maketitle
\section*{Form Proposal Proyek}
1. Judul Proyek: \\
2. Algoritma Pilihan: \\
3. Deskripsi Model Matematis: \\
\textit{Isi dan konsultasikan ke Dosen hari ini.}
\end{document}""",
        "problem_set": r"""\documentclass[11pt, a4paper]{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage[margin=1in]{geometry}\begin{document}\title{Checklist Proyek (Tidak Dinilai Terpisah)}\maketitle
Gunakan waktu ini untuk mengumpulkan data riil atau data referensi IEEE/Benchmark untuk proyek Anda. Pastikan kode Anda bisa berjalan \textit{bug-free} hari ini.
\end{document}"""
    }
}

base_dir = "/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026"

for week, files in materials.items():
    create_file(os.path.join(base_dir, f"slides/week{week}_slides.tex"), files["slides"])
    create_file(os.path.join(base_dir, f"notes/week{week}_notes.tex"), files["notes"])
    create_file(os.path.join(base_dir, f"worksheets/week{week}_worksheet.tex"), files["worksheet"])
    create_file(os.path.join(base_dir, f"problem_sets/week{week}_problem_set.tex"), files["problem_set"])

# Compile all
import glob
for doc_type in ['slides', 'notes', 'worksheets', 'problem_sets']:
    for tex_file in glob.glob(os.path.join(base_dir, doc_type, 'week*.tex')):
        subprocess.run(['pdflatex', '-output-directory=../pdf_exports', os.path.basename(tex_file)], cwd=os.path.dirname(tex_file))
