import os

def create_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

# Slides
slides_content = r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tcolorbox}

% Theme styling based on UNIB colors
\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\definecolor{UNIBYellow}{RGB}{255, 204, 0}
\setbeamercolor{palette primary}{bg=UNIBBlue,fg=white}
\setbeamercolor{palette secondary}{bg=UNIBYellow,fg=black}
\setbeamercolor{structure}{fg=UNIBBlue}
\setbeamercolor{title}{fg=UNIBBlue}
\setbeamercolor{frametitle}{bg=UNIBBlue,fg=white}

\title{Optimisasi untuk Teknik Elektro}
\subtitle{Minggu 3: Metode Simpleks \& Dualitas}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Semester Ganjil 2026}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Tujuan Pembelajaran (CPMK-1, CPMK-3)}
    \begin{itemize}
        \item Mampu merubah bentuk standar program linier dengan variabel slack/surplus.
        \item Mampu menjelaskan konsep dan iterasi Metode Simpleks secara konseptual.
        \item Mampu mendefinisikan masalah dual dari masalah primal program linier.
        \item Mampu menginterpretasikan nilai dual secara ekonomi dalam konteks keteknikan.
    \end{itemize}
\end{frame}

\begin{frame}{Bentuk Standar Program Linier}
    \textbf{Semua kendala harus berupa persamaan ($=$) dengan ruas kanan non-negatif.}
    \vspace{0.5cm}
    \begin{itemize}
        \item Kendala $\le$ (kurang dari atau sama dengan) ditambahkan \textbf{Variabel Slack} (waktu/sumber daya yang tersisa).
        \item Kendala $\ge$ (lebih dari atau sama dengan) dikurangi \textbf{Variabel Surplus} (kelebihan di atas minimum).
    \end{itemize}
\end{frame}

\begin{frame}{Contoh Bentuk Standar}
    \textbf{Primal:}
    Maksimumkan $Z = 3x_1 + 5x_2$ \\
    s.t. \\
    $x_1 \le 4$ \\
    $2x_2 \le 12$ \\
    $3x_1 + 2x_2 \le 18$ \\
    $x_1, x_2 \ge 0$
    
    \vspace{0.5cm}
    \textbf{Bentuk Standar:}
    Maksimumkan $Z = 3x_1 + 5x_2 + 0s_1 + 0s_2 + 0s_3$ \\
    s.t. \\
    $x_1 + s_1 = 4$ \\
    $2x_2 + s_2 = 12$ \\
    $3x_1 + 2x_2 + s_3 = 18$ \\
    $x_1, x_2, s_1, s_2, s_3 \ge 0$
\end{frame}

\begin{frame}{Metode Simpleks (Konseptual)}
    \begin{itemize}
        \item Pendekatan aljabar untuk mencari solusi optimal pada titik-titik sudut (extreme points) ruang solusi yang feasible.
        \item Iterasi bergerak dari satu solusi dasar feasible (BFS) ke BFS tetangganya yang memberikan nilai fungsi tujuan yang lebih baik.
        \item \textbf{Langkah Utama:}
        \begin{enumerate}
            \item Tentukan BFS awal (biasanya di titik asal).
            \item Cek optimalitas (Apakah ada variabel non-basis yang jika dimasukkan ke basis akan memperbaiki Z?).
            \item Tentukan variabel masuk (entering variable) dan variabel keluar (leaving variable).
            \item Lakukan operasi baris dasar (pivoting) untuk mendapatkan matriks BFS baru.
            \item Ulangi hingga kondisi optimal tercapai.
        \end{enumerate}
    \end{itemize}
\end{frame}

\begin{frame}{Dualitas (Duality)}
    \textbf{Setiap masalah program linier (Primal) memiliki masalah terkait yang disebut Dual.}
    \vspace{0.3cm}
    \begin{itemize}
        \item Solusi optimal Primal dan Dual memiliki nilai fungsi tujuan yang sama (Strong Duality).
        \item Memberikan perspektif ekonomi pada masalah.
        \item Variabel Dual ($y_i$) merepresentasikan \textit{shadow price} atau nilai batas dari sumber daya (kendala $i$).
    \end{itemize}
\end{frame}

\begin{frame}{Interpretasi Ekonomi Dualitas di Teknik Elektro}
    Contoh: Optimisasi alokasi daya dari beberapa generator untuk memenuhi beban (Primal: Minimisasi Biaya).
    \vspace{0.5cm}
    \begin{tcolorbox}[colback=UNIBYellow!20,colframe=UNIBBlue,title=Makna Variabel Dual (Shadow Price)]
        Berapa banyak \textbf{penghematan biaya} (atau pengurangan total objektif) yang bisa didapatkan jika batas kendala kapasitas daya diperlonggar sebesar 1 unit (1 MW)?
    \end{tcolorbox}
    \begin{itemize}
        \item Jika generator A sudah pada kapasitas maksimum, shadow price-nya positif (ada manfaat ekonomi jika kapasitas ditambah).
        \item Jika generator B belum mencapai batas, shadow price-nya nol (menambah batas kapasitas B tidak mengubah solusi optimal).
    \end{itemize}
\end{frame}

\begin{frame}{Kesimpulan}
    \begin{itemize}
        \item Bentuk standar (slack/surplus) diperlukan untuk algoritma aljabar seperti Simpleks.
        \item Metode Simpleks mencari solusi efisien di titik-titik sudut.
        \item Dualitas memberikan nilai "harga bayangan" (shadow price) untuk setiap sumber daya atau batas operasional, sangat berharga dalam keputusan investasi keteknikan.
    \end{itemize}
\end{frame}

\end{document}
"""

# Notes
notes_content = r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{tcolorbox}
\usepackage{fancyhdr}
\usepackage{hyperref}

\definecolor{UNIBBlue}{RGB}{0, 51, 102}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Catatan Kuliah - Minggu 3}
\cfoot{\thepage}

\title{\vspace{-1cm}\color{UNIBBlue}\textbf{Metode Simpleks \& Dualitas}\vspace{-0.5cm}}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Minggu 3}

\begin{document}
\maketitle

\section{Pendahuluan}
Setelah kita mempelajari metode grafis pada minggu lalu, kita menyadari batasan utamanya: metode grafis hanya praktis untuk masalah dengan 2 variabel (atau paling banyak 3). Di dunia teknik elektro nyata, masalah optimisasi bisa memiliki ribuan variabel dan kendala. Metode Simpleks (Simplex Method) adalah algoritma aljabar efisien yang dapat menangani masalah LP berskala besar.

\section{Bentuk Standar Program Linier}
Syarat utama algoritma Simpleks adalah masalah harus dalam \textbf{Bentuk Standar}:
1. Semua kendala berupa persamaan ($=$).
2. Sisi kanan persamaan harus bernilai non-negatif.
3. Semua variabel harus non-negatif.

\subsection{Variabel Slack dan Surplus}
Untuk mengubah pertidaksamaan menjadi persamaan:
\begin{itemize}
    \item \textbf{Variabel Slack ($s \ge 0$)}: Ditambahkan pada kendala $\le$. Mewakili jumlah sisa atau kapasitas yang tidak terpakai dari sumber daya tertentu.
    \item \textbf{Variabel Surplus ($e \ge 0$)}: Dikurangkan pada kendala $\ge$. Mewakili kelebihan output di atas syarat minimum.
\end{itemize}

\section{Metode Simpleks (Garis Besar)}
Algoritma simpleks bekerja dengan cara berpindah dari satu titik sudut (extreme point) dasar yang feasible ke titik sudut berdekatan yang memperbaiki nilai fungsi objektif.
\begin{enumerate}
    \item \textbf{Inisialisasi}: Menentukan Basic Feasible Solution (BFS) awal, biasanya titik (0,0) di mana semua variabel Slack merupakan variabel basis, dan variabel asli bernilai nol (non-basis).
    \item \textbf{Uji Optimalitas}: Menggunakan baris-$Z$, dicek apakah terdapat koefisien negatif (untuk masalah maksimasi). Jika tidak ada koefisien negatif di baris-$Z$, solusi telah optimal.
    \item \textbf{Pivoting}:
    \begin{itemize}
        \item \textbf{Kolom Pivot (Variabel Masuk)}: Pilih variabel non-basis dengan koefisien negatif terbesar pada baris-$Z$.
        \item \textbf{Baris Pivot (Variabel Keluar)}: Pilih baris dengan rasio minimum (Nilai Ruas Kanan / Koefisien di Kolom Pivot yang $>0$).
        \item Lakukan operasi baris elementer untuk membentuk matriks identitas baru pada kolom pivot.
    \end{itemize}
\end{enumerate}

\section{Dualitas}
Setiap model Program Linier (Primal) berkorelasi dengan sebuah model lain yang disebut \textbf{Dual}.

\begin{tcolorbox}[colback=blue!5,colframe=UNIBBlue,title=Aturan Primal-Dual Dasar]
\begin{itemize}
    \item Jika Primal adalah \textbf{Maksimisasi}, Dual adalah \textbf{Minimisasi}.
    \item Jika Primal memiliki $n$ variabel dan $m$ kendala, Dual akan memiliki $m$ variabel dan $n$ kendala.
    \item Nilai ruas kanan kendala Primal menjadi koefisien fungsi objektif Dual.
    \item Koefisien fungsi objektif Primal menjadi ruas kanan kendala Dual.
\end{itemize}
\end{tcolorbox}

\subsection{Shadow Price (Harga Bayangan) dalam Keteknikan}
Nilai optimal dari variabel Dual (dinotasikan sebagai $y_i$) sering disebut \textbf{Shadow Price}. Shadow price merepresentasikan peningkatan/penurunan marjinal dari nilai fungsi tujuan optimal (Z) per unit kenaikan batasan kapasitas kendala (ruas kanan $b_i$).

Dalam teknik elektro, misal pada masalah perutean jaringan komunikasi (minimasi delay): shadow price pada sebuah link transmisi yang jenuh (congestion) menunjukkan berapa banyak delay total akan berkurang jika kapasitas bandwidth link tersebut ditambah 1 Mbps. Informasi ini sangat berguna bagi insinyur dalam memprioritaskan mana infrastruktur yang harus di-upgrade.

\end{document}
"""

# Worksheet
worksheet_content = r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{tcolorbox}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\definecolor{UNIBBlue}{RGB}{0, 51, 102}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Lembar Kerja Mahasiswa (LKM) - Minggu 3}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{LEMBAR KERJA MAHASISWA (LKM)}\\
    \large\textbf{Minggu 3: Analisis Solusi LP \& Ekstraksi Output Dual}
\end{center}

\vspace{0.5cm}
\textbf{Nama Praktikan :} \rule{6cm}{0.4pt} \\
\textbf{NPM :} \rule{6.8cm}{0.4pt}

\section*{Tujuan Praktikum}
1. Mahasiswa mampu menggunakan Julia/JuMP atau Python/SciPy untuk memecahkan model LP. \\
2. Mahasiswa mampu mengekstraksi dan menginterpretasikan nilai \textit{dual variable} (shadow price).

\section*{Kasus: Alokasi Frekuensi Komunikasi (Dimensi C3 - C4)}
Sebuah menara BTS melayani dua jenis spektrum layanan: Layanan Voice ($x_1$) dan Layanan Data Broadband ($x_2$). Keuntungan per Mbps alokasi Voice adalah \$40 dan Data Broadband adalah \$50.
Terdapat batasan perangkat keras:
1. Total bandwidth prosesor sinyal (DSP) maksimal 200 Mbps. ($x_1 + 2x_2 \le 200$)
2. Total daya transmisi maksimal 150 unit. ($2x_1 + x_2 \le 150$)
3. Batas regulasi layanan data maksimal 80 Mbps. ($x_2 \le 80$)

Tujuan: Maksimalkan Total Keuntungan $Z = 40x_1 + 50x_2$.

\section*{Aktivitas 1: Implementasi dengan Julia/JuMP (C3)}
\begin{lstlisting}[language=Python]
using JuMP
using HiGHS

model = Model(HiGHS.Optimizer)
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

@objective(model, Max, 40*x1 + 50*x2)

# Beri nama untuk constraint agar mudah diekstrak nilai dual-nya
@constraint(model, dsp_limit, x1 + 2*x2 <= 200)
@constraint(model, power_limit, 2*x1 + x2 <= 150)
@constraint(model, reg_limit, x2 <= 80)

optimize!(model)

println("Solusi Optimal x1 = ", value(x1))
println("Solusi Optimal x2 = ", value(x2))
println("Keuntungan Maksimal Z = ", objective_value(model))
\end{lstlisting}

\section*{Aktivitas 2: Ekstraksi dan Interpretasi Shadow Price (C4)}
Tambahkan kode berikut pada bagian bawah program di atas:
\begin{lstlisting}[language=Python]
println("Shadow Price Batas DSP = ", dual(dsp_limit))
println("Shadow Price Batas Daya = ", dual(power_limit))
println("Shadow Price Batas Regulasi = ", dual(reg_limit))
\end{lstlisting}

\textbf{Pertanyaan Analisis:}
Berdasarkan nilai \textit{dual variable} (shadow price) yang Anda dapatkan, jelaskan kepada manajemen \textbf{batas sumber daya mana} (DSP, Daya, atau Regulasi) yang harus diprioritaskan untuk di-upgrade kapasitasnya tahun depan agar keuntungan perusahaan meningkat paling besar! Jelaskan alasan kuantitatifnya.

\vspace{2cm}
\textbf{Jawaban:} \\
\rule{\textwidth}{0.4pt} \\
\rule{\textwidth}{0.4pt} \\
\rule{\textwidth}{0.4pt} \\
\rule{\textwidth}{0.4pt}

\end{document}
"""

# Problem Set
problem_set_content = r"""\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{tcolorbox}
\usepackage{fancyhdr}

\definecolor{UNIBBlue}{RGB}{0, 51, 102}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textbf{Optimisasi untuk Teknik Elektro}}
\lhead{Problem Set - Minggu 3}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{PROBLEM SET}\\
    \large\textbf{Minggu 3: Metode Simpleks dan Dualitas}
\end{center}

\vspace{0.5cm}

\textbf{Instruksi:} Kerjakan soal-soal di bawah ini dengan menyertakan langkah-langkah analitis.

\section*{Soal 1: Pemahaman Konsep (C1 - C2)}
1. (C1) Apakah yang dimaksud dengan variabel slack dan variabel surplus dalam optimisasi linier? \\
2. (C2) Jelaskan hubungan antara Solusi Optimal Primal dan Solusi Optimal Dual (Strong Duality Theorem).

\section*{Soal 2: Formulasi Bentuk Standar (C3)}
Diberikan masalah program linier untuk desain \textit{Microgrid} berikut: \\
Minimumkan $C = 10P_g + 2P_w$ \\
s.t. \\
$P_g + P_w \ge 100$ (Pemenuhan Beban Minimum) \\
$P_g \le 80$ (Kapasitas Grid) \\
$2P_g + 5P_w \le 250$ (Batas Emisi dan Kebisingan Ekuivalen) \\
$P_g, P_w \ge 0$

Ubahlah model di atas ke dalam \textbf{Bentuk Standar} dengan menambahkan variabel slack dan surplus yang sesuai!

\section*{Soal 3: Transformasi Primal-Dual (C4)}
Berdasarkan model awal (Primal) pada Soal 2 di atas, formulasikan model \textbf{Dual}-nya secara lengkap (Tujuan, Variabel, dan Kendala)!

\section*{Soal 4: Evaluasi Solusi dan Shadow Price (C5 - C6)}
Sebuah pabrik produksi komponen elektronik memodelkan maksimasi profitnya. Hasil analisis optimisasi menggunakan \textit{software} JuMP menunjukkan bahwa \textit{shadow price} untuk:
\begin{itemize}
    \item Kendala Mesin Solder (Kapasitas 500 jam/minggu) = Rp 150.000 / jam
    \item Kendala Mesin Testing (Kapasitas 300 jam/minggu) = Rp 0 / jam
    \item Kendala Suplai Bahan Baku (Kapasitas 1000 unit/minggu) = Rp 25.000 / unit
\end{itemize}

Pabrik memiliki dana investasi terbatas untuk \textit{upgrade} peralatan.
(C5) Berikan argumentasi mesin mana (Solder atau Testing) yang menguntungkan untuk di-upgrade kapasitasnya, atau apakah lebih baik mencari supplier bahan baku tambahan? 
(C6) Berikan saran kebijakan strategis kepada manajer operasional terkait mesin yang nilai \textit{shadow price}-nya Rp 0.

\end{document}
"""

create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/slides/week03_slides.tex', slides_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/notes/week03_notes.tex', notes_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/worksheets/week03_worksheet.tex', worksheet_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/problem_sets/week03_problem_set.tex', problem_set_content)

print("Week 3 files created successfully.")
