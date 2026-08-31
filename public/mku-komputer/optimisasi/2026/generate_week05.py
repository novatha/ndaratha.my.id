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
\subtitle{Minggu 5: Optimisasi Non-Linier (NLP) Tak Terkendala}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Semester Ganjil 2026}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Tujuan Pembelajaran (CPMK-1, CPMK-3)}
    \begin{itemize}
        \item Mampu menjelaskan perbedaan mendasar antara optimisasi linier dan non-linier.
        \item Mampu mengimplementasikan algoritma Metode Penurunan Tercuram (Steepest/Gradient Descent).
        \item Mampu menjelaskan konsep dan mengimplementasikan Metode Newton untuk optimisasi tak terkendala.
        \item Mampu mengevaluasi kelebihan dan kekurangan dari masing-masing metode berbasis gradien.
    \end{itemize}
\end{frame}

\begin{frame}{Mengapa Non-Linier?}
    Banyak fenomena fisik di Teknik Elektro yang tidak linier:
    \begin{itemize}
        \item \textbf{Daya Aktif ($P$) dan Reaktif ($Q$)} pada aliran daya merupakan fungsi trigonometri non-linier dari sudut fase dan magnitudo tegangan.
        \item \textbf{Kerugian daya (Losses)} sebanding dengan kuadrat arus ($I^2R$).
        \item \textbf{Kapasitas Kanal Shannon} adalah fungsi logaritmik dari Signal-to-Noise Ratio (SNR).
    \end{itemize}
    \textbf{Fokus Minggu Ini:} Pencarian nilai minimum untuk fungsi non-linier \textit{tanpa} adanya batasan ruang (Tak Terkendala / Unconstrained).
\end{frame}

\begin{frame}{Gradient Descent (Penurunan Tercuram)}
    \textbf{Ide Dasar:} Jika kita berada di sebuah pegunungan dalam keadaan buta, bagaimana cara tercepat untuk mencapai lembah terbawah?
    \textit{Jawab: Selalu melangkah ke arah yang paling curam ke bawah.}
    \vspace{0.3cm}
    \begin{tcolorbox}[colback=blue!5,colframe=UNIBBlue,title=Aturan Pembaruan (Update Rule)]
        $x_{k+1} = x_k - \alpha \nabla f(x_k)$
    \end{tcolorbox}
    \begin{itemize}
        \item $\nabla f(x_k)$: Gradien (turunan pertama) dari fungsi evaluasi di titik saat ini. (Menunjukkan arah kenaikan paling terjal).
        \item $\alpha$: \textit{Learning rate} atau ukuran langkah (step size). Menentukan seberapa jauh kita melangkah.
        \item Tanda negatif $(-)$ mengarahkan kita \textit{berlawanan} dengan gradien (turun).
    \end{itemize}
\end{frame}

\begin{frame}{Isu pada Gradient Descent}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \textbf{Pemilihan $\alpha$ sangat krusial:}
            \begin{itemize}
                \item Jika terlalu \textbf{kecil}: Konvergensi sangat lambat.
                \item Jika terlalu \textbf{besar}: Dapat melompati minimum dan bahkan divergen (osilasi).
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \textbf{Sifat Bentang Alam (Landscape):}
            \begin{itemize}
                \item Sangat lambat saat berada di area yang relatif datar (plateau).
                \item Rawan terjebak di Minimum Lokal (bukan Global) pada fungsi non-cembung.
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Metode Newton}
    \textbf{Ide Dasar:} Menggunakan informasi turunan kedua untuk memprediksi kelengkungan fungsi, sehingga langkah pencarian bisa lebih akurat dan konvergensinya lebih cepat.
    \vspace{0.3cm}
    \begin{tcolorbox}[colback=yellow!5,colframe=UNIBYellow,title=Aturan Pembaruan Metode Newton]
        $x_{k+1} = x_k - [H(x_k)]^{-1} \nabla f(x_k)$
    \end{tcolorbox}
    \begin{itemize}
        \item $\nabla f(x_k)$: Gradien (vektor turunan pertama).
        \item $H(x_k)$: Matriks Hessian (matriks dari turunan parsial kedua).
        \item Tidak memerlukan parameter $\alpha$ (meskipun terkadang dimodifikasi dengan Damped Newton).
    \end{itemize}
\end{frame}

\begin{frame}{Gradient Descent vs Metode Newton}
    \begin{table}[]
    \begin{tabular}{@{}p{3.5cm}p{4cm}p{4cm}@{}}
    \toprule
    \textbf{Karakteristik} & \textbf{Gradient Descent} & \textbf{Metode Newton} \\ \midrule
    Informasi Turunan & Turunan ke-1 (Gradien) & Turunan ke-1 \& ke-2 (Hessian) \\ 
    Kecepatan Konvergensi & Linier (Lambat di dekat optimum) & Kuadratik (Sangat cepat di dekat optimum) \\ 
    Beban Komputasi per Iterasi & Rendah ($O(N)$) & Tinggi ($O(N^3)$ karena invers Hessian) \\ 
    Kestabilan & Tergantung $\alpha$ & Rentan jika Hessian tidak \textit{Positive Definite} \\ \bottomrule
    \end{tabular}
    \end{table}
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
\lhead{Catatan Kuliah - Minggu 5}
\cfoot{\thepage}

\title{\vspace{-1cm}\color{UNIBBlue}\textbf{Optimisasi Non-Linier Tak Terkendala}\vspace{-0.5cm}}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Minggu 5}

\begin{document}
\maketitle

\section{Pendahuluan}
Banyak permasalahan keteknikan memiliki fungsi objektif yang kompleks dan non-linier. Jika kita melepaskan semua batasan operasional (unconstrained), kita hanya fokus pada pencarian titik dasar di mana fungsi bernilai minimum (atau maksimum). Syarat mutlak (Necessary Condition) untuk suatu titik optimal tak terkendala adalah nilai turunan pertamanya (Gradien) sama dengan nol: $\nabla f(x) = 0$.

\section{Metode Gradient Descent}
Metode ini adalah algoritma iteratif urutan pertama. Dari posisi tebakan awal $x_0$, metode ini melangkah berlawanan dengan arah gradien.
\[ x_{k+1} = x_k - \alpha \nabla f(x_k) \]

\subsection{Pemilihan Learning Rate ($\alpha$)}
\begin{itemize}
    \item \textbf{Konstan}: $\alpha$ ditetapkan di awal. Sederhana, namun bisa menyebabkan overshoot jika terlalu besar atau sangat lambat jika terlalu kecil.
    \item \textbf{Line Search}: Mengubah algoritma pencarian menjadi pencarian 1-Dimensi di setiap iterasi. Kita mencari $\alpha$ yang meminimalkan $f(x_k - \alpha \nabla f(x_k))$. Jauh lebih stabil namun menambah beban komputasi per iterasi.
\end{itemize}

\section{Metode Newton}
Metode Newton adalah metode urutan kedua yang mengeksploitasi matriks Hessian (kelengkungan). Metode ini mengasumsikan fungsi secara lokal berbentuk kuadratik.
\[ x_{k+1} = x_k - H(x_k)^{-1} \nabla f(x_k) \]
Syarat utama agar Metode Newton dapat mengarah ke \textbf{minimum} adalah Matriks Hessian $H(x_k)$ harus \textbf{Positive Definite} di sekitar titik tersebut. Jika Hessian tidak invertible (singular) atau tidak positive definite, metode ini dapat bergerak menjauhi minimum atau bahkan divergen.

\subsection{Tantangan Komputasional Hessian}
Untuk masalah dengan ribuan variabel (seperti pelatihan Jaringan Saraf Tiruan), menghitung dan membalikkan matriks Hessian $N \times N$ sangatlah mahal. Oleh karena itu, di ranah machine learning dan optimisasi skala besar, metode \textbf{Quasi-Newton} (seperti BFGS atau L-BFGS) lebih disukai karena mereka mengaproksimasi invers Hessian menggunakan pembaruan gradien dari iterasi sebelumnya tanpa perlu menghitung turunan kedua secara eksplist.

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
\lhead{Lembar Kerja Mahasiswa (LKM) - Minggu 5}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{LEMBAR KERJA MAHASISWA (LKM)}\\
    \large\textbf{Minggu 5: Implementasi Algoritma Berbasis Gradien}
\end{center}

\vspace{0.5cm}
\textbf{Nama Praktikan :} \rule{6cm}{0.4pt} \\
\textbf{NPM :} \rule{6.8cm}{0.4pt}

\section*{Tujuan Praktikum}
1. Mampu mengimplementasikan algoritma Gradient Descent dari awal (from scratch). \\
2. Mampu menganalisis pengaruh \textit{learning rate} terhadap konvergensi.

\section*{Kasus Uji: Meminimalkan Fungsi 2D Sederhana}
Misalkan kita memiliki fungsi biaya operasional sistem:
\[ f(x, y) = x^2 + 2y^2 + 2x - 4y + 5 \]

\section*{Aktivitas 1: Turunan Parsial (C3)}
Tentukan turunan parsial terhadap $x$ dan $y$:
\begin{align*}
    \frac{\partial f}{\partial x} &= \ldots \\
    \frac{\partial f}{\partial y} &= \ldots
\end{align*}

\section*{Aktivitas 2: Implementasi Python (C4)}
Ketik dan jalankan kode berikut di Jupyter Notebook/Google Colab:

\begin{lstlisting}[language=Python]
import numpy as np

# Definisikan fungsi dan gradien
def f(x, y):
    return x**2 + 2*y**2 + 2*x - 4*y + 5

def grad_f(x, y):
    df_dx = 2*x + 2
    df_dy = 4*y - 4
    return np.array([df_dx, df_dy])

# Hyperparameters
alpha = 0.1 # Learning rate
epochs = 50
p = np.array([5.0, 5.0]) # Tebakan awal (Initial guess)

history = []

for i in range(epochs):
    gradient = grad_f(p[0], p[1])
    p = p - alpha * gradient
    history.append(f(p[0], p[1]))
    if i % 10 == 0:
        print(f"Iterasi {i}: Koordinat = {p}, f(x,y) = {f(p[0], p[1]):.4f}")

print(f"Optimal di temukan di: {p}")
\end{lstlisting}

\section*{Aktivitas 3: Analisis Konvergensi (C5)}
Ubah nilai `alpha` pada program di atas menjadi:
a) `alpha = 0.01` \\
b) `alpha = 0.5` \\
c) `alpha = 1.0` \\

Amati keluaran setiap skenario. Apa yang terjadi pada nilai koordinat dan $f(x,y)$ pada setiap kondisi?

\vspace{1cm}
\textbf{Laporan Pengamatan:} \\
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
\lhead{Problem Set - Minggu 5}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{PROBLEM SET}\\
    \large\textbf{Minggu 5: Optimisasi Non-Linier (NLP) Tak Terkendala}
\end{center}

\vspace{0.5cm}

\textbf{Instruksi:} Kerjakan soal-soal di bawah ini dengan lengkap.

\section*{Soal 1: Metode Newton 1D (C3)}
Diberikan fungsi $f(x) = x^4 - 3x^3 + 2$.
Lakukan 2 iterasi Metode Newton secara manual dengan tebakan awal $x_0 = 3$. 
Tunjukkan nilai $f'(x)$, $f''(x)$, dan $x_{k+1}$ pada setiap iterasi.

\section*{Soal 2: Hessian Matriks (C4)}
Diberikan fungsi biaya dengan dua variabel $f(x_1, x_2) = 2x_1^2 + x_2^2 - x_1 x_2 - 4x_1$. \\
(a) Tentukan vektor gradien $\nabla f(x_1, x_2)$. \\
(b) Tentukan Matriks Hessian $H(x_1, x_2)$. \\
(c) Apakah Matriks Hessian tersebut \textit{Positive Definite}? Buktikan menggunakan nilai eigen (eigenvalues) atau determinan principal minor.

\section*{Soal 3: Komparasi dan Evaluasi (C5)}
Dalam aplikasi estimasi kanal (Channel Estimation) untuk telekomunikasi 5G, seorang insinyur ingin memperbarui bobot filter secara \textit{real-time} (setiap mikrodetik). Model error yang dioptimasi memiliki tingkat kelengkungan yang cukup konsisten, namun jumlah parameternya (variabel) mencapai $10^4$ (sepuluh ribu).

(C5) Metode mana yang akan Anda rekomendasikan untuk di-deploy ke perangkat keras (FPGA): Gradient Descent murni, Metode Newton, atau varian Quasi-Newton? Evaluasi kelebihan dan kekurangan ketiganya dalam konteks kebutuhan real-time dan ukuran variabel!

\section*{Soal 4: Sintesis Solusi Masalah Kegagalan (C6)}
Algoritma Gradient Descent yang diimplementasikan pada kontroler cerdas sebuah robot penjejak garis ternyata berosilasi (bergetar maju mundur) secara tidak terkendali di sekitar garis target dan tidak mau diam (konvergen).

(C6) Rancanglah dua buah modifikasi (strategi) secara matematis pada algoritma pembaruan (update rule) untuk memecahkan masalah osilasi ini. Jelaskan mekanisme kerja masing-masing strategi secara logis.

\end{document}
"""

create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/slides/week05_slides.tex', slides_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/notes/week05_notes.tex', notes_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/worksheets/week05_worksheet.tex', worksheet_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/problem_sets/week05_problem_set.tex', problem_set_content)

print("Week 5 files created successfully.")
