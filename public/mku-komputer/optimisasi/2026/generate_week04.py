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
\subtitle{Minggu 4: Analisis Sensitivitas}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Semester Ganjil 2026}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Tujuan Pembelajaran (CPMK-1, CPMK-4)}
    \begin{itemize}
        \item Mampu menjelaskan konsep dan tujuan dari Analisis Sensitivitas (Sensitivity Analysis/Post-Optimality Analysis).
        \item Mampu menentukan rentang optimalitas (Range of Optimality) untuk koefisien fungsi tujuan.
        \item Mampu menentukan rentang kelayakan (Range of Feasibility) untuk nilai ruas kanan kendala.
        \item Mampu menginterpretasikan laporan sensitivitas dari \textit{solver} komputasional untuk pengambilan keputusan di keteknikan.
    \end{itemize}
\end{frame}

\begin{frame}{Mengapa Analisis Sensitivitas?}
    \textbf{Dunia nyata selalu berubah.} Model program linier didasarkan pada parameter yang diestimasi atau diprediksi, yang bisa saja tidak akurat atau berubah di masa depan.
    \vspace{0.3cm}
    \begin{itemize}
        \item Bagaimana jika harga komponen pemancar radio naik? Apakah desain alokasi daya saat ini masih merupakan yang terbaik?
        \item Bagaimana jika kapasitas bandwidth backhaul diturunkan oleh provider? Seberapa drastis performa jaringan akan turun?
    \end{itemize}
    \textbf{Analisis Sensitivitas} menjawab pertanyaan "Bagaimana jika..." (What-if analysis) \textbf{tanpa} harus menyelesaikan ulang model dari awal.
\end{frame}

\begin{frame}{Dua Jenis Analisis Utama}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{tcolorbox}[colback=blue!5,colframe=UNIBBlue,title=Perubahan Koefisien Tujuan ($c_j$)]
                \textbf{Rentang Optimalitas} \\
                Mencari seberapa besar koefisien profit/biaya suatu variabel dapat berubah sebelum solusi basis (titik sudut optimal) \textit{berubah}. \\
                (Nilai $Z$ akan berubah, tetapi strategi $X$ yang dipilih tetap sama).
            \end{tcolorbox}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{tcolorbox}[colback=yellow!5,colframe=UNIBYellow,title=Perubahan Sisi Kanan ($b_i$)]
                \textbf{Rentang Kelayakan} \\
                Mencari seberapa besar kapasitas/batasan kendala dapat diubah dengan mempertahankan Shadow Price yang konstan. \\
                (Titik optimal \textit{pasti berubah}, tetapi kombinasi variabel aktif dalam basis tetap sama).
            \end{tcolorbox}
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Membaca Laporan Sensitivitas (Contoh)}
    \begin{table}[]
    \centering
    \begin{tabular}{@{}llccc@{}}
    \toprule
    \textbf{Variabel} & \textbf{Final Value} & \textbf{Obj. Coef.} & \textbf{Allowable Increase} & \textbf{Allowable Decrease} \\ \midrule
    $x_1$ (Daya Gen 1) & 100 MW & \$10 & 2.5 & 4.0 \\
    $x_2$ (Daya Gen 2) & 50 MW & \$15 & 1E+30 ($\infty$) & 2.5 \\ \bottomrule
    \end{tabular}
    \end{table}
    \vspace{0.2cm}
    \textbf{Interpretasi Rentang Optimalitas}: \\
    Strategi alokasi daya (Gen 1 = 100 MW, Gen 2 = 50 MW) \textbf{akan tetap optimal} selama biaya operasional Gen 1 berada di rentang: \\
    $\$10 - \$4.0 \le c_1 \le \$10 + \$2.5$ \\
    $\mathbf{\$6.0 \le c_1 \le \$12.5}$
\end{frame}

\begin{frame}{Reduced Cost}
    \textbf{Reduced Cost} berlaku untuk variabel non-basis (variabel yang nilainya 0 pada solusi optimal).
    \vspace{0.3cm}
    \begin{itemize}
        \item Ini mengukur \textbf{seberapa banyak koefisien tujuan harus diperbaiki} (misal biaya diturunkan atau profit dinaikkan) agar variabel tersebut mulai masuk ke dalam solusi optimal ($> 0$).
        \item Jika sebuah generator tidak dihidupkan (daya = 0) karena terlalu mahal, \textit{reduced cost} memberitahu kita: seberapa jauh kita harus memotong biaya operasionalnya agar layak dinyalakan.
    \end{itemize}
\end{frame}

\begin{frame}{Aplikasi di Teknik Elektro}
    \begin{itemize}
        \item \textbf{Perencanaan Investasi:} Evaluasi batasan jaringan listrik mana yang paling sensitif dan memberikan Return on Investment (ROI) tertinggi dari \textit{shadow price}-nya, asalkan penambahan kapasitas masih dalam \textit{allowable increase}.
        \item \textbf{Penjadwalan Beban (Load Dispatch):} Mengetahui ketahanan strategi alokasi daya saat ini jika terjadi fluktuasi mendadak pada harga bahan bakar (analisis koefisien objektif).
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
\lhead{Catatan Kuliah - Minggu 4}
\cfoot{\thepage}

\title{\vspace{-1cm}\color{UNIBBlue}\textbf{Analisis Sensitivitas}\vspace{-0.5cm}}
\author{Ir. Novalio Daratha S.T., M.Sc., Ph.D.}
\date{Minggu 4}

\begin{document}
\maketitle

\section{Pendahuluan}
Analisis Sensitivitas (juga disebut analisis pasca-optimalitas) adalah studi tentang bagaimana perubahan parameter model (seperti koefisien fungsi tujuan dan batas kapasitas) berdampak pada solusi optimal. Dalam praktiknya, insinyur jarang berhadapan dengan data yang 100\% akurat; harga dapat berfluktuasi, dan ketersediaan bandwidth dapat bervariasi. Oleh karena itu, kita perlu mengetahui seberapa kuat (robust) solusi kita terhadap ketidakpastian ini.

\section{Perubahan Koefisien Fungsi Tujuan ($c_j$)}
Perubahan pada $c_j$ memengaruhi kemiringan dari fungsi tujuan pada grafik (untuk masalah 2D).
\begin{itemize}
    \item \textbf{Range of Optimality (Rentang Optimalitas)}: Batas-batas nilai (batas atas dan batas bawah) dari koefisien $c_j$ sedemikian rupa sehingga variabel dasar (kombinasi variabel yang nilainya tidak nol) tidak berubah.
    \item Jika $c_j$ berubah dalam rentang ini, titik sudut optimal (misal $x_1=5, x_2=10$) \textbf{tetap optimal}. Namun, nilai akhir objek $Z$ tentu akan berubah mengikuti rumus $Z$.
    \item Laporan \textit{solver} biasanya menyajikan nilai ini sebagai \textit{Allowable Increase} dan \textit{Allowable Decrease}. Rentangnya adalah: $[c_j - Decrease, \ c_j + Increase]$.
\end{itemize}

\section{Perubahan Nilai Ruas Kanan ($b_i$)}
Perubahan pada $b_i$ secara geometris akan menggeser kendala sejajar dengan posisi aslinya, sehingga memperbesar atau memperkecil ruang feasibel.
\begin{itemize}
    \item \textbf{Range of Feasibility (Rentang Kelayakan)}: Batas perubahan pada kapasitas sumber daya ($b_i$) di mana \textit{shadow price} dari sumber daya tersebut tetap valid dan konstan.
    \item Dalam rentang ini, titik optimal pasti akan bergeser (nilai variabel berubah), namun kombinasi status sumber daya (misalnya mana kendala yang 'binding/ketat' dan mana yang memiliki sisa) tetap sama.
    \item Jika penambahan kapasitas melewati batas \textit{Allowable Increase}, maka kendala tersebut mungkin tidak lagi menjadi \textit{bottleneck}, dan \textit{shadow price} akan turun.
\end{itemize}

\section{Reduced Cost}
Reduced Cost terkait erat dengan \textbf{variabel non-basis} (variabel yang bernilai 0 pada kondisi optimal). Ia menunjukkan penalti per unit terhadap fungsi objektif jika kita memaksa variabel non-basis tersebut masuk ke dalam solusi (bernilai $>0$). Secara ekonomi, ia menunjukkan margin yang perlu diatasi agar sebuah alternatif yang saat ini "terlalu mahal" (atau "kurang menguntungkan") menjadi opsi yang visibel.

\section{Aturan 100\% (The 100\% Rule)}
Bagaimana jika beberapa parameter berubah secara bersamaan? Aturan 100\% memberikan indikasi apakah solusi masih terjamin optimal.
Caranya: Hitung persentase perubahan masing-masing koefisien terhadap \textit{allowable change}-nya. Jika jumlah persentase perubahannya $\le 100\%$, maka solusi basis terjamin tidak berubah (meskipun nilai numeriknya dapat bergeser). Jika $> 100\%$, belum tentu berubah, namun kita harus menyelesaikan ulang model untuk memastikannya.

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
\lhead{Lembar Kerja Mahasiswa (LKM) - Minggu 4}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{LEMBAR KERJA MAHASISWA (LKM)}\\
    \large\textbf{Minggu 4: Membuat dan Menginterpretasikan Laporan Sensitivitas}
\end{center}

\vspace{0.5cm}
\textbf{Nama Praktikan :} \rule{6cm}{0.4pt} \\
\textbf{NPM :} \rule{6.8cm}{0.4pt}

\section*{Tujuan Praktikum}
1. Mahasiswa mampu menghasilkan laporan rentang sensitivitas menggunakan Julia/JuMP atau \textit{solver} di Python. \\
2. Mahasiswa mampu menerjemahkan keluaran komputasional ke dalam strategi operasi keteknikan.

\section*{Kasus: Alokasi Produksi Baterai (Dimensi C3 - C4)}
Sebuah pabrik memproduksi baterai Li-Ion untuk Drone ($x_1$) dan Baterai untuk Mobil Listrik EV ($x_2$). Profit untuk setiap batch baterai Drone adalah Rp 300 Juta, dan EV adalah Rp 500 Juta. Pabrik dibatasi oleh kapasitas mesin Assembly (A) dan mesin Testing (T).
\begin{itemize}
    \item Perakitan: 2 jam/$x_1$ dan 4 jam/$x_2$. Maksimal tersedia 160 jam. ($2x_1 + 4x_2 \le 160$)
    \item Pengujian: 3 jam/$x_1$ dan 2 jam/$x_2$. Maksimal tersedia 120 jam. ($3x_1 + 2x_2 \le 120$)
\end{itemize}

\section*{Aktivitas 1: Ekstraksi Rentang Optimasi (C3)}
Pada \textit{solver} LP modern (seperti Linprog pada SciPy atau opsi tertentu di HiGHS), kita dapat mengekstrak sensitivitas rentang objektif dan RHS.
\textbf{Implementasi Python (SciPy):}
\begin{lstlisting}[language=Python]
from scipy.optimize import linprog

# C = [-300, -500] untuk maksimasi
c = [-300, -500]
A = [[2, 4], [3, 2]]
b = [160, 120]

# Metode highs mendukung analisis margin
res = linprog(c, A_ub=A, b_ub=b, method='highs')

print("Solusi:", res.x)
print("Profit Maksimal:", -res.fun)
print("Shadow Prices (Inequalities):", -res.ineqlin.marginals)
\end{lstlisting}
\textit{Catatan: Di kelas, kita akan menggunakan \textit{package} tambahan atau perhitungan manual sederhana dari matriks basis optimal jika menggunakan bahasa Julia standar, namun konsepnya tetap sama.}

\section*{Aktivitas 2: Analisis Skenario Berdasarkan Rentang (C4)}
Anggap laporan analisis yang dihasilkan sistem menyatakan untuk Baterai Drone ($c_1 = 300$):
\begin{itemize}
    \item \textbf{Allowable Increase}: 50
    \item \textbf{Allowable Decrease}: 112.5
\end{itemize}
Untuk Mesin Assembly ($b_1 = 160$):
\begin{itemize}
    \item \textbf{Shadow Price}: 112.5
    \item \textbf{Allowable Increase}: 80
    \item \textbf{Allowable Decrease}: 80
\end{itemize}

\textbf{Pertanyaan:}
1. Jika terjadi persaingan pasar yang memaksa kita menurunkan profit Baterai Drone menjadi Rp 250 Juta/batch, apakah strategi alokasi produksi kita saat ini harus diubah? Jelaskan!
2. Jika ada tawaran sewa mesin perakitan tambahan seharga Rp 100 Juta/jam dengan kuota 50 jam, apakah tawaran ini menguntungkan untuk diambil? Evaluasi menggunakan prinsip Range of Feasibility dan Shadow Price.

\vspace{1cm}
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
\lhead{Problem Set - Minggu 4}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \Large\color{UNIBBlue}\textbf{PROBLEM SET}\\
    \large\textbf{Minggu 4: Analisis Sensitivitas}
\end{center}

\vspace{0.5cm}

\textbf{Instruksi:} Kerjakan soal-soal di bawah ini dengan menyertakan langkah-langkah analitis dan dasar teori analisis sensitivitas.

\section*{Soal 1: Terminologi (C1 - C2)}
1. (C1) Apakah perbedaan antara Range of Optimality dan Range of Feasibility? \\
2. (C2) Mengapa analisis sensitivitas sangat penting bagi seorang insinyur perencana (\textit{planning engineer}) sistem telekomunikasi ketika memprediksi kebutuhan infrastruktur jaringan ke depan?

\section*{Soal 2: Evaluasi Skenario (C4)}
Diberikan laporan sensitivitas untuk maksimasi kinerja produksi panel surya Tipe A ($x_1$) dan Tipe B ($x_2$):

\begin{table}[h]
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Produk} & \textbf{Final Value} & \textbf{Obj. Coef. (\$)} & \textbf{Allowable Increase (\$)} & \textbf{Allowable Decrease (\$)} \\ \hline
Tipe A ($x_1$) & 120 & 40 & 10 & 25 \\ \hline
Tipe B ($x_2$) & 80  & 30 & 50 & 6  \\ \hline
\end{tabular}
\end{table}

(a) Tentukan rentang optimalitas untuk koefisien tujuan dari Tipe A ($c_1$) dan Tipe B ($c_2$). \\
(b) Jika karena kelangkaan material tipe B, harga jual (objektif) tipe B turun menjadi \$25, apakah konfigurasi optimal produksi berubah?

\section*{Soal 3: The 100\% Rule (C5)}
Dengan menggunakan data dari Soal 2, asumsikan terjadi perubahan secara bersamaan: profit Tipe A meningkat sebesar \$5 (karena subsidi hijau pemerintah), dan profit Tipe B menurun sebesar \$4 (karena kalah saing). 

Gunakan \textit{The 100\% Rule} untuk menentukan apakah solusi optimal saat ini terjamin tetap optimal, atau ada kemungkinan berubah. Tunjukkan perhitungan persentasenya secara lengkap.

\section*{Soal 4: Sintesis Pengambilan Keputusan (C6)}
Sebuah analisis alokasi spektrum frekuensi menghasilkan \textit{shadow price} sebesar 5.0 (Mb/s) per MHz untuk alokasi band 5GHz, dengan \textit{allowable increase} sebesar 20 MHz.
Saat ini regulator menawarkan izin blok tambahan 15 MHz di band 5GHz dengan biaya sewa tahunan yang setara jika dikonversi menjadi denda performansi senilai kehilangan throughput total 60 Mb/s.

(C6) Sebagai Chief Technology Officer (CTO), apakah Anda akan membeli blok 15 MHz tersebut? Sertakan argumen matematis berdasarkan perbandingan \textit{shadow price} dan biaya margin, serta nyatakan validitasnya melalui \textit{Range of Feasibility}.

\end{document}
"""

create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/slides/week04_slides.tex', slides_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/notes/week04_notes.tex', notes_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/worksheets/week04_worksheet.tex', worksheet_content)
create_file('/Users/novaliodaratha/Documents/pengajaran/Optimisasi/2026/problem_sets/week04_problem_set.tex', problem_set_content)

print("Week 4 files created successfully.")
