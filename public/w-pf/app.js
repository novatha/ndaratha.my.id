/**
 * Interaktivitas Portal Pelatihan Analisis Aliran Daya Menggunakan Python
 * Mencakup:
 *   1. Pengelolaan skema warna dua-status (sinkronisasi <meta name="color-scheme"> & tab storage)
 *   2. Simulator Topologi Interaktif IEEE 14-Bus & Telemetri Newton-Raphson
 *   3. Eksplorasi & Penyaringan Tabel Kontinjensi N-1 (20 Cabang)
 *   4. Modal Lightbox <dialog> dengan fallback closedBy berbasis koordinat pointer
 *   5. Penyalin Kode & Pemeriksa Kuis Diagnostik Formatif (Bloom C2-C6)
 */

"use strict";

/* ==========================================================================
   Data Terverifikasi Sistem Uji IEEE 14-Bus (newton_raphson.py & n1_ranking.csv)
   ========================================================================== */

const IEEE14_BUS_COORDS = [
  { id: 1, x: 95, y: 375, role: "Slack" },
  { id: 2, x: 220, y: 405, role: "PV" },
  { id: 3, x: 470, y: 405, role: "PV" },
  { id: 4, x: 430, y: 295, role: "PQ" },
  { id: 5, x: 260, y: 295, role: "PQ" },
  { id: 6, x: 175, y: 185, role: "PV" },
  { id: 7, x: 525, y: 225, role: "PQ" },
  { id: 8, x: 675, y: 225, role: "PV" },
  { id: 9, x: 470, y: 155, role: "PQ" },
  { id: 10, x: 375, y: 135, role: "PQ" },
  { id: 11, x: 270, y: 155, role: "PQ" },
  { id: 12, x: 115, y: 90, role: "PQ" },
  { id: 13, x: 245, y: 75, role: "PQ" },
  { id: 14, x: 430, y: 70, role: "PQ" }
];

const IEEE14_BRANCHES = [
  { name: "L1-2", f: 1, t: 2, isTrafo: false },
  { name: "L1-5", f: 1, t: 5, isTrafo: false },
  { name: "L2-3", f: 2, t: 3, isTrafo: false },
  { name: "L2-4", f: 2, t: 4, isTrafo: false },
  { name: "L2-5", f: 2, t: 5, isTrafo: false },
  { name: "L3-4", f: 3, t: 4, isTrafo: false },
  { name: "L4-5", f: 4, t: 5, isTrafo: false },
  { name: "T4-7", f: 4, t: 7, isTrafo: true },
  { name: "T4-9", f: 4, t: 9, isTrafo: true },
  { name: "T5-6", f: 5, t: 6, isTrafo: true },
  { name: "L6-11", f: 6, t: 11, isTrafo: false },
  { name: "L6-12", f: 6, t: 12, isTrafo: false },
  { name: "L6-13", f: 6, t: 13, isTrafo: false },
  { name: "L7-8", f: 7, t: 8, isTrafo: false },
  { name: "L7-9", f: 7, t: 9, isTrafo: false },
  { name: "L9-10", f: 9, t: 10, isTrafo: false },
  { name: "L9-14", f: 9, t: 14, isTrafo: false },
  { name: "L10-11", f: 10, t: 11, isTrafo: false },
  { name: "L12-13", f: 12, t: 13, isTrafo: false },
  { name: "L13-14", f: 13, t: 14, isTrafo: false }
];

const SCENARIO_DATA = {
  base: {
    label: "Kondisi Dasar (Base Case)",
    summary: "Kondisi Dasar: Seluruh 20 cabang beroperasi aman (maks. L1-2 = 79,10%), 0 bus beralih PV→PQ, konvergen dalam 4 iterasi NR.",
    outage: null,
    iter: 4,
    ploss: 13.3933,
    qloss: 30.1224,
    switched: [],
    vm: [1.06, 1.045, 1.01, 1.0177, 1.0195, 1.07, 1.0615, 1.09, 1.0559, 1.051, 1.0569, 1.0552, 1.0504, 1.0355],
    va: [0.0, -4.98, -12.73, -10.31, -8.77, -14.22, -13.36, -13.36, -14.94, -15.1, -14.79, -15.08, -15.16, -16.03],
    loads: {
      "L1-2": 79.1, "L1-5": 75.61, "L2-3": 73.32, "L2-4": 70.19, "L2-5": 69.22,
      "L3-4": 60.37, "L4-5": 70.32, "T4-7": 60.59, "T4-9": 53.91, "T5-6": 70.49,
      "L6-11": 32.68, "L6-12": 32.71, "L6-13": 63.86, "L7-8": 58.74, "L7-9": 63.69,
      "L9-10": 26.87, "L9-14": 40.38, "L10-11": 16.55, "L12-13": 7.13, "L13-14": 23.63
    }
  },
  "L2-3": {
    label: "Kontinjensi Darurat #1: Outage Saluran L2-3",
    summary: "Outage L2-3: Beban 94,2 MW di Bus 3 tersedot lewat L3-4 (252,97% MVA)! Bus 3 jenuh Qmax dan beralih PV→PQ (|V3| turun ke 0,9538 p.u.).",
    outage: "L2-3",
    iter: 7,
    ploss: 24.3316,
    qloss: 64.9384,
    switched: [3],
    vm: [1.06, 1.045, 0.9538, 0.9984, 1.0056, 1.07, 1.0524, 1.09, 1.0465, 1.0431, 1.0528, 1.0545, 1.0489, 1.0294],
    va: [0.0, -4.7, -24.3, -13.68, -11.12, -17.04, -16.59, -16.59, -18.09, -18.19, -17.74, -17.93, -18.02, -19.07],
    loads: {
      "L1-2": 74.73, "L1-5": 95.44, "L2-4": 117.09, "L2-5": 115.33,
      "L3-4": 252.97, "L4-5": 114.61, "T4-7": 61.85, "T4-9": 51.14, "T5-6": 73.39,
      "L6-11": 41.88, "L6-12": 33.98, "L6-13": 67.88, "L7-8": 77.62, "L7-9": 59.54,
      "L9-10": 18.0, "L9-14": 34.56, "L10-11": 25.46, "L12-13": 8.36, "L13-14": 29.42
    }
  },
  "L1-2": {
    label: "Kontinjensi Darurat #2: Outage Saluran L1-2",
    summary: "Outage L1-2: Seluruh ekspor Slack Bus 1 bertumpu pada L1-5 (246,59%)! Keempat PV Bus [2, 3, 6, 8] menyentuh batas reaktif (PV→PQ).",
    outage: "L1-2",
    iter: 44,
    ploss: 35.4204,
    qloss: 130.3894,
    switched: [2, 3, 6, 8],
    vm: [1.06, 0.9942, 0.9725, 0.9702, 0.9666, 1.0383, 1.0231, 1.0615, 1.0163, 1.0126, 1.0217, 1.0233, 1.0177, 0.9993],
    va: [0.0, -33.83, -38.64, -33.58, -30.17, -35.96, -36.08, -36.08, -37.35, -37.36, -36.79, -36.8, -36.92, -38.06],
    loads: {
      "L1-5": 246.59, "L2-3": 41.53, "L2-4": 17.91, "L2-5": 62.14,
      "L3-4": 113.62, "L4-5": 141.24, "T4-7": 52.67, "T4-9": 41.41, "T5-6": 66.53,
      "L6-11": 41.91, "L6-12": 31.05, "L6-13": 63.37, "L7-8": 77.04, "L7-9": 48.76,
      "L9-10": 16.29, "L9-14": 28.73, "L10-11": 26.68, "L12-13": 8.19, "L13-14": 28.68
    }
  },
  "T5-6": {
    label: "Kontinjensi Trafo Antarmuka: Outage T5-6",
    summary: "Outage T5-6: Suplai ke subsistem 13,8 kV terputus dari Bus 5; 6 cabang (T4-7, T4-9, L4-5, L7-9, L9-10, L9-14) mengalami overload serentak!",
    outage: "T5-6",
    iter: 7,
    ploss: 16.4448,
    qloss: 45.5087,
    switched: [6],
    vm: [1.06, 1.045, 1.01, 1.014, 1.0243, 1.0073, 1.0534, 1.09, 1.0443, 1.0305, 1.0159, 0.9937, 0.9945, 1.0029],
    va: [0.0, -5.14, -13.19, -10.98, -8.61, -27.01, -17.28, -17.28, -20.57, -21.98, -24.56, -27.42, -27.0, -24.37],
    loads: {
      "L1-2": 81.47, "L1-5": 73.96, "L2-3": 76.1, "L2-4": 76.96, "L2-5": 62.77,
      "L3-4": 54.59, "L4-5": 111.91, "T4-7": 116.99, "T4-9": 109.64,
      "L6-11": 81.44, "L6-12": 21.71, "L6-13": 29.24, "L7-8": 75.52, "L7-9": 129.52,
      "L9-10": 133.85, "L9-14": 111.06, "L10-11": 95.36, "L12-13": 9.59, "L13-14": 48.31
    }
  },
  qlim8: {
    label: "Eksperimen Modul 2: Batas AVR Bus 8 (Qmax = 10 MVAr)",
    summary: "Batas Reaktif Bus 8 = 10 MVAr: Kondensor sinkron Bus 8 melampaui Qmax (butuh 17,6 MVAr), beralih PV→PQ sehingga |V8| turun dari 1,0900 ke 1,0675 p.u.",
    outage: null,
    iter: 7,
    ploss: 13.4343,
    qloss: 29.8151,
    switched: [8],
    vm: [1.06, 1.045, 1.01, 1.0153, 1.018, 1.07, 1.051, 1.0675, 1.0486, 1.0449, 1.0538, 1.0546, 1.0493, 1.0309],
    va: [0.0, -4.99, -12.74, -10.28, -8.76, -14.28, -13.32, -13.32, -14.91, -15.08, -14.81, -15.14, -15.2, -16.04],
    loads: {
      "L1-2": 79.15, "L1-5": 75.6, "L2-3": 73.42, "L2-4": 70.07, "L2-5": 69.38,
      "L3-4": 60.94, "L4-5": 69.24, "T4-7": 57.25, "T4-9": 53.21, "T5-6": 71.02,
      "L6-11": 36.7, "L6-12": 33.27, "L6-13": 65.44, "L7-8": 33.33, "L7-9": 61.85,
      "L9-10": 22.66, "L9-14": 38.24, "L10-11": 20.55, "L12-13": 7.75, "L13-14": 25.91
    }
  },
  "heavy_L6-13": {
    label: "Beban Puncak 1,6× + Outage L6-13 (Kasus Undervoltage)",
    summary: "Beban Puncak 1,6× + Outage L6-13: Tanpa pasokan langsung L6-13, Bus 13 jatuh ke 0,9362 p.u. dan Bus 14 ke 0,9457 p.u. (pelanggaran undervoltage < 0,95 p.u.).",
    outage: "L6-13",
    iter: 4,
    ploss: 42.8167,
    qloss: 140.8989,
    switched: [],
    vm: [1.06, 1.045, 1.01, 0.994, 0.9982, 1.07, 1.0328, 1.09, 1.0094, 1.0074, 1.0321, 1.0012, 0.9362, 0.9457],
    va: [0.0, -9.29, -22.39, -17.97, -15.32, -24.13, -23.43, -23.43, -26.32, -26.4, -25.45, -27.8, -29.41, -29.52],
    loads: {
      "L1-2": 145.96, "L1-5": 129.51, "L2-3": 121.65, "L2-4": 114.1, "L2-5": 110.98,
      "L3-4": 107.67, "L4-5": 116.35, "T4-7": 97.83, "T4-9": 91.69, "T5-6": 107.57,
      "L6-11": 87.34, "L6-12": 144.14, "L7-8": 117.99, "L7-9": 118.0,
      "L9-10": 10.85, "L9-14": 113.31, "L10-11": 59.14, "L12-13": 94.64, "L13-14": 9.36
    }
  }
};

/* Seluruh 20 baris terverifikasi dari hasil/n1_ranking.csv */
const N1_RANKING_ROWS = [
  { ctg: "L2-3", status: "OK", maxLoad: 301.48, worstBr: "L3-4", nOver: 4, overList: "L2-4, L2-5, L3-4, L4-5", vmin: 1.0100, vminBus: 3, pi: 95.50 },
  { ctg: "L1-2", status: "OK", maxLoad: 251.16, worstBr: "L1-5", nOver: 3, overList: "L1-5, L3-4, L4-5", vmin: 0.9935, vminBus: 5, pi: 60.05 },
  { ctg: "T5-6", status: "OK", maxLoad: 143.85, worstBr: "L7-9", nOver: 7, overList: "L4-5, L7-9, L9-10, L9-14, L10-11, T4-7, T4-9", vmin: 1.0100, vminBus: 3, pi: 27.04 },
  { ctg: "L1-5", status: "OK", maxLoad: 129.99, worstBr: "L2-5", nOver: 3, overList: "L1-2, L2-4, L2-5", vmin: 1.0064, vminBus: 5, pi: 12.84 },
  { ctg: "L2-4", status: "OK", maxLoad: 111.57, worstBr: "L2-5", nOver: 2, overList: "L2-5, L4-5", vmin: 1.0071, vminBus: 4, pi: 10.94 },
  { ctg: "T4-7", status: "OK", maxLoad: 124.91, worstBr: "T4-9", nOver: 1, overList: "T4-9", vmin: 1.0100, vminBus: 3, pi: 10.87 },
  { ctg: "L7-9", status: "OK", maxLoad: 121.25, worstBr: "T4-9", nOver: 1, overList: "T4-9", vmin: 1.0100, vminBus: 3, pi: 10.11 },
  { ctg: "L4-5", status: "OK", maxLoad: 111.58, worstBr: "L2-4", nOver: 1, overList: "L2-4", vmin: 1.0100, vminBus: 3, pi: 9.69 },
  { ctg: "T4-9", status: "OK", maxLoad: 97.99, worstBr: "L7-9", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 9.38 },
  { ctg: "L9-14", status: "OK", maxLoad: 93.13, worstBr: "L6-13", nOver: 0, overList: "—", vmin: 0.9969, vminBus: 14, pi: 9.25 },
  { ctg: "L2-5", status: "OK", maxLoad: 93.20, worstBr: "L2-4", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.63 },
  { ctg: "L10-11", status: "OK", maxLoad: 76.19, worstBr: "L7-9", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.59 },
  { ctg: "L13-14", status: "OK", maxLoad: 77.38, worstBr: "L7-9", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.53 },
  { ctg: "L6-13", status: "OK", maxLoad: 82.57, worstBr: "L6-12", nOver: 0, overList: "—", vmin: 0.9980, vminBus: 13, pi: 8.52 },
  { ctg: "L12-13", status: "OK", maxLoad: 73.59, worstBr: "L1-2", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.50 },
  { ctg: "L3-4", status: "OK", maxLoad: 98.39, worstBr: "L2-3", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.47 },
  { ctg: "L6-12", status: "OK", maxLoad: 88.10, worstBr: "L6-13", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.44 },
  { ctg: "L6-11", status: "OK", maxLoad: 80.70, worstBr: "L7-9", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.30 },
  { ctg: "L9-10", status: "OK", maxLoad: 75.47, worstBr: "T5-6", nOver: 0, overList: "—", vmin: 1.0100, vminBus: 3, pi: 8.29 },
  { ctg: "L7-8", status: "ISLANDING", maxLoad: 73.69, worstBr: "L1-2", nOver: 0, overList: "Bus 8 Terisolasi", vmin: 1.0100, vminBus: 3, pi: 4.98 }
];

const QUIZ_SOLUTIONS = {
  q1: {
    ans: "a",
    explain: "Impedansi seri dimodelkan di sisi sekunder setelah transformator ideal a:1, sehingga arus dan tegangan primer diskalakan oleh |a|²."
  },
  q2: {
    ans: "b",
    explain: "Dimensi awal J adalah N_pv + 2·N_pq = 4 + 18 = 22. Saat 2 PV beralih ke PQ, N_pv = 2 dan N_pq = 11 sehingga J menjadi 2 + 22 = 24."
  },
  q3: {
    ans: "a",
    explain: "Asumsi dekopling P-θ dan Q-|V| mensyaratkan G ≪ B (R ≪ X). Saat R/X besar, pengabaian J2 dan J3 merusak arah langkah koreksi."
  },
  q4: {
    ans: "a",
    explain: "Bila Q dipatok di Qmax namun |V_i| melampaui V_set, berarti injeksi reaktif sudah berlebih dan AVR dapat menurunkannya kembali ke mode PV."
  },
  q5: {
    ans: "b",
    explain: "Beban aktif besar 94,2 MW di Bus 3 memerlukan penguatan koridor transmisi (sirkuit ganda) atau skema redispatch/SPS saat L2-3 lepas."
  }
};

function fmtIdNum(val, digits = 2) {
  return Number(val).toFixed(digits).replace(".", ",");
}

/* ==========================================================================
   1. Toggle Tema Dua-Status (Sesuai pedoman dark-mode-toggle)
   ========================================================================== */

function initThemeToggle() {
  const meta = document.querySelector('meta[name="color-scheme"]');
  const toggleBtn = document.getElementById("theme-toggle");
  if (!meta || !toggleBtn) return;

  const systemDarkMql = window.matchMedia("(prefers-color-scheme: dark)");

  const getStoredScheme = () => {
    try {
      const val = localStorage.getItem("color-scheme");
      return val === "light" || val === "dark" ? val : null;
    } catch {
      return null;
    }
  };

  toggleBtn.addEventListener("click", () => {
    const stored = getStoredScheme();
    const systemScheme = systemDarkMql.matches ? "dark" : "light";
    const currentRendered = stored ?? systemScheme;
    const target = currentRendered === "dark" ? "light" : "dark";

    if (target === systemScheme) {
      // Kembali ke preferensi sistem: hapus entri agar tidak memaku skema secara tersembunyi
      try {
        localStorage.removeItem("color-scheme");
      } catch {}
      meta.content = "light dark";
    } else {
      try {
        localStorage.setItem("color-scheme", target);
      } catch {}
      meta.content = target;
    }
  });

  // Sinkronisasi apabila pengguna mengubah skema dari tab lain
  window.addEventListener("storage", (event) => {
    if (event.key === "color-scheme") {
      const next = event.newValue;
      meta.content = next === "light" || next === "dark" ? next : "light dark";
    }
  });
}

/* ==========================================================================
   2. Simulator Topologi Interaktif IEEE 14-Bus
   ========================================================================== */

function bindSvgActivation(el, callback) {
  el.addEventListener("click", callback);
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      callback();
    }
    if (e.key === " ") {
      e.preventDefault();
    }
  });
  el.addEventListener("keyup", (e) => {
    if (e.key === " ") {
      callback();
    }
  });
}

function renderScenario(scenarioKey) {
  const data = SCENARIO_DATA[scenarioKey];
  if (!data) return;

  const svg = document.getElementById("ieee14-svg");
  const inspector = document.getElementById("sim-inspector");
  const kpiIter = document.getElementById("kpi-iter");
  const kpiLoss = document.getElementById("kpi-loss");
  const kpiVmin = document.getElementById("kpi-vmin");
  const kpiMaxBr = document.getElementById("kpi-maxbr");
  const vbarsContainer = document.getElementById("voltage-bars");

  if (!svg || !inspector || !vbarsContainer) return;

  inspector.textContent = data.summary;
  kpiIter.textContent = `${data.iter} iter · KONVERGEN`;
  kpiLoss.textContent = `${fmtIdNum(data.ploss, 2)} MW / ${fmtIdNum(data.qloss, 2)} MVAr`;

  let minV = Infinity;
  let minBus = 1;
  data.vm.forEach((v, idx) => {
    if (v < minV) {
      minV = v;
      minBus = idx + 1;
    }
  });
  kpiVmin.textContent = `${fmtIdNum(minV, 4)} p.u. (Bus ${minBus})`;

  let maxBrName = "—";
  let maxBrVal = 0;
  Object.entries(data.loads).forEach(([brName, pct]) => {
    if (pct > maxBrVal) {
      maxBrVal = pct;
      maxBrName = brName;
    }
  });
  kpiMaxBr.textContent = `${maxBrName} (${fmtIdNum(maxBrVal, 2)}%)`;

  // Bersihkan isi SVG selain elemen <title>
  const titleEl = svg.querySelector("title");
  svg.replaceChildren();
  if (titleEl) svg.appendChild(titleEl);

  const svgNS = "http://www.w3.org/2000/svg";
  const busLookup = Object.fromEntries(IEEE14_BUS_COORDS.map((b) => [b.id, b]));

  // Gambar 20 Cabang
  IEEE14_BRANCHES.forEach((br) => {
    const bFrom = busLookup[br.f];
    const bTo = busLookup[br.t];
    const isOut = data.outage === br.name;
    const loadPct = isOut ? 0 : (data.loads[br.name] ?? 0);
    const isOver = !isOut && loadPct > 100.0;

    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("role", "button");
    g.setAttribute("tabindex", "0");
    const statusText = isOut ? "LEPAS (Outage)" : isOver ? `OVERLOAD ${fmtIdNum(loadPct, 2)}%` : `Aman ${fmtIdNum(loadPct, 2)}%`;
    g.setAttribute("aria-label", `Cabang ${br.name}: ${statusText}`);

    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(bFrom.x));
    line.setAttribute("y1", String(bFrom.y));
    line.setAttribute("x2", String(bTo.x));
    line.setAttribute("y2", String(bTo.y));
    line.setAttribute(
      "class",
      `svg-branch${isOut ? " is-outage" : ""}${isOver ? " is-overload" : ""}`
    );
    g.appendChild(line);

    const midX = (bFrom.x + bTo.x) / 2;
    const midY = (bFrom.y + bTo.y) / 2;

    if (isOut || isOver) {
      const tag = document.createElementNS(svgNS, "text");
      tag.setAttribute("x", String(midX));
      tag.setAttribute("y", String(midY - 6));
      tag.setAttribute("class", "svg-bus-label");
      tag.textContent = isOut ? `✕ ${br.name}` : `⚠️ ${br.name} (${Math.round(loadPct)}%)`;
      g.appendChild(tag);
    }

    bindSvgActivation(g, () => {
      inspector.textContent = isOut
        ? `Inspeksi Cabang ${br.name} (Bus ${br.f}–Bus ${br.t}): Status LEPAS (Outage Kontinjensi Aktif, arus = 0 p.u.).`
        : `Inspeksi Cabang ${br.name} (${br.isTrafo ? "Transformator" : "Saluran"} Bus ${br.f}–Bus ${br.t}): Pembebanan = ${fmtIdNum(loadPct, 2)}% (${isOver ? "⚠️ MELAMPAUI BATAS TERMAL 100%" : "✅ Dalam batas aman"}).`;
    });

    svg.appendChild(g);
  });

  // Gambar 14 Bus
  IEEE14_BUS_COORDS.forEach((bus) => {
    const vm = data.vm[bus.id - 1];
    const va = data.va[bus.id - 1];
    const isSwitched = data.switched.includes(bus.id);
    const isUnder = vm < 0.95;
    const isSlack = bus.role === "Slack";

    const g = document.createElementNS(svgNS, "g");
    g.setAttribute(
      "class",
      `svg-bus-node${isSlack ? " is-slack" : ""}${isSwitched ? " is-switched" : ""}${isUnder ? " is-undervolt" : ""}`
    );
    g.setAttribute("role", "button");
    g.setAttribute("tabindex", "0");
    g.setAttribute(
      "aria-label",
      `Bus ${bus.id} (${bus.role}): tegangan ${fmtIdNum(vm, 4)} per-unit sudut ${fmtIdNum(va, 2)} derajat`
    );

    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("cx", String(bus.x));
    circle.setAttribute("cy", String(bus.y));
    circle.setAttribute("r", "18");
    circle.setAttribute("class", "svg-bus-circle");
    g.appendChild(circle);

    const lbl = document.createElementNS(svgNS, "text");
    lbl.setAttribute("x", String(bus.x));
    lbl.setAttribute("y", String(bus.y));
    lbl.setAttribute("class", "svg-bus-label");
    lbl.textContent = `B${bus.id}`;
    g.appendChild(lbl);

    const vtxt = document.createElementNS(svgNS, "text");
    vtxt.setAttribute("x", String(bus.x));
    vtxt.setAttribute("y", String(bus.y + 29));
    vtxt.setAttribute("class", "svg-bus-volt");
    vtxt.textContent = `${fmtIdNum(vm, 3)} pu`;
    g.appendChild(vtxt);

    bindSvgActivation(g, () => {
      const extra = isSwitched
        ? " · 🔄 Beralih PV→PQ (Batas Reaktif Aktif)"
        : isUnder
          ? " · ⚠️ UNDERVOLTAGE (< 0,95 p.u.)"
          : "";
      inspector.textContent = `Inspeksi Bus ${bus.id} [Tipe awal: ${bus.role}]: |V| = ${fmtIdNum(vm, 4)} p.u., sudut θ = ${fmtIdNum(va, 2)}°${extra}.`;
    });

    svg.appendChild(g);
  });

  // Gambar Batang Profil Tegangan 14 Bus menggunakan CSS Custom Property --v-pct
  vbarsContainer.replaceChildren();
  data.vm.forEach((vm, i) => {
    const busId = i + 1;
    const isLow = vm < 0.95;
    const isSwitched = data.switched.includes(busId);
    const pct = Math.max(4, Math.min(100, ((vm - 0.90) / 0.20) * 100));

    const row = document.createElement("div");
    row.className = `vbar-row${isLow ? " is-low" : ""}${isSwitched ? " is-switched" : ""}`;

    const nameSpan = document.createElement("span");
    nameSpan.textContent = `Bus ${busId}`;

    const track = document.createElement("div");
    track.className = "vbar-track";
    const fill = document.createElement("div");
    fill.className = "vbar-fill";
    fill.style.setProperty("--v-pct", `${pct.toFixed(1)}%`);
    track.appendChild(fill);

    const valSpan = document.createElement("span");
    valSpan.textContent = `${fmtIdNum(vm, 4)}${isLow ? " ⚠️" : isSwitched ? " 🔄" : ""}`;

    row.append(nameSpan, track, valSpan);
    vbarsContainer.appendChild(row);
  });
}

function initSimulator() {
  const buttons = document.querySelectorAll(".scenario-btn");
  if (!buttons.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.scenario;
      buttons.forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
      renderScenario(key);
    });
  });

  renderScenario("base");
}

/* ==========================================================================
   3. Tabel Responsif & Penyaringan Kontinjensi N-1
   ========================================================================== */

function initContingencyTable() {
  const tbody = document.getElementById("n1-tbody");
  const searchInput = document.getElementById("n1-search");
  const statusSelect = document.getElementById("n1-status-filter");
  const sortSelect = document.getElementById("n1-sort");
  const statusAnnouncement = document.getElementById("n1-filter-status");

  if (!tbody || !searchInput || !statusSelect || !sortSelect || !statusAnnouncement) return;

  const renderRows = () => {
    const q = searchInput.value.trim().toLowerCase();
    const statusFilter = statusSelect.value;
    const sortMode = sortSelect.value;

    const filtered = N1_RANKING_ROWS.filter((row) => {
      const matchesQuery =
        !q ||
        row.ctg.toLowerCase().includes(q) ||
        row.worstBr.toLowerCase().includes(q) ||
        row.overList.toLowerCase().includes(q);

      if (!matchesQuery) return false;
      if (statusFilter === "overload") return row.status === "OK" && row.nOver > 0;
      if (statusFilter === "safe") return row.status === "OK" && row.nOver === 0;
      if (statusFilter === "islanding") return row.status === "ISLANDING";
      return true;
    });

    filtered.sort((a, b) => {
      if (sortMode === "load_desc") return b.maxLoad - a.maxLoad;
      if (sortMode === "vmin_asc") return a.vmin - b.vmin;
      return b.pi - a.pi;
    });

    tbody.replaceChildren();

    filtered.forEach((row) => {
      const tr = document.createElement("tr");

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = row.ctg;

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      if (row.status === "ISLANDING") {
        badge.className = "status-badge badge-warn";
        badge.innerHTML = '<span aria-hidden="true">🔌</span> ISLANDING';
      } else if (row.nOver > 0) {
        badge.className = "status-badge badge-crit";
        badge.innerHTML = `<span aria-hidden="true">⚠️</span> OVERLOAD (${row.nOver})`;
      } else {
        badge.className = "status-badge badge-ok";
        badge.innerHTML = '<span aria-hidden="true">✅</span> AMAN';
      }
      tdStatus.appendChild(badge);

      const tdMaxLoad = document.createElement("td");
      tdMaxLoad.textContent = `${fmtIdNum(row.maxLoad, 2)}%`;

      const tdWorst = document.createElement("td");
      const codeEl = document.createElement("code");
      codeEl.textContent = row.worstBr;
      tdWorst.appendChild(codeEl);

      const tdCount = document.createElement("td");
      tdCount.textContent = String(row.nOver);

      const tdVmin = document.createElement("td");
      tdVmin.textContent = `${fmtIdNum(row.vmin, 4)} p.u. (Bus ${row.vminBus})`;

      const tdPi = document.createElement("td");
      tdPi.textContent = fmtIdNum(row.pi, 2);

      const tdList = document.createElement("td");
      tdList.textContent = row.overList;

      tr.append(th, tdStatus, tdMaxLoad, tdWorst, tdCount, tdVmin, tdPi, tdList);
      tbody.appendChild(tr);
    });

    statusAnnouncement.textContent = `Menampilkan ${filtered.length} dari ${N1_RANKING_ROWS.length} skenario kontinjensi N-1.`;
  };

  searchInput.addEventListener("input", renderRows);
  statusSelect.addEventListener("change", renderRows);
  sortSelect.addEventListener("change", renderRows);

  renderRows();
}

/* ==========================================================================
   4. Modal Lightbox <dialog> & Fallback closedBy (Sesuai pedoman dialog)
   ========================================================================== */

function initGalleryLightbox() {
  const dialog = document.getElementById("figure-lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbTitle = document.getElementById("lightbox-title");
  const lbDesc = document.getElementById("lightbox-desc");

  if (!dialog || !lbImg || !lbTitle || !lbDesc) return;

  document.querySelectorAll(".zoom-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const src = btn.dataset.imgSrc;
      const title = btn.dataset.imgTitle;
      const desc = btn.dataset.imgDesc;
      const w = Number(btn.dataset.imgW);
      const h = Number(btn.dataset.imgH);
      if (src) lbImg.src = src;
      if (w > 0 && h > 0) {
        lbImg.width = w;
        lbImg.height = h;
      }
      if (title) {
        lbTitle.textContent = title;
        lbImg.alt = title;
      }
      lbDesc.textContent = desc ?? "";
      dialog.showModal();
    });
  });

  dialog.addEventListener("close", () => {
    dialog.returnValue = "";
  });

  // Fallback closedby="any" berbasis kotak batas pointer untuk browser tanpa HTMLDialogElement.closedBy (mis. Safari)
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dlg) => {
      const r = dlg.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dlg of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dlg.addEventListener("pointerdown", (e) => {
        pressedOutside = outside(e, dlg);
      });
      dlg.addEventListener("pointerup", (e) => {
        if (pressedOutside && outside(e, dlg)) {
          dlg.close();
        }
      });
    }
  }
}

/* ==========================================================================
   5. Tombol Salin Kode & Pemeriksa Kuis Diagnostik (Bloom C2-C6)
   ========================================================================== */

function initCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetId = btn.dataset.copyTarget;
      const codeEl = targetId ? document.getElementById(targetId) : null;
      if (!codeEl) return;

      const originalText = btn.textContent;
      try {
        await navigator.clipboard.writeText(codeEl.textContent ?? "");
        btn.textContent = "✓ Tersalin";
      } catch {
        btn.textContent = "Salin Manual";
      }
      window.setTimeout(() => {
        btn.textContent = originalText;
      }, 1800);
    });
  });
}

function initQuiz() {
  const form = document.getElementById("quiz-form");
  const scoreEl = document.getElementById("quiz-score");
  if (!form || !scoreEl) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    let correctCount = 0;
    const keys = Object.keys(QUIZ_SOLUTIONS);

    keys.forEach((qKey) => {
      const picked = formData.get(qKey);
      const solution = QUIZ_SOLUTIONS[qKey];
      const fb = document.getElementById(`fb-${qKey}`);
      if (!fb) return;

      fb.hidden = false;
      if (picked === solution.ans) {
        correctCount += 1;
        fb.className = "quiz-feedback is-correct";
        fb.textContent = `✅ Tepat! ${solution.explain}`;
      } else if (picked) {
        fb.className = "quiz-feedback is-wrong";
        fb.textContent = `❌ Belum tepat (Kunci: ${solution.ans.toUpperCase()}). ${solution.explain}`;
      } else {
        fb.className = "quiz-feedback is-wrong";
        fb.textContent = "⚠️ Pilih salah satu opsi terlebih dahulu sebelum memeriksa jawaban.";
      }
    });

    scoreEl.textContent = `Skor Diagnostik Anda: ${correctCount} / ${keys.length} benar.`;
  });

  form.addEventListener("reset", () => {
    Object.keys(QUIZ_SOLUTIONS).forEach((qKey) => {
      const fb = document.getElementById(`fb-${qKey}`);
      if (fb) {
        fb.hidden = true;
        fb.textContent = "";
      }
    });
    scoreEl.textContent = "";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initSimulator();
  initContingencyTable();
  initGalleryLightbox();
  initCopyButtons();
  initQuiz();
});
