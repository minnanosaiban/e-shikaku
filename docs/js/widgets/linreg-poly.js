// 多項式回帰：次数 d と正則化 λ で、過少適合・過剰適合・Ridge の効果を見る
(function () {
  const E = window.ENote;

  const f0 = (x) => Math.sin(2 * Math.PI * x);
  // 固定の乱数（再現できるように）
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  }
  function makeData(n, seed, sd) {
    const r = rng(seed);
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) / n + (r() - 0.5) * 0.04;
      xs.push(x);
      ys.push(f0(x) + sd * gauss(r));
    }
    return { xs, ys };
  }
  const TRAIN = makeData(10, 7, 0.25);
  const TEST = (() => {
    const r = rng(99);
    const xs = [], ys = [];
    for (let i = 0; i < 200; i++) {
      const x = r();
      xs.push(x);
      ys.push(f0(x) + 0.25 * gauss(r));
    }
    return { xs, ys };
  })();

  // 基底 φ_j(x)=(2x-1)^j（j=0..d）。切片 w_0 は正則化しない
  const phi = (x, d) => Array.from({ length: d + 1 }, (_, j) => Math.pow(2 * x - 1, j));

  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const k = M[r][c] / M[c][c];
        for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = M[i][n];
      for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  // Ridge の正規方程式 (ΦᵀΦ+λI')w=Φᵀy を解く
  function fit(xs, ys, d, lam) {
    const P = xs.map((x) => phi(x, d));
    const n = d + 1;
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => P.reduce((s, row) => s + row[i] * row[j], 0) + (i === j && i > 0 ? lam : 0)));
    const b = Array.from({ length: n }, (_, i) => P.reduce((s, row, k) => s + row[i] * ys[k], 0));
    return solve(A, b);
  }
  const predict = (w, x) => phi(x, w.length - 1).reduce((s, v, j) => s + v * w[j], 0);
  const rmse = (w, D) => Math.sqrt(D.xs.reduce((s, x, i) => s + (predict(w, x) - D.ys[i]) ** 2, 0) / D.xs.length);

  if (typeof module !== "undefined") module.exports = { TRAIN, TEST, fit, predict, rmse };
  if (!E) return;

  const X0 = 50, X1 = 610, Y0 = 20, Y1 = 270, YR = 2;
  const sx = (x) => X0 + x * (X1 - X0);
  const sy = (y) => Y1 - ((Math.min(YR, Math.max(-YR, y)) + YR) / (2 * YR)) * (Y1 - Y0);

  E.register("linreg-poly", (el) => {
    const svg = E.svg(el, 640, 296, "10個の訓練データに多項式を当てはめた曲線。次数と正則化の強さで形が変わる図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "d", label: "次数\\ d", min: 0, max: 9, step: 1, value: 9, digits: 0 },
        { key: "ll", label: "\\log_{10}\\lambda", min: -6, max: 1, step: 0.5, value: -6, digits: 1 },
      ],
      () => draw()
    );
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "灰の破線＝真の関数 sin(2πx)、●＝訓練データ（ノイズあり）、紺の実線＝当てはめた曲線。log₁₀λ=−6 はほぼ正則化なし。テスト誤差は別の200点で測る。"));

    // 目盛
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
    });
    [-2, -1, 0, 1, 2].forEach((v) => {
      svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
      svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
    });
    const tp = [];
    for (let i = 0; i <= 100; i++) tp.push(sx(i / 100).toFixed(1) + "," + sy(f0(i / 100)).toFixed(1));
    svg.append(E.s("polyline", { points: tp.join(" "), class: "ew-edge", style: "stroke:var(--muted);stroke-dasharray:4 3;stroke-width:1.2" }));
    TRAIN.xs.forEach((x, i) => svg.append(E.s("circle", { "data-k": "pts", cx: sx(x), cy: sy(TRAIN.ys[i]), r: 4.5, class: "ew-dot-f" })));
    const curve = E.s("polyline", { "data-k": "fit", class: "ew-edge", style: "stroke:var(--main);stroke-width:2.2" });
    svg.append(curve);

    function draw() {
      const { d, ll } = sl.get();
      const lam = Math.pow(10, ll);
      const f = E.fmt;
      const w = fit(TRAIN.xs, TRAIN.ys, d, lam);
      const pts = [];
      for (let i = 0; i <= 200; i++) pts.push(sx(i / 200).toFixed(1) + "," + sy(predict(w, i / 200)).toFixed(1));
      curve.setAttribute("points", pts.join(" "));
      const w2 = w.slice(1).reduce((s, v) => s + v * v, 0);
      E.formulaRows(rows, svg, [
        { tex: "\\hat{y}=\\sum_{j=0}^{d}w_j\\,(2x-1)^j", num: "d=" + d, hl: ["fit"] },
        { tex: "\\text{訓練 RMSE}=\\sqrt{\\tfrac1N\\textstyle\\sum_n(\\hat y_n-y_n)^2}", num: f(rmse(w, TRAIN)), hl: ["pts", "fit"] },
        { tex: "\\text{テスト RMSE}\\ （汎化誤差の推定）", num: f(rmse(w, TEST)), hl: ["fit"] },
        { tex: "\\|\\mathbf{w}\\|_2^2\\ （切片を除く）", num: E.fmtSci(w2), hl: ["fit"] },
      ]);
    }

    draw();
  });
})();
