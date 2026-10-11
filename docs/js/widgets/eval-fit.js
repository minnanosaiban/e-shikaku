// 過剰適合：多項式の次数と L2 正則化の強さを変え、訓練誤差と検証誤差の食い違いを見る
(function () {
  // ---- 計算 ----
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
  const gauss = (r) => Math.sqrt(-2 * Math.log(Math.max(r(), 1e-9))) * Math.cos(2 * Math.PI * r());
  const truth = (x) => Math.sin(2 * Math.PI * x);
  const SIGMA = 0.25;
  const NTR = 10, NTE = 200;
  const mk = (n, seed, grid) => {
    const r = rng(seed);
    return Array.from({ length: n }, (_, i) => {
      const x = grid ? (i + 0.5) / n : r();
      return [x, truth(x) + SIGMA * gauss(r)];
    });
  };
  const TRAIN = mk(NTR, 5, true), TEST = mk(NTE, 99, false);

  // x ∈ [0,1] を [-1,1] に写して、M 次の多項式特徴にする
  const feat = (x, M) => Array.from({ length: M + 1 }, (_, j) => (2 * x - 1) ** j);
  function solve(A, b) {
    const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      [M[c], M[p]] = [M[p], M[c]];
      for (let i = c + 1; i < n; i++) {
        const q = M[i][c] / M[c][c];
        for (let j = c; j <= n; j++) M[i][j] -= q * M[c][j];
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
  // リッジ回帰：(ΦᵀΦ + λI')w = Φᵀy（バイアス項は罰しない）
  function fit(M, lam) {
    const Phi = TRAIN.map(([x]) => feat(x, M)), n = M + 1;
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => Phi.reduce((s, row) => s + row[i] * row[j], 0) + (i === j && i > 0 ? lam : 0)));
    const b = Array.from({ length: n }, (_, i) => Phi.reduce((s, row, k) => s + row[i] * TRAIN[k][1], 0));
    return solve(A, b);
  }
  const predict = (w, x) => feat(x, w.length - 1).reduce((s, v, j) => s + v * w[j], 0);
  const mse = (w, D) => D.reduce((s, [x, y]) => s + (predict(w, x) - y) ** 2, 0) / D.length;

  if (typeof module !== "undefined") module.exports = { fit, predict, mse, TRAIN, TEST };
  if (typeof window === "undefined") return;

  // ---- 描画 ----
  const E = window.ENote;
  const MAXM = 9;
  const SUP = "⁻⁰¹²³⁴⁵⁶⁷⁸⁹";
  const sup = (n) => String(n).replace(/[-0-9]/g, (c) => SUP["-0123456789".indexOf(c)]);

  // 桁が大きく変わる値の表示
  const sci = (v) => {
    if (v >= 0.01 && v < 1000) return E.fmt(v, 3);
    const [m, e] = v.toExponential(2).split("e");
    return m + "\\times 10^{" + Number(e) + "}";
  };

  E.register("eval-fit", (el) => {
    const svg = E.svg(el, 640, 312, "多項式回帰のあてはめ曲線と、次数ごとの訓練誤差・検証誤差を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "M", label: "M\\ （次数）", min: 0, max: MAXM, step: 1, value: 9, digits: 0 },
        { key: "lg", label: "\\log_{10}\\lambda", min: -8, max: 1, step: 0.1, value: -8, digits: 1 },
      ],
      () => draw()
    );

    // 左：あてはめ。x ∈ [0,1]、y ∈ [-1.6, 1.6]
    const L0 = 40, L1 = 330, TOP = 24, BOT = 270;
    const lx = (x) => L0 + x * (L1 - L0);
    const ly = (y) => BOT - ((Math.min(1.6, Math.max(-1.6, y)) + 1.6) / 3.2) * (BOT - TOP);
    // 右：誤差（対数目盛、10^-2.5 〜 10^2）
    const R0 = 390, R1 = 610;
    const rx = (M) => R0 + (M / MAXM) * (R1 - R0);
    const LO = -2.5, HI = 2;
    const ry = (e) => BOT - ((Math.min(HI, Math.max(LO, Math.log10(Math.max(e, 1e-12)))) - LO) / (HI - LO)) * (BOT - TOP);

    function draw() {
      const { M, lg } = sl.get();
      const lam = Math.pow(10, lg);
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      const w = fit(M, lam);
      const trE = mse(w, TRAIN), teE = mse(w, TEST);

      // 左
      svg.append(E.s("rect", { x: L0, y: TOP, width: L1 - L0, height: BOT - TOP, class: "ew-node", style: "fill:none" }));
      const curve = (g) => {
        const out = [];
        for (let i = 0; i <= 120; i++) { const x = i / 120; out.push(lx(x).toFixed(1) + "," + ly(g(x)).toFixed(1)); }
        return out.join(" ");
      };
      svg.append(E.s("polyline", { points: curve(truth), class: "ew-edge", "stroke-dasharray": "4 3" }));
      svg.append(E.s("polyline", { "data-k": "fit", points: curve((x) => predict(w, x)), class: "ew-curve-f", style: "stroke:var(--main)" }));
      TRAIN.forEach(([x, y]) => svg.append(E.s("circle", { "data-k": "tr", cx: lx(x), cy: ly(y), r: 4, class: "ew-dot-f" })));
      svg.append(E.s("text", { x: L0, y: 10, class: "ew-s ew-start" }, "点線：真の関数　●：訓練データ　紺：あてはめ"));
      svg.append(E.s("text", { x: lx(0), y: BOT + 16, class: "ew-s ew-c" }, "0"));
      svg.append(E.s("text", { x: lx(1), y: BOT + 16, class: "ew-s ew-c" }, "1"));
      svg.append(E.s("text", { x: lx(0.5), y: BOT + 16, class: "ew-s ew-c" }, "x"));

      // 右
      svg.append(E.s("rect", { x: R0, y: TOP, width: R1 - R0, height: BOT - TOP, class: "ew-node", style: "fill:none" }));
      [-2, 0, 2].forEach((e) => {
        svg.append(E.s("line", { x1: R0, y1: ry(Math.pow(10, e)), x2: R1, y2: ry(Math.pow(10, e)), class: "ew-grid" }));
        svg.append(E.s("text", { x: R0 - 6, y: ry(Math.pow(10, e)), class: "ew-s ew-end" }, "10" + sup(e)));
      });
      for (let m = 0; m <= MAXM; m += 3) svg.append(E.s("text", { x: rx(m), y: BOT + 16, class: "ew-s ew-c" }, String(m)));
      svg.append(E.s("text", { x: rx(MAXM / 2), y: BOT + 34, class: "ew-s ew-c" }, "次数 M"));
      svg.append(E.s("text", { x: R0, y: 10, class: "ew-s ew-start" }, "平均二乗誤差（実線：訓練、破線：検証）"));
      const tr = [], te = [];
      for (let m = 0; m <= MAXM; m++) {
        const wm = fit(m, lam);
        tr.push(rx(m).toFixed(1) + "," + ry(mse(wm, TRAIN)).toFixed(1));
        te.push(rx(m).toFixed(1) + "," + ry(mse(wm, TEST)).toFixed(1));
      }
      svg.append(E.s("polyline", { "data-k": "trE", points: tr.join(" "), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "teE", points: te.join(" "), class: "ew-curve-d" }));
      svg.append(E.s("circle", { "data-k": "trE", cx: rx(M), cy: ry(trE), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "teE", cx: rx(M), cy: ry(teE), r: 4.5, class: "ew-dot-d" }));

      E.formulaRows(rows, svg, [
        { tex: "\\hat{y}(x)=\\sum_{j=0}^{M}w_j x^j", num: "M=" + M + "\\ \\text{（パラメータ" + (M + 1) + "個、訓練データ" + NTR + "個）}", hl: ["fit"] },
        { tex: "E_{\\text{train}}=\\dfrac{1}{N}\\sum(\\hat{y}-y)^2", num: "=" + sci(trE), hl: ["tr", "trE"] },
        { tex: "E_{\\text{valid}}\\ （検証データ" + NTE + "個）", num: "=" + sci(teE), hl: ["teE"] },
        { tex: "\\text{ノイズの分散}\\ \\sigma^2\\ （これ以下にはならない）", num: "=" + f(SIGMA * SIGMA), hl: [] },
      ]);
    }
    draw();
  });
})();
