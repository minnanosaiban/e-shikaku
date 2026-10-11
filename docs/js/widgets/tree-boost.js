// 勾配ブースティングとバギング：回帰木を足し合わせる（逐次）／平均する（並列）
(function () {
  const E = window.ENote;

  const f0 = (x) => Math.sin(2 * Math.PI * x);
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
  const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const mk = (n, seed, sd, jitter) => {
    const r = rng(seed), xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const x = jitter ? (i + 0.5) / n + (r() - 0.5) * 0.02 : r();
      xs.push(x);
      ys.push(f0(x) + sd * gauss(r));
    }
    return { xs, ys };
  };
  const TRAIN = mk(30, 11, 0.3, true);
  const TEST = mk(200, 5, 0.3, false);

  // 回帰木（二乗誤差の減少が最大の分割を貪欲に選ぶ）。{ th, l, r } または { v }
  function buildTree(xs, ys, idx, depth) {
    const mean = idx.reduce((s, i) => s + ys[i], 0) / idx.length;
    if (depth === 0 || idx.length < 2) return { v: mean };
    const sorted = idx.slice().sort((a, b) => xs[a] - xs[b]);
    const n = sorted.length;
    let sumAll = 0;
    sorted.forEach((i) => (sumAll += ys[i]));
    // 分割後の SSE 合計 = Σy² - (ΣL)²/nL - (ΣR)²/nR を最小化 ⇔ (ΣL)²/nL + (ΣR)²/nR を最大化
    let best = -1, bk = -1, sl = 0;
    for (let k = 0; k < n - 1; k++) {
      sl += ys[sorted[k]];
      if (xs[sorted[k]] === xs[sorted[k + 1]]) continue;
      const g = (sl * sl) / (k + 1) + ((sumAll - sl) * (sumAll - sl)) / (n - k - 1);
      if (g > best + 1e-12) { best = g; bk = k; }
    }
    if (bk < 0) return { v: mean };
    const th = (xs[sorted[bk]] + xs[sorted[bk + 1]]) / 2;
    return {
      th,
      l: buildTree(xs, ys, sorted.slice(0, bk + 1), depth - 1),
      r: buildTree(xs, ys, sorted.slice(bk + 1), depth - 1),
    };
  }
  const predTree = (t, x) => (t.v !== undefined ? t.v : x <= t.th ? predTree(t.l, x) : predTree(t.r, x));

  // 勾配ブースティング（二乗損失）：F_0=平均、F_t=F_{t-1}+η h_t、h_t は残差 y-F に当てはめる
  function boost(T, eta, depth) {
    const n = TRAIN.xs.length, all = [...Array(n).keys()];
    const F0 = TRAIN.ys.reduce((s, v) => s + v, 0) / n;
    const trees = [];
    const F = new Array(n).fill(F0);
    for (let t = 0; t < T; t++) {
      const r = TRAIN.ys.map((y, i) => y - F[i]);
      const tr = buildTree(TRAIN.xs, r, all, depth);
      trees.push(tr);
      for (let i = 0; i < n; i++) F[i] += eta * predTree(tr, TRAIN.xs[i]);
    }
    return (x) => trees.reduce((s, tr) => s + eta * predTree(tr, x), F0);
  }
  // バギング：ブートストラップ標本ごとに木を作り、平均する
  const bagCache = {};
  function bag(B, depth) {
    const n = TRAIN.xs.length;
    const trees = [];
    for (let b = 0; b < B; b++) {
      const key = b + ":" + depth;
      if (!bagCache[key]) {
        const r = rng(1000 + b);
        const idx = Array.from({ length: n }, () => Math.floor(r() * n));
        bagCache[key] = buildTree(TRAIN.xs, TRAIN.ys, idx, depth);
      }
      trees.push(bagCache[key]);
    }
    return (x) => trees.reduce((s, tr) => s + predTree(tr, x), 0) / B;
  }
  const rmse = (F, D) => Math.sqrt(D.xs.reduce((s, x, i) => s + (F(x) - D.ys[i]) ** 2, 0) / D.xs.length);

  if (typeof module !== "undefined") module.exports = { TRAIN, TEST, buildTree, predTree, boost, bag, rmse };
  if (!E) return;

  const X0 = 50, X1 = 610, Y0 = 20, Y1 = 250, YR = 1.8;
  const sx = (x) => X0 + x * (X1 - X0);
  const sy = (y) => Y1 - ((Math.min(YR, Math.max(-YR, y)) + YR) / (2 * YR)) * (Y1 - Y0);

  E.register("tree-boost", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "boost", label: "勾配ブースティング（逐次に足す）" }, { value: "bag", label: "バギング（平均する）" }],
      "boost",
      (v) => {
        setSlider(2, v === "boost" ? 2 : 4);
        setSlider(1, v === "boost" ? 0.3 : 1);
        draw();
      }
    );
    const svg = E.svg(el, 640, 280, "回帰木を足し合わせる（ブースティング）か平均する（バギング）ことで、データへの当てはめがどう変わるかを示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "T", label: "T\\ （木の本数）", min: 1, max: 50, step: 1, value: 10, digits: 0 },
        { key: "eta", label: "\\eta", min: 0.05, max: 1, step: 0.05, value: 0.3 },
        { key: "depth", label: "木の深さ", min: 1, max: 4, step: 1, value: 2, digits: 0 },
      ],
      () => draw()
    );
    const inputs = [...el.querySelectorAll(".ew-sliders input")];
    function setSlider(i, v) {
      inputs[i].value = v;
      inputs[i].dispatchEvent(new Event("input"));
    }
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "●＝訓練データ（30点、ノイズあり）、灰の破線＝真の関数 sin(2πx)、紺の階段＝モデルの予測。η はブースティングのときだけ効く。テスト誤差は別の200点で測る。"));

    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
    });
    [-1, 0, 1].forEach((v) => {
      svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
      svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
    });
    const tp = [];
    for (let i = 0; i <= 100; i++) tp.push(sx(i / 100).toFixed(1) + "," + sy(f0(i / 100)).toFixed(1));
    svg.append(E.s("polyline", { points: tp.join(" "), class: "ew-edge", style: "stroke:var(--muted);stroke-dasharray:4 3;stroke-width:1.2" }));
    TRAIN.xs.forEach((x, i) => svg.append(E.s("circle", { "data-k": "pts", cx: sx(x), cy: sy(TRAIN.ys[i]), r: 4, class: "ew-dot-f" })));
    const curve = E.s("polyline", { "data-k": "fit", class: "ew-edge", style: "stroke:var(--main);stroke-width:2.2" });
    svg.append(curve);

    function draw() {
      const { T, eta, depth } = sl.get();
      const f = E.fmt;
      const isBoost = mode.get() === "boost";
      const F = isBoost ? boost(T, eta, depth) : bag(T, depth);
      const pts = [];
      for (let i = 0; i <= 280; i++) pts.push(sx(i / 280).toFixed(1) + "," + sy(F(i / 280)).toFixed(1));
      curve.setAttribute("points", pts.join(" "));
      const rowsDef = isBoost
        ? [
            { tex: "r_t^{(n)}=y_n-F_{t-1}(x_n)\\ \\text{（二乗損失の負の勾配）}", num: "t=1,\\ldots," + T, hl: ["pts"] },
            { tex: "F_t(x)=F_{t-1}(x)+\\eta\\,h_t(x)", num: "\\eta=" + f(eta, 2) + ",\\ \\text{深さ}\\ " + depth, hl: ["fit"] },
          ]
        : [
            { tex: "\\text{ブートストラップ標本 }b=1,\\ldots,B\\text{ で木 }h_b\\text{ を学習}", num: "B=" + T + ",\\ \\text{深さ}\\ " + depth, hl: ["pts"] },
            { tex: "\\hat{y}(x)=\\dfrac1B\\sum_{b=1}^{B}h_b(x)", num: "\\text{（}\\eta\\text{ は使わない）}", hl: ["fit"] },
          ];
      rowsDef.push({ tex: "\\text{訓練 RMSE}", num: f(rmse(F, TRAIN)), hl: ["pts", "fit"] });
      rowsDef.push({ tex: "\\text{テスト RMSE}\\ \\text{（汎化誤差の推定）}", num: f(rmse(F, TEST)), hl: ["fit"] });
      E.formulaRows(rows, svg, rowsDef);
    }

    draw();
  });
})();
