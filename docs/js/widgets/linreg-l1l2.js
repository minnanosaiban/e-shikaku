// L1・L2 正則化の幾何：二乗誤差の等高線と、罰則項の等高線が接する点が解
(function () {
  const E = window.ENote;

  const W0 = [1.6, 0.7]; // 正則化なしの解（最小二乗解）
  const ALPHA = 0.5;     // Elastic Net の混合比

  const R = {
    l2: (w) => 0.5 * (w[0] * w[0] + w[1] * w[1]),
    l1: (w) => Math.abs(w[0]) + Math.abs(w[1]),
    en: (w) => ALPHA * (Math.abs(w[0]) + Math.abs(w[1])) + ((1 - ALPHA) / 2) * (w[0] * w[0] + w[1] * w[1]),
  };
  const soft = (z, t) => Math.sign(z) * Math.max(Math.abs(z) - t, 0);

  // J(w)=½(w-w0)ᵀA(w-w0)+λR(w)、A=[[1,ρ],[ρ,1]] を座標降下法で最小化する
  function solve(kind, lam, rho) {
    const A = [[1, rho], [rho, 1]];
    let w = W0.slice();
    for (let it = 0; it < 400; it++) {
      for (let j = 0; j < 2; j++) {
        const d0 = w[0] - W0[0], d1 = w[1] - W0[1];
        const g = A[j][0] * d0 + A[j][1] * d1;
        const z = A[j][j] * w[j] - g; // 他の係数を固定したときの、罰則なしの最小点 × A_jj
        if (kind === "l2") w[j] = z / (A[j][j] + lam);
        else if (kind === "l1") w[j] = soft(z, lam) / A[j][j];
        else w[j] = soft(z, lam * ALPHA) / (A[j][j] + lam * (1 - ALPHA));
      }
    }
    return w;
  }
  const J = (w, kind, lam, rho) => {
    const d0 = w[0] - W0[0], d1 = w[1] - W0[1];
    return 0.5 * (d0 * d0 + 2 * rho * d0 * d1 + d1 * d1) + lam * R[kind](w);
  };
  const lossOnly = (w, rho) => {
    const d0 = w[0] - W0[0], d1 = w[1] - W0[1];
    return 0.5 * (d0 * d0 + 2 * rho * d0 * d1 + d1 * d1);
  };
  // 方向 θ で R(w)=r となる点（R は原点から単調に増えるので二分法）
  function ray(kind, th, r) {
    const c = Math.cos(th), s = Math.sin(th);
    let lo = 0, hi = 100;
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (R[kind]([m * c, m * s]) < r) lo = m; else hi = m;
    }
    return [lo * c, lo * s];
  }
  // 損失の等高線 ½dᵀAd=c（A の固有値 1+ρ, 1-ρ、固有ベクトル (1,1)/√2, (1,-1)/√2）
  function ellipse(c, rho, th) {
    const a = Math.cos(th) * Math.sqrt(2 * c / (1 + rho)), b = Math.sin(th) * Math.sqrt(2 * c / (1 - rho));
    return [W0[0] + (a + b) / Math.SQRT2, W0[1] + (a - b) / Math.SQRT2];
  }

  if (typeof module !== "undefined") module.exports = { solve, J, R, ray, ellipse, lossOnly, W0 };
  if (!E) return;

  const SC = 68, OX = 166, OY = 184; // w1∈[-2,3]、w2∈[-1.5,2.5]
  const sx = (x) => OX + SC * x;
  const sy = (y) => OY - SC * y;
  const NAMES = { l2: "L2（Ridge）", l1: "L1（Lasso）", en: "Elastic Net" };
  const RTEX = {
    l2: "R(\\mathbf{w})=\\tfrac12(w_1^2+w_2^2)",
    l1: "R(\\mathbf{w})=|w_1|+|w_2|",
    en: "R(\\mathbf{w})=\\tfrac12\\|\\mathbf{w}\\|_1+\\tfrac14\\|\\mathbf{w}\\|_2^2",
  };

  E.register("linreg-l1l2", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "l2", label: "L2（Ridge）" }, { value: "l1", label: "L1（Lasso）" }, { value: "en", label: "Elastic Net（α=0.5）" }],
      "l1",
      () => draw()
    );
    const svg = E.svg(el, 640, 300, "二乗誤差の楕円の等高線と、L1 または L2 の罰則の等高線が接する点が正則化付きの解になることを示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "lam", label: "\\lambda", min: 0, max: 3, step: 0.02, value: 0.6 },
        { key: "rho", label: "\\rho", min: 0, max: 0.9, step: 0.05, value: 0.6 },
      ],
      () => draw()
    );
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "楕円＝二乗誤差が等しい点（中心が最小二乗解）。紺の太線＝罰則 R が解と同じ値の点の集まり。ρ は特徴量どうしの相関で、大きいほど楕円が細長くなる。"));

    const clipId = "ew-clip-" + Math.random().toString(36).slice(2, 8);
    const cp = E.s("clipPath", { id: clipId });
    cp.append(E.s("rect", { x: 30, y: 14, width: 340, height: 272 }));
    svg.childNodes[1].append(cp);
    svg.append(E.s("g", {})); // 位置合わせ用（以降が描き直し対象）

    function draw() {
      const kind = mode.get();
      const { lam, rho } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 3) svg.lastChild.remove();

      // 軸
      [-2, -1, 0, 1, 2, 3].forEach((v) => svg.append(E.s("line", { x1: sx(v), y1: sy(-1.5), x2: sx(v), y2: sy(2.5), class: "ew-grid" })));
      [-1, 0, 1, 2].forEach((v) => svg.append(E.s("line", { x1: sx(-2), y1: sy(v), x2: sx(3), y2: sy(v), class: "ew-grid" })));
      svg.append(E.s("line", { x1: sx(-2), y1: sy(0), x2: sx(3), y2: sy(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: sx(0), y1: sy(-1.5), x2: sx(0), y2: sy(2.5), class: "ew-edge" }));
      svg.append(E.s("text", { x: sx(3) - 4, y: sy(0) + 14, class: "ew-s ew-end" }, "w₁"));
      svg.append(E.s("text", { x: sx(0) + 8, y: sy(2.5) + 6, class: "ew-s ew-start" }, "w₂"));

      const poly = (g, n, k, style) => {
        const pts = [];
        for (let i = 0; i <= n; i++) {
          const [a, b] = g((i / n) * 2 * Math.PI);
          pts.push(sx(a).toFixed(1) + "," + sy(b).toFixed(1));
        }
        svg.append(E.s("polyline", { "data-k": k, points: pts.join(" "), class: "ew-edge", style, "clip-path": "url(#" + clipId + ")" }));
      };

      const w = solve(kind, lam, rho);
      const Lw = lossOnly(w, rho);
      // 二乗誤差の等高線（解を通る線だけ紺）
      [0.1, 0.4, 1, 2].forEach((c) => poly((t) => ellipse(c, rho, t), 100, "x", "stroke:var(--line-strong);stroke-width:1"));
      if (Lw > 1e-9) poly((t) => ellipse(Lw, rho, t), 100, "ell", "stroke:var(--main);stroke-width:1.2");
      // 罰則の等高線
      const rv = R[kind](w);
      if (rv > 1e-9) poly((t) => ray(kind, t, rv), 120, "reg", "stroke:var(--main);stroke-width:2.4");

      // λ を 0 から増やしたときの解の軌跡
      const path = [];
      for (let i = 0; i <= 60; i++) {
        const q = solve(kind, (i / 60) * 3, rho);
        path.push(sx(q[0]).toFixed(1) + "," + sy(q[1]).toFixed(1));
      }
      svg.append(E.s("polyline", { "data-k": "path", points: path.join(" "), class: "ew-edge", style: "stroke:var(--muted);stroke-width:1.6;stroke-dasharray:2 3" }));

      svg.append(E.s("circle", { "data-k": "w0", cx: sx(W0[0]), cy: sy(W0[1]), r: 4.5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.8" }));
      svg.append(E.s("circle", { "data-k": "w", cx: sx(w[0]), cy: sy(w[1]), r: 5.5, class: "ew-dot-d" }));

      // 凡例（右の余白）
      const lg = (y, text, kind2) => {
        if (kind2 === "w0") svg.append(E.s("circle", { cx: 412, cy: y, r: 4.5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.8" }));
        else if (kind2 === "w") svg.append(E.s("circle", { cx: 412, cy: y, r: 5, class: "ew-dot-d" }));
        else svg.append(E.s("line", { x1: 402, y1: y, x2: 422, y2: y, class: "ew-edge", style: kind2 }));
        svg.append(E.s("text", { x: 434, y, class: "ew-s ew-start" }, text));
      };
      lg(60, "最小二乗解", "w0");
      lg(86, "正則化付きの解", "w");
      lg(112, "λ を動かしたときの軌跡", "stroke:var(--muted);stroke-width:1.6;stroke-dasharray:2 3");
      lg(138, "罰則 R の等高線", "stroke:var(--main);stroke-width:2.4");
      lg(164, "二乗誤差の等高線", "stroke:var(--main);stroke-width:1.2");

      E.formulaRows(rows, svg, [
        { tex: "J(\\mathbf{w})=\\tfrac12(\\mathbf{w}-\\mathbf{w}_0)^{\\top}\\mathbf{A}(\\mathbf{w}-\\mathbf{w}_0)+\\lambda R(\\mathbf{w})", num: "J=" + f(J(w, kind, lam, rho)) + "\\ \\text{（" + NAMES[kind] + "）}", hl: ["ell", "reg"] },
        { tex: RTEX[kind], num: "R=" + f(rv), hl: ["reg"] },
        { tex: "\\hat{\\mathbf{w}}=\\mathop{\\rm argmin}_{\\mathbf{w}}J(\\mathbf{w})", num: "\\hat{\\mathbf{w}}=(" + f(w[0]) + ",\\ " + f(w[1]) + ")", hl: ["w", "path"] },
        { tex: "\\|\\hat{\\mathbf{w}}\\|_0\\ \\text{（0 でない係数の数）}", num: String(w.filter((v) => Math.abs(v) > 1e-9).length), hl: ["w"] },
      ]);
    }

    draw();
  });
})();
