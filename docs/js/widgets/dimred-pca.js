// 主成分分析：射影する向き θ を回して、射影後の分散 s² が固有値で最大になる様子を見る
(function () {
  // ---- 計算（node でも検算できるように、描画から切り離す）----
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
  function normals(n, seed) {
    const r = rng(seed), out = [];
    while (out.length < n) {
      const u = Math.max(r(), 1e-9), v = r();
      out.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
      out.push(Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v));
    }
    return out.slice(0, n);
  }
  const N = 40;
  const Z1 = normals(N, 11), Z2 = normals(N, 29);
  const PHI = (30 * Math.PI) / 180; // データの長軸の向き
  const S1 = 2.4; // 長軸方向の標準偏差

  // 比 r = 短軸/長軸 のデータを作り、平均を 0 にそろえる
  function makeData(r) {
    const m1 = Z1.reduce((s, v) => s + v, 0) / N, m2 = Z2.reduce((s, v) => s + v, 0) / N;
    return Z1.map((_, i) => {
      const a = (Z1[i] - m1) * S1, b = (Z2[i] - m2) * S1 * r;
      return [a * Math.cos(PHI) - b * Math.sin(PHI), a * Math.sin(PHI) + b * Math.cos(PHI)];
    });
  }
  // 共分散行列 [[a, b], [b, c]]（1/N）と固有値
  function cov(P) {
    let a = 0, b = 0, c = 0;
    P.forEach(([x, y]) => { a += x * x; b += x * y; c += y * y; });
    a /= P.length; b /= P.length; c /= P.length;
    const m = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    return { a, b, c, l1: m + d, l2: m - d, ang: 0.5 * Math.atan2(2 * b, a - c) };
  }
  const var1 = (S, th) => {
    const co = Math.cos(th), si = Math.sin(th);
    return S.a * co * co + 2 * S.b * co * si + S.c * si * si;
  };

  if (typeof module !== "undefined") module.exports = { makeData, cov, var1 };
  if (typeof window === "undefined") return;

  // ---- 描画 ----
  const E = window.ENote;

  E.register("dimred-pca", (el) => {
    const svg = E.svg(el, 640, 300, "2次元の点群を直線に射影したときの分散と、向きによる分散の変化を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "th", label: "\\theta\\ [^\\circ]", min: 0, max: 180, step: 1, value: 75, digits: 0 },
        { key: "r", label: "r=\\sigma_2/\\sigma_1", min: 0.1, max: 1, step: 0.05, value: 0.35 },
      ],
      () => draw()
    );

    const CX = 175, CY = 150, U = 27; // 左の図：原点の位置と 1 単位あたりの画素数
    const px = (x) => CX + x * U, py = (y) => CY - y * U;
    const RX0 = 410, RX1 = 610, RY0 = 40, RY1 = 240; // 右の図
    const rx = (th) => RX0 + (th / Math.PI) * (RX1 - RX0);

    function draw() {
      const { th: thDeg, r } = sl.get();
      const f = E.fmt;
      const th = (thDeg * Math.PI) / 180;
      const P = makeData(r), S = cov(P), tr = S.a + S.c;
      const w = [Math.cos(th), Math.sin(th)];
      const s2 = var1(S, th);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 左：データと射影
      svg.append(E.s("line", { x1: px(-6), y1: CY, x2: px(6), y2: CY, class: "ew-grid" }));
      P.forEach(([x, y]) => {
        const t = x * w[0] + y * w[1];
        svg.append(E.s("line", { "data-k": "res", x1: px(x), y1: py(y), x2: px(t * w[0]), y2: py(t * w[1]), class: "ew-edge", style: "stroke-width:0.8" }));
      });
      svg.append(E.s("line", {
        "data-k": "ax", x1: px(-4.6 * w[0]), y1: py(-4.6 * w[1]), x2: px(4.6 * w[0]), y2: py(4.6 * w[1]),
        class: "ew-edge", style: "stroke:var(--main);stroke-width:1.6",
      }));
      P.forEach(([x, y]) => {
        svg.append(E.s("circle", { cx: px(x), cy: py(y), r: 3.2, class: "ew-dot-f", style: "opacity:0.55" }));
      });
      P.forEach(([x, y]) => {
        const t = x * w[0] + y * w[1];
        svg.append(E.s("circle", { "data-k": "foot", cx: px(t * w[0]), cy: py(t * w[1]), r: 2.6, class: "ew-dot-d" }));
      });
      svg.append(E.s("text", { x: 14, y: 12, class: "ew-s ew-start" }, "データ（グレー）と射影（紺）"));

      // 右：向きごとの分散
      svg.append(E.s("text", { x: RX0, y: 16, class: "ew-s ew-start" }, "射影後の分散 s²"));
      const ymax = S.l1 * 1.15;
      const ry = (v) => RY1 - (v / ymax) * (RY1 - RY0);
      [0, 90, 180].forEach((d) => {
        svg.append(E.s("line", { x1: rx((d * Math.PI) / 180), y1: RY0, x2: rx((d * Math.PI) / 180), y2: RY1, class: "ew-grid" }));
        svg.append(E.s("text", { x: rx((d * Math.PI) / 180), y: RY1 + 16, class: "ew-s ew-c" }, d + "°"));
      });
      svg.append(E.s("text", { x: (RX0 + RX1) / 2, y: RY1 + 38, class: "ew-s ew-c" }, "向き θ"));
      [[S.l1, "λ₁", "l1"], [S.l2, "λ₂", "l2"]].forEach(([v, lab, k]) => {
        svg.append(E.s("line", { "data-k": k, x1: RX0, y1: ry(v), x2: RX1, y2: ry(v), class: "ew-edge", "stroke-dasharray": "4 3" }));
        svg.append(E.s("text", { x: RX0 - 8, y: ry(v), class: "ew-s ew-end" }, lab));
      });
      const pts = [];
      for (let i = 0; i <= 90; i++) {
        const t = (i / 90) * Math.PI;
        pts.push(rx(t).toFixed(1) + "," + ry(var1(S, t)).toFixed(1));
      }
      svg.append(E.s("polyline", { "data-k": "foot", points: pts.join(" "), class: "ew-curve-f" }));
      svg.append(E.s("circle", { "data-k": "foot", cx: rx(th), cy: ry(s2), r: 5, class: "ew-dot-d" }));

      const angDeg = ((S.ang * 180) / Math.PI + 180) % 180;
      E.formulaRows(rows, svg, [
        { tex: "\\mathbf{w}=(\\cos\\theta,\\ \\sin\\theta)", num: "=(" + f(w[0]) + ",\\ " + f(w[1]) + ")", hl: ["ax"] },
        { tex: "s^2=\\mathbf{w}^{\\top}\\boldsymbol{\\Sigma}\\,\\mathbf{w}", num: "=" + f(s2), hl: ["foot"] },
        { tex: "\\text{残差}=\\operatorname{tr}\\boldsymbol{\\Sigma}-s^2", num: "=" + f(tr) + "-" + f(s2) + "=" + f(tr - s2), hl: ["res"] },
        { tex: "\\text{寄与率}=s^2/\\operatorname{tr}\\boldsymbol{\\Sigma}", num: "=" + f(s2 / tr), hl: ["foot"] },
        { tex: "\\lambda_1,\\ \\lambda_2\\ （固有値）", num: "=" + f(S.l1) + ",\\ " + f(S.l2) + "\\quad（第1主成分の向き " + f(angDeg, 0) + "^\\circ）", hl: ["l1", "l2"] },
      ]);
    }
    draw();
  });
})();
