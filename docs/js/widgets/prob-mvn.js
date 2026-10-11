// 2次元の正規分布：共分散行列 Σ の固有ベクトルが楕円の主軸、固有値が主軸方向の分散
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 対称な 2×2 行列 [[A,B],[B,C]] の固有値（大きい順）と、第1固有ベクトルの角度
  E.math.eigSym2 = function (A, B, C) {
    const half = (A + C) / 2, rad = Math.sqrt(((A - C) / 2) ** 2 + B * B);
    return { l1: half + rad, l2: Math.max(half - rad, 0), phi: 0.5 * Math.atan2(2 * B, A - C) };
  };

  // 乱数（再現できるように種つき）。標準正規乱数を Box–Muller 法で作る
  function makeNormals(n, seed) {
    let s = seed >>> 0;
    const u = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
    const out = [];
    for (let i = 0; i < n; i++) {
      const r = Math.sqrt(-2 * Math.log(u())), t = 2 * Math.PI * u();
      out.push([r * Math.cos(t), r * Math.sin(t)]);
    }
    return out;
  }
  E.math.makeNormals = makeNormals;

  const CX = 320, CY = 172, SC = 30; // 1 目盛 = 30px、範囲 ±4.5 程度
  const px = (x) => CX + SC * x;
  const py = (y) => CY - SC * y;
  const Z = makeNormals(300, 12345);

  E.register("prob-mvn", (el) => {
    const svg = E.svg(el, 640, 340, "2次元正規分布のサンプル点と、マハラノビス距離 1 と 2 の等高線（楕円）。楕円の主軸は共分散行列の固有ベクトルの向き");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "sx", label: "\\sigma_x", min: 0.5, max: 1.5, step: 0.1, value: 1.2 },
        { key: "sy", label: "\\sigma_y", min: 0.5, max: 1.5, step: 0.1, value: 0.8 },
        { key: "rho", label: "\\rho", min: -0.95, max: 0.95, step: 0.05, value: 0.6 },
      ],
      () => draw()
    );

    function draw() {
      const { sx, sy, rho } = sl.get();
      const f = E.fmt;
      const cxy = rho * sx * sy;
      const ev = E.math.eigSym2(sx * sx, cxy, sy * sy);
      const e1 = [Math.cos(ev.phi), Math.sin(ev.phi)], e2 = [-Math.sin(ev.phi), Math.cos(ev.phi)];
      const r1 = Math.sqrt(ev.l1), r2 = Math.sqrt(ev.l2);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      for (let k = -4; k <= 4; k++) {
        svg.append(E.s("line", { x1: px(k), y1: py(-4), x2: px(k), y2: py(4), class: "ew-grid" }));
        svg.append(E.s("line", { x1: px(-4), y1: py(k), x2: px(4), y2: py(k), class: "ew-grid" }));
      }
      svg.append(E.s("line", { x1: px(-4.4), y1: py(0), x2: px(4.4), y2: py(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: px(0), y1: py(-4.4), x2: px(0), y2: py(4.4), class: "ew-edge" }));

      // サンプル点（枠の外に出たものは描かない）
      Z.forEach(([z1, z2]) => {
        const x = sx * z1, y = sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
        if (Math.abs(x) > 4.4 || Math.abs(y) > 4.0) return;
        svg.append(E.s("circle", { cx: px(x).toFixed(1), cy: py(y).toFixed(1), r: 2.3, class: "ew-dot-f", "fill-opacity": 0.4, "data-k": "smp" }));
      });
      // 主軸（灰）
      [[e1, r1], [e2, r2]].forEach(([e, r]) => {
        svg.append(E.s("line", { x1: px(-2 * r * e[0]), y1: py(-2 * r * e[1]), x2: px(2 * r * e[0]), y2: py(2 * r * e[1]), class: "ew-edge", style: "stroke-width:1.4", "data-k": "ax" }));
      });
      // 等高線：マハラノビス距離 d=1（実線）、d=2（破線）
      [[1, "c1", "stroke:var(--main);fill:none;stroke-width:2"], [2, "c2", "stroke:var(--main);fill:none;stroke-width:1.6;stroke-dasharray:5 3"]].forEach(([d, k, st]) => {
        const pts = [];
        for (let i = 0; i <= 120; i++) {
          const t = (2 * Math.PI * i) / 120;
          const a = d * r1 * Math.cos(t), b = d * r2 * Math.sin(t);
          pts.push(px(a * e1[0] + b * e2[0]).toFixed(1) + "," + py(a * e1[1] + b * e2[1]).toFixed(1));
        }
        svg.append(E.s("polyline", { points: pts.join(" "), style: st, "data-k": k }));
      });

      svg.append(E.s("text", { x: 16, y: 16, class: "ew-s ew-start", style: "fill:var(--main)" }, "紺：マハラノビス距離 d=1（実線）、d=2（破線）"));
      svg.append(E.s("text", { x: 16, y: 36, class: "ew-s ew-start" }, "灰の線：主軸（Σ の固有ベクトルの向き）、点：サンプル"));

      const det = ev.l1 * ev.l2;
      const sig = "\\begin{pmatrix}" + f(sx * sx, 2) + "&" + f(cxy, 2) + "\\\\" + f(cxy, 2) + "&" + f(sy * sy, 2) + "\\end{pmatrix}";
      E.formulaRows(rows, svg, [
        { tex: "\\boldsymbol{\\Sigma}=\\begin{pmatrix}\\sigma_x^2&\\rho\\sigma_x\\sigma_y\\\\\\rho\\sigma_x\\sigma_y&\\sigma_y^2\\end{pmatrix}", num: "=" + sig, hl: ["smp", "c1", "c2"] },
        { tex: "\\lambda_1,\\ \\lambda_2\\ (\\boldsymbol{\\Sigma}\\text{ の固有値})", num: "=" + f(ev.l1, 3) + ",\\ " + f(ev.l2, 3) + "\\quad(\\sqrt{\\lambda}=" + f(r1, 2) + ",\\ " + f(r2, 2) + ")", hl: ["ax"] },
        { tex: "(\\mathbf{x}-\\boldsymbol{\\mu})^{\\top}\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})=d^2", num: "d=1,\\ 2\\ \\text{の等高線}", hl: ["c1", "c2"] },
        { tex: "\\mathcal{N}(\\boldsymbol{\\mu}\\mid\\boldsymbol{\\mu},\\boldsymbol{\\Sigma})=\\dfrac{1}{2\\pi\\sqrt{\\det\\boldsymbol{\\Sigma}}}", num: "\\det\\boldsymbol{\\Sigma}=" + f(det, 3) + ",\\ \\text{山の高さ}=" + f(1 / (2 * Math.PI * Math.sqrt(det)), 3), hl: ["c1"] },
      ]);
    }
    draw();
  });
})();
