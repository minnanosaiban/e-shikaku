// p ノルムの単位球：L1・L2・L∞ と任意の p。点 x の各ノルムの値を比べる
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // ‖x‖_p（p=Infinity なら最大ノルム）
  E.math.pnorm = function (v, p) {
    if (p === Infinity) return Math.max(...v.map(Math.abs));
    return Math.pow(v.reduce((s, x) => s + Math.pow(Math.abs(x), p), 0), 1 / p);
  };

  const CX = 320, CY = 150, SC = 100; // 範囲 ±1.4 程度
  const px = (x) => CX + SC * x;
  const py = (y) => CY - SC * y;

  // 単位球 ‖x‖_p = 1 の輪郭（角度 t でたどる）
  function ball(p) {
    const pts = [];
    const n = 240;
    for (let i = 0; i <= n; i++) {
      const t = (2 * Math.PI * i) / n;
      const c = Math.cos(t), s = Math.sin(t);
      let x, y;
      if (p === Infinity) {
        const m = Math.max(Math.abs(c), Math.abs(s));
        x = c / m; y = s / m;
      } else {
        const e = 2 / p;
        x = Math.sign(c) * Math.pow(Math.abs(c), e);
        y = Math.sign(s) * Math.pow(Math.abs(s), e);
      }
      pts.push(px(x).toFixed(1) + "," + py(y).toFixed(1));
    }
    return pts.join(" ");
  }

  E.register("linalg-norm", (el) => {
    const svg = E.svg(el, 640, 300, "p ノルムの単位球を比べる図。L1 はひし形、L2 は円、L∞ は正方形で、p を変えると中間の形になる");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "p", label: "p", min: 1, max: 10, step: 0.5, value: 3, digits: 1 },
        { key: "x1", label: "x_1", min: -1.3, max: 1.3, step: 0.05, value: 0.8 },
        { key: "x2", label: "x_2", min: -1.3, max: 1.3, step: 0.05, value: 0.5 },
      ],
      () => draw()
    );

    function draw() {
      const { p, x1, x2 } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      svg.append(E.s("line", { x1: px(-1.45), y1: py(0), x2: px(1.45), y2: py(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: px(0), y1: py(-1.4), x2: px(0), y2: py(1.4), class: "ew-edge" }));

      // 灰：L1・L2・L∞ の単位球（入れ子になる）
      [[1, "n1"], [2, "n2"], [Infinity, "ni"]].forEach(([q, k]) => {
        svg.append(E.s("polyline", { points: ball(q), "data-k": k, class: "ew-edge", style: "stroke-width:1.4" }));
      });
      // 紺の破線：選んだ p の単位球
      svg.append(E.s("polyline", { points: ball(p), "data-k": "np", class: "ew-curve-d" }));
      // 点
      svg.append(E.s("circle", { cx: px(x1), cy: py(x2), r: 5, class: "ew-dot-f", "data-k": "pt" }));

      // 凡例
      svg.append(E.s("text", { x: 16, y: 30, class: "ew-s ew-start" }, "灰：L1（ひし形）⊂ L2（円）⊂ L∞（正方形）"));
      svg.append(E.s("line", { x1: 16, y1: 52, x2: 46, y2: 52, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: 54, y: 52, class: "ew-s ew-start" }, "選んだ p の単位球"));

      const v = [x1, x2];
      const n1 = E.math.pnorm(v, 1), n2 = E.math.pnorm(v, 2), ni = E.math.pnorm(v, Infinity), np = E.math.pnorm(v, p);
      E.formulaRows(rows, svg, [
        { tex: "\\|\\mathbf{x}\\|_1=|x_1|+|x_2|", num: "=" + f(Math.abs(x1), 2) + "+" + f(Math.abs(x2), 2) + "=" + f(n1, 3), hl: ["n1", "pt"] },
        { tex: "\\|\\mathbf{x}\\|_2=\\sqrt{x_1^2+x_2^2}", num: "=" + f(n2, 3), hl: ["n2", "pt"] },
        { tex: "\\|\\mathbf{x}\\|_\\infty=\\max(|x_1|,|x_2|)", num: "=" + f(ni, 3), hl: ["ni", "pt"] },
        { tex: "\\|\\mathbf{x}\\|_p=\\bigl(|x_1|^p+|x_2|^p\\bigr)^{1/p}", num: "p=" + f(p, 1) + ":\\ " + f(np, 3), hl: ["np", "pt"] },
      ]);
    }
    draw();
  });
})();
