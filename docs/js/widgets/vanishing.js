// 勾配消失：活性化関数の微分を層の数だけ掛けたときの大きさ（対数目盛）
(function () {
  const E = window.ENote;

  const ACTS = [
    {
      k: "bs", name: "シグモイド",
      d: (u) => { const s = 1 / (1 + Math.exp(-u)); return s * (1 - s); },
      tex: "\\sigma'(u)",
    },
    {
      k: "bt", name: "tanh",
      d: (u) => 1 - Math.tanh(u) ** 2,
      tex: "\\tanh'(u)",
    },
    {
      k: "br", name: "ReLU",
      d: (u) => (u > 0 ? 1 : 0),
      tex: "\\mathrm{ReLU}'(u)",
    },
  ];
  const MIN_EXP = -20; // 目盛の左端 10^-20
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c]).join("");

  E.register("vanishing", (el) => {
    const svg = E.svg(el, 640, 210, "活性化関数の微分を層の数だけ掛けた値を、対数目盛の棒グラフで比べる図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "n", label: "層の数\\ n", min: 1, max: 30, step: 1, value: 5, digits: 0 },
        { key: "u", label: "u", min: -4, max: 4, step: 0.1, value: 0.5 },
      ],
      () => draw()
    );

    const X0 = 120, W = 400;
    // 目盛
    [0, -5, -10, -15, -20].forEach((e) => {
      const x = X0 + W * (1 - e / MIN_EXP);
      svg.append(E.s("line", { x1: x, y1: 20, x2: x, y2: 150, class: "ew-grid" }));
      svg.append(E.s("text", { x, y: 168, class: "ew-s ew-c ew-muted" }, e === 0 ? "1" : "10" + sup(e)));
    });
    const bars = ACTS.map((a, i) => {
      const y = 30 + i * 40;
      svg.append(E.s("text", { x: X0 - 12, y: y + 12, class: "ew-t ew-end" }, a.name));
      const rect = E.s("rect", { "data-k": a.k, x: X0, y, width: 0, height: 24, rx: 3, class: "ew-bar-" + i });
      const label = E.s("text", { x: X0, y: y + 12, class: "ew-s ew-start" });
      svg.append(rect, label);
      return { rect, label };
    });
    svg.append(E.s("text", { x: 320, y: 196, class: "ew-s ew-c ew-muted" }, "重みはすべて 1、どの層も同じ u と仮定した場合の大きさ"));

    function draw() {
      const { n, u } = sl.get();
      const f = E.fmt;
      const r = [];
      ACTS.forEach((a, i) => {
        const d = a.d(u);
        const p = Math.pow(d, n);
        const e = p > 0 ? Math.max(Math.log10(p), MIN_EXP) : MIN_EXP;
        const w = W * (1 - e / MIN_EXP);
        bars[i].rect.setAttribute("width", Math.max(w, 0));
        bars[i].label.setAttribute("x", X0 + Math.max(w, 0) + 8);
        bars[i].label.textContent = p === 0 ? "0" : p >= 0.001 ? f(p) : p.toExponential(1).replace(/e([+-]?\d+)/, (m, e) => " ×10" + sup(+e));
        r.push({
          tex: "\\bigl(" + a.tex + "\\bigr)^{" + n + "}",
          num: "=" + f(d) + "^{" + n + "}=" + E.fmtSci(p),
          hl: [a.k],
        });
      });
      E.formulaRows(rows, svg, r);
    }

    draw();
  });
})();
