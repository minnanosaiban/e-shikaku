// アンサンブル：M 個の予測を平均したときの分散（相関 ρ で下限が決まる）
(function () {
  const E = window.ENote;
  const MMAX = 30;
  const PX0 = 70, PW = 530, PY0 = 28, PY1 = 168;

  E.register("ensemble-var", (el) => {
    const svg = E.svg(el, 640, 232, "モデル数 M に対する、平均予測の分散の比のグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "M", label: "M", min: 1, max: MMAX, step: 1, value: 5, digits: 0 },
        { key: "rho", label: "\\rho", min: 0, max: 1, step: 0.05, value: 0.3 },
      ],
      () => draw()
    );
    const xOf = (M) => PX0 + (PW * (M - 1)) / (MMAX - 1);
    const yOf = (v) => PY1 - (PY1 - PY0) * v;

    function draw() {
      const { M, rho } = sl.get();
      const f = E.fmt;
      const v = (k) => rho + (1 - rho) / k; // Var[平均]/σ²
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [0, 0.5, 1].forEach((t) => {
        svg.append(E.s("line", { x1: PX0, y1: yOf(t), x2: PX0 + PW, y2: yOf(t), class: "ew-grid" }));
        svg.append(E.s("text", { x: PX0 - 8, y: yOf(t) + 4, class: "ew-s ew-end ew-muted" }, String(t)));
      });
      [1, 10, 20, 30].forEach((k) => {
        svg.append(E.s("text", { x: xOf(k), y: 188, class: "ew-s ew-c ew-muted" }, String(k)));
      });
      svg.append(E.s("text", { x: PX0 + PW / 2, y: 214, class: "ew-s ew-c ew-muted" }, "モデル数 M（縦軸は平均予測の分散 ÷ 1個の分散 σ²）"));

      svg.append(E.s("line", { "data-k": "floor", x1: PX0, y1: yOf(rho), x2: PX0 + PW, y2: yOf(rho), class: "ew-curve-d" }));
      let d = "";
      for (let k = 1; k <= MMAX; k += 0.25) d += (k === 1 ? "M" : "L") + f(xOf(k), 2) + " " + f(yOf(v(k)), 2);
      svg.append(E.s("path", { "data-k": "curve", d, class: "ew-curve-f" }));
      svg.append(E.s("circle", { "data-k": "curve", cx: xOf(M), cy: yOf(v(M)), r: 4.5, class: "ew-dot-f" }));

      E.formulaRows(rows, svg, [
        {
          tex: "\\mathrm{Var}\\Bigl[\\tfrac1M\\textstyle\\sum_m f_m\\Bigr]=\\rho\\sigma^2+\\dfrac{1-\\rho}{M}\\sigma^2",
          num: "=(" + f(rho, 2) + "+" + f((1 - rho) / M, 3) + ")\\sigma^2=" + f(v(M)) + "\\,\\sigma^2",
          hl: ["curve"],
        },
        {
          tex: "M\\to\\infty",
          num: "\\to\\rho\\sigma^2=" + f(rho, 2) + "\\,\\sigma^2\\quad(\\text{相関が高いほど下がらない})",
          hl: ["floor"],
        },
      ]);
    }
    draw();
  });
})();
