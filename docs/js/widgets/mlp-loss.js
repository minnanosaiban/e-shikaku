// 出力層のデルタ：シグモイド出力で、MSE と交差エントロピーの δ = ∂L/∂u を比べる
(function () {
  const E = window.ENote;

  const sig = (u) => 1 / (1 + Math.exp(-u));
  const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));

  // 描画範囲：u ∈ [-6, 6]、δ ∈ [-1.1, 1.1]
  const X0 = 70, X1 = 590, Y0 = 20, Y1 = 250;
  const sx = (u) => X0 + ((u + 6) / 12) * (X1 - X0);
  const sy = (v) => Y1 - ((v + 1.1) / 2.2) * (Y1 - Y0);

  const dMse = (u, t) => { const y = sig(u); return (y - t) * y * (1 - y); };
  const dBce = (u, t) => sig(u) - t;

  E.register("mlp-loss", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      [{ value: 1, label: "正解 t = 1" }, { value: 0, label: "正解 t = 0" }],
      1,
      () => draw()
    );
    const svg = E.svg(el, 640, 300, "シグモイド出力で、平均二乗誤差と交差エントロピーのデルタを総入力 u に対して比べるグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [{ key: "u", label: "u", min: -6, max: 6, step: 0.1, value: -3 }],
      () => draw()
    );

    function draw() {
      const t = toggle.get();
      const { u } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 目盛と軸
      [-6, -3, 0, 3, 6].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      [-1, -0.5, 0, 0.5, 1].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
      });
      svg.append(E.s("line", { x1: sx(0), y1: Y0, x2: sx(0), y2: Y1, class: "ew-edge" }));
      svg.append(E.s("line", { x1: X0, y1: sy(0), x2: X1, y2: sy(0), class: "ew-edge" }));
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: Y1 + 38, class: "ew-s ew-c" }, "u（出力層の総入力）"));

      // 曲線
      const pts = (g) => {
        const out = [];
        for (let i = 0; i <= 120; i++) {
          const v = -6 + i * 0.1;
          out.push(sx(v).toFixed(1) + "," + sy(g(v, t)).toFixed(1));
        }
        return out.join(" ");
      };
      svg.append(E.s("polyline", { "data-k": "cb", points: pts(dBce), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "cm", points: pts(dMse), class: "ew-curve-d" }));

      // u の位置
      svg.append(E.s("line", { x1: sx(u), y1: Y0, x2: sx(u), y2: Y1, class: "ew-grid", "stroke-dasharray": "3 3" }));
      svg.append(E.s("circle", { "data-k": "pb", cx: sx(u), cy: sy(dBce(u, t)), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "pm", cx: sx(u), cy: sy(dMse(u, t)), r: 4.5, class: "ew-dot-d" }));

      // 凡例（左上は、どちらの t でも曲線が通らない）
      svg.append(E.s("line", { x1: X0 + 14, y1: 36, x2: X0 + 44, y2: 36, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: X0 + 52, y: 36, class: "ew-s ew-start" }, "δ（交差エントロピー）"));
      svg.append(E.s("line", { x1: X0 + 14, y1: 56, x2: X0 + 44, y2: 56, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: X0 + 52, y: 56, class: "ew-s ew-start" }, "δ（平均二乗誤差）"));

      const y = sig(u);
      const lm = 0.5 * (y - t) ** 2;
      const lb = t === 1 ? softplus(-u) : softplus(u);
      E.formulaRows(rows, svg, [
        { tex: "y=\\sigma(u)", num: "=" + f(y), hl: [] },
        { tex: "L_{\\mathrm{MSE}}=\\tfrac12(y-t)^2", num: "=" + f(lm), hl: [] },
        { tex: "L_{\\mathrm{CE}}=-\\bigl[t\\ln y+(1-t)\\ln(1-y)\\bigr]", num: "=" + f(lb), hl: [] },
        {
          tex: "\\delta_{\\mathrm{MSE}}=(y-t)\\,y(1-y)",
          num: "=" + f(y - t) + "\\times" + f(y * (1 - y)) + "=" + f(dMse(u, t)),
          hl: ["cm", "pm"],
        },
        { tex: "\\delta_{\\mathrm{CE}}=y-t", num: "=" + f(dBce(u, t)), hl: ["cb", "pb"] },
      ]);
    }

    draw();
  });
})();
