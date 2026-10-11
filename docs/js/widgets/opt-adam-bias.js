// Adam のバイアス補正：勾配が一定 (g = 1) のとき、補正の有無で 1 ステップの大きさがどう違うか
(function () {
  const E = window.ENote;

  // g = 1 を入れ続けたとき、m_t = 1-β1^t、v_t = 1-β2^t
  const raw = (t, b1, b2) => (1 - Math.pow(b1, t)) / Math.sqrt(1 - Math.pow(b2, t));

  const X0 = 70, X1 = 590, Y0 = 20, Y1 = 230, TMAX = 100;
  const sx = (t) => X0 + (t / TMAX) * (X1 - X0);

  E.register("opt-adam-bias", (el) => {
    const svg = E.svg(el, 640, 290, "勾配が一定のとき、Adam のバイアス補正がある場合とない場合で、1ステップの大きさを比べるグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "b1", label: "\\beta_1", min: 0.5, max: 0.99, step: 0.01, value: 0.9 },
        { key: "b2", label: "\\beta_2", min: 0.9, max: 0.9999, step: 0.0001, value: 0.999, digits: 4 },
        { key: "t", label: "t", min: 1, max: TMAX, step: 1, value: 3, digits: 0 },
      ],
      () => draw()
    );

    function draw() {
      const { b1, b2, t } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      let maxR = 1;
      for (let i = 1; i <= TMAX; i++) maxR = Math.max(maxR, raw(i, b1, b2));
      const ymax = Math.max(2, Math.ceil(maxR));
      const sy = (v) => Y1 - (Math.min(v, ymax) / ymax) * (Y1 - Y0);

      // 目盛と軸
      [0, 25, 50, 75, 100].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      [0, ymax / 2, ymax].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, f(v, 1)));
      });
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-edge" }));
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: Y1 + 38, class: "ew-s ew-c" }, "ステップ数 t（勾配は常に g = 1 とした場合）"));

      // 曲線：補正なし（実線）と補正あり（紺の破線、常に 1）
      const pts = [];
      for (let i = 1; i <= TMAX; i++) pts.push(sx(i).toFixed(1) + "," + sy(raw(i, b1, b2)).toFixed(1));
      svg.append(E.s("polyline", { "data-k": "cr", points: pts.join(" "), class: "ew-curve-f" }));
      svg.append(E.s("line", { "data-k": "cc", x1: sx(1), y1: sy(1), x2: sx(TMAX), y2: sy(1), class: "ew-curve-d" }));

      // t の位置
      const rv = raw(t, b1, b2);
      svg.append(E.s("line", { x1: sx(t), y1: Y0, x2: sx(t), y2: Y1, class: "ew-grid", "stroke-dasharray": "3 3" }));
      svg.append(E.s("circle", { "data-k": "pr", cx: sx(t), cy: sy(rv), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "pc", cx: sx(t), cy: sy(1), r: 4.5, class: "ew-dot-d" }));

      // 凡例（右上。曲線は t が大きいと 1 に近づくので、上側は空く）
      svg.append(E.s("line", { x1: X1 - 250, y1: 36, x2: X1 - 220, y2: 36, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: X1 - 212, y: 36, class: "ew-s ew-start" }, "補正なし  m / (√v + ε)"));
      svg.append(E.s("line", { x1: X1 - 250, y1: 56, x2: X1 - 220, y2: 56, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: X1 - 212, y: 56, class: "ew-s ew-start" }, "補正あり（m, v を 1−β^t で割る）"));

      const mt = 1 - Math.pow(b1, t), vt = 1 - Math.pow(b2, t);
      E.formulaRows(rows, svg, [
        {
          tex: "m_t=(1-\\beta_1^{\\,t})\\,g,\\quad v_t=(1-\\beta_2^{\\,t})\\,g^2",
          num: "m_t=" + f(mt) + ",\\ v_t=" + f(vt, 4),
          hl: [],
        },
        {
          tex: "\\hat{m}_t=\\dfrac{m_t}{1-\\beta_1^{\\,t}},\\quad \\hat{v}_t=\\dfrac{v_t}{1-\\beta_2^{\\,t}}",
          num: "\\hat{m}_t=1,\\ \\hat{v}_t=1",
          hl: ["cc", "pc"],
        },
        {
          tex: "\\dfrac{m_t}{\\sqrt{v_t}}=\\dfrac{1-\\beta_1^{\\,t}}{\\sqrt{1-\\beta_2^{\\,t}}}",
          num: "=\\dfrac{" + f(mt) + "}{" + f(Math.sqrt(vt)) + "}=" + f(rv),
          hl: ["cr", "pr"],
        },
      ]);
    }

    draw();
  });
})();
