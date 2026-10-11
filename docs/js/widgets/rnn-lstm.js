// LSTM と GRU のゲート：1次元（1ユニット）で、ゲートの開き具合と状態の更新を見る
(function () {
  const E = window.ENote;
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const X0 = 190, W = 270, VX = 490;

  E.register("rnn-lstm", (el) => {
    const bar = E.h("div", { class: "ew-bar" });
    el.append(bar);
    const mode = E.toggle(bar, [{ value: "lstm", label: "LSTM" }, { value: "gru", label: "GRU" }], "lstm", () => show());
    const svg = E.svg(el, 640, 345, "ゲートの開き具合（0から1）と、記憶セルや隠れ状態の更新を棒グラフで示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const wrapL = E.h("div");
    const wrapG = E.h("div");
    el.append(wrapL, wrapG);
    const slL = E.sliders(
      wrapL,
      [
        { key: "af", label: "a_f", min: -4, max: 4, step: 0.1, value: 2 },
        { key: "ai", label: "a_i", min: -4, max: 4, step: 0.1, value: 0 },
        { key: "ao", label: "a_o", min: -4, max: 4, step: 0.1, value: 1 },
        { key: "ac", label: "a_c", min: -4, max: 4, step: 0.1, value: 1 },
        { key: "c0", label: "c_{t-1}", min: -2, max: 2, step: 0.1, value: 1 },
      ],
      () => draw()
    );
    const slG = E.sliders(
      wrapG,
      [
        { key: "az", label: "a_z", min: -4, max: 4, step: 0.1, value: 0 },
        { key: "ar", label: "a_r", min: -4, max: 4, step: 0.1, value: 1 },
        { key: "ax", label: "a_x", min: -4, max: 4, step: 0.1, value: 1 },
        { key: "h0", label: "h_{t-1}", min: -1, max: 1, step: 0.05, value: 0.8 },
      ],
      () => draw()
    );

    function show() {
      wrapL.style.display = mode.get() === "lstm" ? "" : "none";
      wrapG.style.display = mode.get() === "gru" ? "" : "none";
      draw();
    }

    function draw() {
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      let gates, vals, vmax, r;
      if (mode.get() === "lstm") {
        const { af, ai, ao, ac, c0 } = slL.get();
        const F = sig(af), I = sig(ai), O = sig(ao), g = Math.tanh(ac);
        const c1 = F * c0 + I * g, h = O * Math.tanh(c1);
        gates = [["gf", "忘却ゲート f", F], ["gi", "入力ゲート i", I], ["go", "出力ゲート o", O]];
        vals = [
          ["v0", "c(t−1)", c0], ["v1", "f ⊙ c(t−1)", F * c0], ["v2", "候補 tanh(a_c)", g],
          ["v3", "i ⊙ 候補", I * g], ["v4", "c_t（新しい記憶）", c1], ["v5", "h_t（出力）", h],
        ];
        vmax = 3;
        r = [
          { tex: "f=\\sigma(a_f)", num: "=" + f(F), hl: ["gf"] },
          { tex: "i=\\sigma(a_i)", num: "=" + f(I), hl: ["gi"] },
          { tex: "o=\\sigma(a_o)", num: "=" + f(O), hl: ["go"] },
          { tex: "\\tilde c=\\tanh(a_c)", num: "=" + f(g), hl: ["v2"] },
          {
            tex: "c_t=f\\,c_{t-1}+i\\,\\tilde c",
            num: "=" + f(F) + "\\cdot" + f(c0) + "+" + f(I) + "\\cdot" + f(g) + "=" + f(c1),
            hl: ["gf", "gi", "v0", "v1", "v2", "v3", "v4"],
          },
          {
            tex: "h_t=o\\,\\tanh(c_t)",
            num: "=" + f(O) + "\\cdot" + f(Math.tanh(c1)) + "=" + f(h),
            hl: ["go", "v4", "v5"],
          },
        ];
      } else {
        const { az, ar, ax, h0 } = slG.get();
        const Z = sig(az), R = sig(ar), ht = Math.tanh(ax + R * h0);
        const h1 = Z * h0 + (1 - Z) * ht;
        gates = [["gz", "更新ゲート z", Z], ["gr", "リセットゲート r", R]];
        vals = [
          ["v0", "h(t−1)", h0], ["v1", "r ⊙ h(t−1)", R * h0], ["v2", "候補", ht],
          ["v3", "z ⊙ h(t−1)", Z * h0], ["v4", "(1−z) ⊙ 候補", (1 - Z) * ht], ["v5", "h_t", h1],
        ];
        vmax = 1.5;
        r = [
          { tex: "z=\\sigma(a_z)", num: "=" + f(Z), hl: ["gz"] },
          { tex: "r=\\sigma(a_r)", num: "=" + f(R), hl: ["gr"] },
          {
            tex: "\\tilde h=\\tanh(a_x+r\\,h_{t-1})",
            num: "=\\tanh(" + f(ax, 2) + "+" + f(R) + "\\cdot" + f(h0) + ")=" + f(ht),
            hl: ["gr", "v1", "v2"],
          },
          {
            tex: "h_t=z\\,h_{t-1}+(1-z)\\,\\tilde h",
            num: "=" + f(Z) + "\\cdot" + f(h0) + "+" + f(1 - Z) + "\\cdot" + f(ht) + "=" + f(h1),
            hl: ["gz", "v0", "v2", "v3", "v4", "v5"],
          },
        ];
      }

      // ゲート（0〜1）
      let y = 10;
      svg.append(E.s("text", { x: 16, y: y + 6, class: "ew-s ew-start ew-muted" }, "ゲート（0 = 閉じる、1 = 開く）"));
      y += 22;
      gates.forEach(([k, name, v]) => {
        svg.append(E.s("text", { x: X0 - 10, y: y + 11, class: "ew-t ew-end" }, name));
        svg.append(E.s("line", { x1: X0, y1: y - 2, x2: X0, y2: y + 26, class: "ew-grid" }));
        svg.append(E.s("line", { x1: X0 + W, y1: y - 2, x2: X0 + W, y2: y + 26, class: "ew-grid" }));
        svg.append(E.s("rect", { "data-k": k, x: X0, y, width: Math.max(W * v, 1), height: 22, rx: 3, class: "ew-bar-0" }));
        svg.append(E.s("text", { x: VX, y: y + 11, class: "ew-s ew-start" }, f(v)));
        y += 32;
      });

      // 状態（符号つき）
      y += 8;
      svg.append(E.s("text", { x: 16, y: y + 6, class: "ew-s ew-start ew-muted" }, "状態の更新（中央が 0）"));
      y += 22;
      const XC = X0 + W / 2, sc = W / 2 / vmax;
      svg.append(E.s("line", { x1: XC, y1: y - 4, x2: XC, y2: y - 4 + 6 * 30, class: "ew-grid" }));
      vals.forEach(([k, name, v]) => {
        svg.append(E.s("text", { x: X0 - 10, y: y + 11, class: "ew-t ew-end" }, name));
        const len = Math.max(Math.abs(v) * sc, 1);
        svg.append(E.s("rect", { "data-k": k, x: v >= 0 ? XC : XC - len, y, width: len, height: 22, rx: 3, class: "ew-bar-0" }));
        svg.append(E.s("text", { x: VX, y: y + 11, class: "ew-s ew-start" }, f(v)));
        y += 30;
      });
      E.formulaRows(rows, svg, r);
    }

    show();
  });
})();
