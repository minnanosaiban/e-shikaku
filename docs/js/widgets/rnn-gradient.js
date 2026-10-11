// BPTT で時間をさかのぼる勾配の大きさ：通常の RNN と LSTM の記憶セル経路（1次元で単純化、対数目盛）
(function () {
  const E = window.ENote;
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c]).join("");
  const X0 = 110, W = 480, MAXE = 20;
  const XC = X0 + W / 2;

  // TeX 用：大きい値も小さい値も 10 のべき乗で表す
  function sci(v) {
    if (v >= 0.001 && v < 1000) return E.fmt(v, 3);
    const [m, e] = v.toExponential(2).split("e");
    return m + "\\times 10^{" + Number(e) + "}";
  }
  // SVG の文字用
  function sciText(v) {
    if (v >= 0.001 && v < 1000) return E.fmt(v, 3);
    const [m, e] = v.toExponential(1).split("e");
    return m + "×10" + sup(Number(e));
  }

  E.register("rnn-gradient", (el) => {
    const svg = E.svg(el, 640, 190, "時間をさかのぼるたびに掛かる係数のT乗を、通常のRNNとLSTMの記憶セルで比べる対数目盛の図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "T", label: "T", min: 1, max: 50, step: 1, value: 20, digits: 0 },
        { key: "w", label: "|w|", min: 0.1, max: 2.5, step: 0.05, value: 0.9 },
        { key: "u", label: "u", min: -3, max: 3, step: 0.1, value: 0.5, digits: 1 },
        { key: "f", label: "f", min: 0.5, max: 1, step: 0.01, value: 0.95 },
      ],
      () => draw()
    );

    [-20, -10, 0, 10, 20].forEach((e) => {
      const x = XC + (W / 2) * (e / MAXE);
      svg.append(E.s("line", { x1: x, y1: 20, x2: x, y2: 120, class: "ew-grid" }));
      svg.append(E.s("text", { x, y: 138, class: "ew-s ew-c ew-muted" }, e === 0 ? "1" : "10" + sup(e)));
    });
    const NAMES = ["RNN", "LSTM"];
    const bars = NAMES.map((name, i) => {
      const y = 36 + i * 40;
      svg.append(E.s("text", { x: X0 - 12, y: y + 12, class: "ew-t ew-end" }, name));
      const rect = E.s("rect", { "data-k": "g" + i, x: XC, y, width: 0, height: 24, rx: 3, class: "ew-bar-" + i });
      const label = E.s("text", { x: XC, y: y + 12, class: "ew-s" });
      svg.append(rect, label);
      return { rect, label };
    });
    svg.append(E.s("text", { x: 320, y: 172, class: "ew-s ew-c ew-muted" }, "左へ伸びる＝勾配消失、右へ伸びる＝勾配爆発（中央が 1）"));

    function draw() {
      const { T, w, u, f } = sl.get();
      const d = 1 - Math.tanh(u) ** 2;
      const a = w * d; // RNN の1ステップあたりの係数 |w|·tanh'(u)
      const vals = [Math.pow(a, T), Math.pow(f, T)];
      vals.forEach((v, i) => {
        const e = Math.max(-MAXE, Math.min(MAXE, Math.log10(v)));
        const len = Math.max(((W / 2) * Math.abs(e)) / MAXE, 1);
        const left = e < 0;
        bars[i].rect.setAttribute("x", left ? XC - len : XC);
        bars[i].rect.setAttribute("width", len);
        // 値の文字は、棒の反対側（中央の脇）に置いて重ならないようにする
        bars[i].label.setAttribute("x", left ? XC + 8 : XC - 8);
        bars[i].label.setAttribute("class", "ew-s " + (left ? "ew-start" : "ew-end"));
        bars[i].label.textContent = sciText(v);
      });
      const fm = E.fmt;
      E.formulaRows(rows, svg, [
        {
          tex: "\\bigl(|w|\\,\\tanh'(u)\\bigr)^{T}",
          num: "=(" + fm(w, 2) + "\\times" + fm(d) + ")^{" + T + "}=" + fm(a) + "^{" + T + "}=" + sci(vals[0]),
          hl: ["g0"],
        },
        {
          tex: "f^{T}\\ \\ (\\text{記憶セル：}\\partial c_t/\\partial c_{t-1}=f_t)",
          num: "=" + fm(f, 2) + "^{" + T + "}=" + sci(vals[1]),
          hl: ["g1"],
        },
      ]);
    }

    draw();
  });
})();
