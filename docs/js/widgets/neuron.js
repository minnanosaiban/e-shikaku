// 1つのニューロン（シグモイド＋二乗誤差）で、順伝播と逆伝播を追う
(function () {
  const E = window.ENote;

  E.register("neuron", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "f", label: "順伝播" }, { value: "b", label: "逆伝播" }],
      "f",
      () => draw()
    );
    const svg = E.svg(el, 640, 230, "入力 x1, x2 が重み w1, w2 でニューロンに入り、シグモイドを通って損失 L になる計算グラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "x1", label: "x_1", min: -2, max: 2, step: 0.1, value: 1 },
        { key: "x2", label: "x_2", min: -2, max: 2, step: 0.1, value: 0.5 },
        { key: "w1", label: "w_1", min: -2, max: 2, step: 0.1, value: 0.8 },
        { key: "w2", label: "w_2", min: -2, max: 2, step: 0.1, value: -0.4 },
        { key: "b", label: "b", min: -2, max: 2, step: 0.1, value: 0.1 },
        { key: "t", label: "t", min: 0, max: 1, step: 0.05, value: 1 },
      ],
      () => draw()
    );

    // 図の骨組み（値以外）は1回だけ描く
    const edge = (k, x1, y1, x2, y2) =>
      svg.append(E.s("line", { "data-k": k, x1, y1, x2, y2, class: "ew-edge", "marker-end": svg.arrow }));
    edge("e1", 92, 64, 264, 108);
    edge("e2", 92, 166, 264, 128);
    edge("e3", 332, 118, 412, 118);
    edge("e4", 488, 118, 554, 118);
    const node = (k, cx, cy, r, label, cls) => {
      svg.append(E.s("circle", { "data-k": k, cx, cy, r, class: cls }));
      svg.append(E.s("text", { x: cx, y: cy, class: "ew-t ew-c" }, label));
    };
    node("nx1", 70, 60, 22, "x₁", "ew-node-in");
    node("nx2", 70, 170, 22, "x₂", "ew-node-in");
    node("nz", 300, 118, 32, "z", "ew-node");
    node("na", 450, 118, 36, "a=σ(z)", "ew-node");
    svg.append(E.s("rect", { "data-k": "nL", x: 556, y: 96, width: 68, height: 44, rx: 6, class: "ew-node-loss" }));
    svg.append(E.s("text", { x: 590, y: 118, class: "ew-t ew-c" }, "L"));
    const val = (x, y) => {
      const t = E.s("text", { x, y, class: "ew-s ew-c" });
      svg.append(t);
      return t;
    };
    const v = {
      w1: val(160, 50), w2: val(160, 184),
      z: val(300, 172), a: val(450, 176), L: val(590, 160),
      x1: val(70, 100), x2: val(70, 210),
    };

    function draw() {
      const { x1, x2, w1, w2, b, t } = sl.get();
      const back = mode.get() === "b";
      const f = E.fmt;
      const z = w1 * x1 + w2 * x2 + b;
      const a = 1 / (1 + Math.exp(-z));
      const L = 0.5 * (a - t) ** 2;
      const dLa = a - t, dad = a * (1 - a), dz = dLa * dad, dw1 = dz * x1, dw2 = dz * x2;

      const set = (t, text, cls) => { t.textContent = text; t.setAttribute("class", "ew-s ew-c " + cls); };
      set(v.x1, "x₁ = " + f(x1, 2), "ew-fwd");
      set(v.x2, "x₂ = " + f(x2, 2), "ew-fwd");
      set(v.L, "L = " + f(L, 4), "ew-fwd");
      if (back) {
        set(v.w1, "∂L/∂w₁ = " + f(dw1), "ew-back");
        set(v.w2, "∂L/∂w₂ = " + f(dw2), "ew-back");
        set(v.z, "∂L/∂z = " + f(dz), "ew-back");
        set(v.a, "∂L/∂a = " + f(dLa), "ew-back");
      } else {
        set(v.w1, "w₁ = " + f(w1, 2), "ew-fwd");
        set(v.w2, "w₂ = " + f(w2, 2), "ew-fwd");
        set(v.z, "z = " + f(z), "ew-fwd");
        set(v.a, "a = " + f(a), "ew-fwd");
      }
      svg.classList.toggle("ew-backmode", back);

      const r = back
        ? [
            { tex: "\\frac{\\partial L}{\\partial a}=a-t", num: "=" + f(a) + "-" + f(t, 2) + "=" + f(dLa), hl: ["e4", "na", "nL"] },
            { tex: "\\frac{\\partial a}{\\partial z}=a(1-a)", num: "=" + f(a) + "\\times" + f(1 - a) + "=" + f(dad), hl: ["e3", "na"] },
            { tex: "\\frac{\\partial L}{\\partial z}=\\frac{\\partial L}{\\partial a}\\frac{\\partial a}{\\partial z}", num: "=" + f(dLa) + "\\times" + f(dad) + "=" + f(dz), hl: ["e3", "e4", "nz"] },
            { tex: "\\frac{\\partial L}{\\partial w_1}=\\frac{\\partial L}{\\partial z}\\,x_1", num: "=" + f(dz) + "\\times" + f(x1, 2) + "=" + f(dw1), hl: ["e1", "nx1"] },
            { tex: "\\frac{\\partial L}{\\partial w_2}=\\frac{\\partial L}{\\partial z}\\,x_2", num: "=" + f(dz) + "\\times" + f(x2, 2) + "=" + f(dw2), hl: ["e2", "nx2"] },
            { tex: "\\frac{\\partial L}{\\partial b}=\\frac{\\partial L}{\\partial z}", num: "=" + f(dz), hl: ["nz"] },
          ]
        : [
            { tex: "z=w_1x_1+w_2x_2+b", num: "=" + f(w1, 2) + "\\cdot" + f(x1, 2) + "+(" + f(w2, 2) + ")\\cdot" + f(x2, 2) + "+" + f(b, 2) + "=" + f(z), hl: ["e1", "e2", "nz"] },
            { tex: "a=\\sigma(z)=\\frac{1}{1+e^{-z}}", num: "=\\sigma(" + f(z) + ")=" + f(a), hl: ["e3", "na"] },
            { tex: "L=\\tfrac12(a-t)^2", num: "=\\tfrac12(" + f(a) + "-" + f(t, 2) + ")^2=" + f(L, 4), hl: ["e4", "nL"] },
          ];
      E.formulaRows(rows, svg, r);
    }

    draw();
  });
})();
