// 計算グラフの1ノード：上流の勾配 × 局所勾配 を下流へ渡す様子
(function () {
  const E = window.ENote;

  const TYPES = {
    add: {
      label: "加算", sym: "+", binary: true,
      f: (x, y) => x + y,
      fwd: "z=x+y",
      dx: () => 1, dy: () => 1,
      dxTex: "\\frac{\\partial L}{\\partial z}\\cdot 1",
      dyTex: "\\frac{\\partial L}{\\partial z}\\cdot 1",
    },
    mul: {
      label: "乗算", sym: "×", binary: true,
      f: (x, y) => x * y,
      fwd: "z=xy",
      dx: (x, y) => y, dy: (x) => x,
      dxTex: "\\frac{\\partial L}{\\partial z}\\cdot y",
      dyTex: "\\frac{\\partial L}{\\partial z}\\cdot x",
    },
    sigmoid: {
      label: "シグモイド", sym: "σ", binary: false,
      f: (x) => 1 / (1 + Math.exp(-x)),
      fwd: "z=\\sigma(x)",
      dx: (x, y, z) => z * (1 - z),
      dxTex: "\\frac{\\partial L}{\\partial z}\\cdot z(1-z)",
    },
    relu: {
      label: "ReLU", sym: "ReLU", binary: false,
      f: (x) => Math.max(0, x),
      fwd: "z=\\max(0,x)",
      dx: (x) => (x > 0 ? 1 : 0),
      dxTex: "\\frac{\\partial L}{\\partial z}\\cdot \\mathbb{1}[x>0]",
    },
  };

  E.register("gate", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      Object.entries(TYPES).map(([value, t]) => ({ value, label: t.label })),
      "mul",
      () => draw()
    );
    const svg = E.svg(el, 640, 210, "計算グラフの1ノードで、順伝播の値と逆伝播の勾配を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "x", label: "x", min: -3, max: 3, step: 0.1, value: 2 },
        { key: "y", label: "y", min: -3, max: 3, step: 0.1, value: -1.5 },
        { key: "g", label: "\\partial L/\\partial z", min: -2, max: 2, step: 0.1, value: 1 },
      ],
      () => draw()
    );

    function draw() {
      const t = TYPES[toggle.get()];
      const { x, y, g } = sl.get();
      const z = t.f(x, y);
      const dx = g * t.dx(x, y, z);
      const dy = t.binary ? g * t.dy(x, y, z) : null;
      const f = E.fmt;

      while (svg.childNodes.length > 2) svg.lastChild.remove(); // title と defs は残す
      const xy = t.binary ? 50 : 105; // 入力ノードの高さ（単項なら中央）

      // 辺（順伝播は左→右）
      const edge = (k, x1, y1, x2, y2) =>
        svg.append(E.s("line", { "data-k": k, x1, y1, x2, y2, class: "ew-edge", "marker-end": svg.arrow }));
      edge("ex", 172, t.binary ? 58 : 105, 264, t.binary ? 94 : 105);
      if (t.binary) edge("ey", 172, 152, 264, 116);
      edge("ez", 334, 105, 440, 105);
      edge("eL", 496, 105, 566, 105);

      // ノード
      const node = (k, cx, cy, r, label, cls) => {
        svg.append(E.s("circle", { "data-k": k, cx, cy, r, class: cls }));
        svg.append(E.s("text", { x: cx, y: cy, class: "ew-t ew-c" }, label));
      };
      node("nx", 150, xy, 22, "x", "ew-node-in");
      if (t.binary) node("ny", 150, 160, 22, "y", "ew-node-in");
      node("op", 300, 105, 34, t.sym, "ew-node");
      node("nz", 468, 105, 26, "z", "ew-node");
      svg.append(E.s("rect", { "data-k": "nL", x: 568, y: 85, width: 56, height: 40, rx: 6, class: "ew-node-loss" }));
      svg.append(E.s("text", { x: 596, y: 105, class: "ew-t ew-c" }, "L"));

      // 値（順伝播は上、逆伝播は下）
      const lab = (x0, y0, text, cls) =>
        svg.append(E.s("text", { x: x0, y: y0, class: "ew-s ew-c " + cls }, text));
      // 入力側の値は、入力ノードの左に順伝播・逆伝播の2行で置く
      const inLab = (cy, name, fv, bv) => {
        svg.append(E.s("text", { x: 116, y: cy - 9, class: "ew-s ew-end ew-fwd" }, name + " = " + f(fv, 2)));
        svg.append(E.s("text", { x: 116, y: cy + 9, class: "ew-s ew-end ew-back" }, "∂L/∂" + name + " = " + f(bv)));
      };
      inLab(xy, "x", x, dx);
      if (t.binary) inLab(160, "y", y, dy);
      lab(387, 92, "z = " + f(z), "ew-fwd");
      lab(387, 124, "∂L/∂z = " + f(g, 2), "ew-back");
      lab(300, 196, "上：順伝播の値　下：逆伝播で流れる勾配", "ew-muted");

      const r = [
        { tex: t.fwd, num: "=" + f(z), hl: ["ex", "ey", "op", "ez"] },
        {
          tex: "\\frac{\\partial L}{\\partial x}=" + t.dxTex,
          num: "=" + f(g, 2) + "\\times" + f(t.dx(x, y, z)) + "=" + f(dx),
          hl: ["eL", "ez", "op", "ex"],
        },
      ];
      if (t.binary)
        r.push({
          tex: "\\frac{\\partial L}{\\partial y}=" + t.dyTex,
          num: "=" + f(g, 2) + "\\times" + f(t.dy(x, y, z)) + "=" + f(dy),
          hl: ["eL", "ez", "op", "ey"],
        });
      E.formulaRows(rows, svg, r);
    }

    draw();
  });
})();
