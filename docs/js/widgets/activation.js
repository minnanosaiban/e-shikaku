// 活性化関数：f(x) と f'(x) の曲線を重ね、x の位置の値を数式の行で追う
(function () {
  const E = window.ENote;

  // 標準正規分布の累積分布関数 Φ（erf は Abramowitz-Stegun 7.1.26、誤差 1.5e-7）
  const erf = (v) => {
    const s = Math.sign(v), a = Math.abs(v);
    const t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  };
  const Phi = (v) => 0.5 * (1 + erf(v / Math.SQRT2));
  const phi = (v) => Math.exp(-v * v / 2) / Math.sqrt(2 * Math.PI);
  const sig = (v) => 1 / (1 + Math.exp(-v));

  const FUNCS = {
    sigmoid: {
      label: "シグモイド",
      f: (x) => sig(x),
      d: (x) => sig(x) * (1 - sig(x)),
      texF: "\\frac{1}{1+e^{-x}}",
      texD: "f(x)\\bigl(1-f(x)\\bigr)",
    },
    tanh: {
      label: "tanh",
      f: (x) => Math.tanh(x),
      d: (x) => 1 - Math.tanh(x) ** 2,
      texF: "\\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}",
      texD: "1-f(x)^2",
    },
    relu: {
      label: "ReLU",
      f: (x) => Math.max(0, x),
      d: (x) => (x > 0 ? 1 : 0),
      texF: "\\max(0,\\,x)",
      texD: "\\begin{cases}1&(x>0)\\\\0&(x\\le0)\\end{cases}",
    },
    leaky: {
      label: "Leaky ReLU",
      f: (x, a) => (x > 0 ? x : a * x),
      d: (x, a) => (x > 0 ? 1 : a),
      texF: "\\max(\\alpha x,\\,x)",
      texD: "\\begin{cases}1&(x>0)\\\\\\alpha&(x\\le0)\\end{cases}",
    },
    elu: {
      label: "ELU",
      f: (x, a) => (x >= 0 ? x : a * (Math.exp(x) - 1)),
      d: (x, a) => (x >= 0 ? 1 : a * Math.exp(x)),
      texF: "\\begin{cases}x&(x\\ge0)\\\\\\alpha(e^{x}-1)&(x<0)\\end{cases}",
      texD: "\\begin{cases}1&(x\\ge0)\\\\f(x)+\\alpha&(x<0)\\end{cases}",
    },
    gelu: {
      label: "GELU",
      f: (x) => x * Phi(x),
      d: (x) => Phi(x) + x * phi(x),
      texF: "x\\,\\Phi(x)",
      texD: "\\Phi(x)+x\\,\\phi(x)",
    },
    swish: {
      label: "Swish",
      f: (x) => x * sig(x),
      d: (x) => sig(x) * (1 + x * (1 - sig(x))),
      texF: "x\\,\\sigma(x)",
      texD: "\\sigma(x)\\bigl(1+x(1-\\sigma(x))\\bigr)",
    },
  };

  // 描画範囲：x ∈ [-4, 4]、y ∈ [-2, 4]
  const X0 = 70, X1 = 590, Y0 = 20, Y1 = 250;
  const sx = (x) => X0 + ((x + 4) / 8) * (X1 - X0);
  const sy = (y) => Y1 - ((Math.min(4, Math.max(-2, y)) + 2) / 6) * (Y1 - Y0);

  E.register("activation", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      Object.entries(FUNCS).map(([value, t]) => ({ value, label: t.label })),
      "sigmoid",
      () => draw()
    );
    const svg = E.svg(el, 640, 280, "活性化関数 f(x) とその導関数 f'(x) の曲線を重ねて示すグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "x", label: "x", min: -4, max: 4, step: 0.1, value: 1 },
        { key: "a", label: "\\alpha", min: 0.01, max: 1, step: 0.01, value: 0.2 },
      ],
      () => draw()
    );

    function draw() {
      const fn = FUNCS[toggle.get()];
      const { x, a } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove(); // title と defs は残す

      // 目盛と軸
      [-4, -2, 0, 2, 4].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      [-2, 0, 2, 4].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
      });
      svg.append(E.s("line", { x1: sx(0), y1: Y0, x2: sx(0), y2: Y1, class: "ew-edge" }));
      svg.append(E.s("line", { x1: X0, y1: sy(0), x2: X1, y2: sy(0), class: "ew-edge" }));

      // 曲線
      const pts = (g) => {
        const out = [];
        for (let i = 0; i <= 160; i++) {
          const v = -4 + i * 0.05;
          out.push(sx(v).toFixed(1) + "," + sy(g(v, a)).toFixed(1));
        }
        return out.join(" ");
      };
      svg.append(E.s("polyline", { "data-k": "cf", points: pts(fn.f), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "cd", points: pts(fn.d), class: "ew-curve-d" }));

      // x の位置
      svg.append(E.s("line", { x1: sx(x), y1: Y0, x2: sx(x), y2: Y1, class: "ew-grid", "stroke-dasharray": "3 3" }));
      const fv = fn.f(x, a), dv = fn.d(x, a);
      svg.append(E.s("circle", { "data-k": "pf", cx: sx(x), cy: sy(fv), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "pd", cx: sx(x), cy: sy(dv), r: 4.5, class: "ew-dot-d" }));

      // 凡例（左上は、どの関数でも曲線が通らない）
      svg.append(E.s("line", { x1: X0 + 14, y1: 36, x2: X0 + 44, y2: 36, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: X0 + 52, y: 36, class: "ew-s ew-start" }, "f(x)"));
      svg.append(E.s("line", { x1: X0 + 14, y1: 56, x2: X0 + 44, y2: 56, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: X0 + 52, y: 56, class: "ew-s ew-start" }, "f′(x)"));

      const xs = f(x, 2);
      E.formulaRows(rows, svg, [
        { tex: "f(x)=" + fn.texF, num: "f(" + xs + ")=" + f(fv), hl: ["cf", "pf"] },
        { tex: "f'(x)=" + fn.texD, num: "f'(" + xs + ")=" + f(dv), hl: ["cd", "pd"] },
      ]);
    }

    draw();
  });
})();
