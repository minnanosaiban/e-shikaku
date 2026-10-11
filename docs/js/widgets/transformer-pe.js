// 位置エンコーディング：次元の組 i ごとに、波長の違う sin / cos を位置 pos で読む
(function () {
  const E = window.ENote;
  const DM = 64; // d_model（図では固定）
  const NPOS = 128;
  const PX0 = 56, PW = 560, PY = 100, AMP = 66;

  E.register("transformer-pe", (el) => {
    const svg = E.svg(el, 640, 236, "位置に対する位置エンコーディングの sin と cos の波");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "pos", label: "pos", min: 0, max: NPOS - 1, step: 1, value: 20, digits: 0 },
        { key: "i", label: "i", min: 0, max: DM / 2 - 1, step: 1, value: 4, digits: 0 },
      ],
      () => draw()
    );
    const xOf = (p) => PX0 + (PW * p) / (NPOS - 1);
    const yOf = (v) => PY - AMP * v;

    function draw() {
      const { pos, i } = sl.get();
      const f = E.fmt;
      const w = Math.pow(10000, -(2 * i) / DM); // 角周波数 1/10000^{2i/d_model}
      const lam = (2 * Math.PI) / w;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [1, 0, -1].forEach((v) => {
        svg.append(E.s("line", { x1: PX0, y1: yOf(v), x2: PX0 + PW, y2: yOf(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: PX0 - 8, y: yOf(v) + 4, class: "ew-s ew-end ew-muted" }, String(v)));
      });
      [0, 32, 64, 96, 127].forEach((p) => {
        svg.append(E.s("text", { x: xOf(p), y: 190, class: "ew-s ew-c ew-muted" }, String(p)));
      });
      svg.append(E.s("text", { x: PX0 + PW / 2, y: 212, class: "ew-s ew-c ew-muted" }, "位置 pos（d_model = 64 の場合。次元 2i が sin、2i+1 が cos）"));

      const pathOf = (fn) => {
        let d = "";
        for (let k = 0; k <= NPOS - 1; k += 0.25) {
          d += (k === 0 ? "M" : "L") + f(xOf(k), 2) + " " + f(yOf(fn(w * k)), 2);
        }
        return d;
      };
      svg.append(E.s("path", { "data-k": "sin", d: pathOf(Math.sin), class: "ew-curve-f" }));
      svg.append(E.s("path", { "data-k": "cos", d: pathOf(Math.cos), class: "ew-curve-d" }));
      svg.append(E.s("line", { x1: xOf(pos), y1: yOf(1), x2: xOf(pos), y2: yOf(-1), class: "ew-grid" }));
      svg.append(E.s("circle", { "data-k": "sin", cx: xOf(pos), cy: yOf(Math.sin(w * pos)), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "cos", cx: xOf(pos), cy: yOf(Math.cos(w * pos)), r: 4.5, class: "ew-dot-d" }));

      E.formulaRows(rows, svg, [
        {
          tex: "\\omega_i=\\dfrac{1}{10000^{2i/d_{\\mathrm{model}}}}",
          num: "=10000^{-" + f((2 * i) / DM, 3) + "}=" + E.fmtSci(w) + ",\\quad \\text{波長}\\ \\tfrac{2\\pi}{\\omega_i}=" + f(lam, 1),
          hl: ["sin", "cos"],
        },
        {
          tex: "\\mathrm{PE}_{(pos,\\,2i)}=\\sin(\\omega_i\\,pos)",
          num: "=\\sin(" + f(w * pos, 3) + ")=" + f(Math.sin(w * pos)),
          hl: ["sin"],
        },
        {
          tex: "\\mathrm{PE}_{(pos,\\,2i+1)}=\\cos(\\omega_i\\,pos)",
          num: "=\\cos(" + f(w * pos, 3) + ")=" + f(Math.cos(w * pos)),
          hl: ["cos"],
        },
      ]);
    }
    draw();
  });
})();
