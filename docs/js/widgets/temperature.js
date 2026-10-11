// 温度つきソフトマックス：温度 T で確率の鋭さが変わる
(function () {
  const E = window.ENote;
  const BX = 130, BW = 380;

  E.register("temperature", (el) => {
    const svg = E.svg(el, 640, 170, "3つのクラスのロジットから、温度つきソフトマックスで求めた確率を示す棒グラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "z1", label: "z_1", min: -3, max: 3, step: 0.1, value: 2 },
        { key: "z2", label: "z_2", min: -3, max: 3, step: 0.1, value: 1 },
        { key: "z3", label: "z_3", min: -3, max: 3, step: 0.1, value: 0.1 },
        { key: "T", label: "T", min: 0.1, max: 5, step: 0.1, value: 1, digits: 1 },
      ],
      () => draw()
    );

    function draw() {
      const { z1, z2, z3, T } = sl.get();
      const z = [z1, z2, z3];
      const f = E.fmt;
      // 最大値を引いてから指数をとる（T が小さいときのオーバーフロー対策。結果は同じ）
      const zm = Math.max(...z);
      const ex = z.map((v) => Math.exp((v - zm) / T));
      const sum = ex.reduce((s, v) => s + v, 0);
      const p = ex.map((v) => v / sum);

      while (svg.childNodes.length > 2) svg.lastChild.remove();
      [0, 0.5, 1].forEach((v) => {
        const x = BX + BW * v;
        svg.append(E.s("line", { x1: x, y1: 14, x2: x, y2: 124, class: "ew-grid" }));
        svg.append(E.s("text", { x, y: 142, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      p.forEach((v, i) => {
        const y = 22 + i * 34;
        svg.append(E.s("text", { x: BX - 12, y: y + 12, class: "ew-t ew-end" }, "z" + "₁₂₃"[i] + " = " + f(z[i], 1)));
        svg.append(E.s("rect", { "data-k": "b" + i, x: BX, y, width: Math.max(BW * v, 1), height: 24, rx: 3, class: "ew-bar-0" }));
        svg.append(E.s("text", { x: BX + BW * v + 8, y: y + 12, class: "ew-s ew-start" }, f(v)));
      });

      const r = p.map((v, i) => ({
        tex: "y_" + (i + 1) + "=\\frac{e^{z_" + (i + 1) + "/T}}{\\sum_j e^{z_j/T}}",
        num: "z_" + (i + 1) + "/T=" + f(z[i] / T, 2) + ",\\quad y_" + (i + 1) + "=" + f(v),
        hl: ["b" + i],
      }));
      r.push({ tex: "\\sum_i y_i", num: "=" + f(p.reduce((s, v) => s + v, 0)), hl: ["b0", "b1", "b2"] });
      E.formulaRows(rows, svg, r);
    }

    draw();
  });
})();
