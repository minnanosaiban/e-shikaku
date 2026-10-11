// ドロップアウト：マスクごとに出力はばらつくが、平均は「推論時に q を掛けた出力」に一致する
(function () {
  const E = window.ENote;

  const Y = [0.9, 0.4, 1.3, 0.7, 1.1, 0.2, 0.8, 1.0];     // 前の層の出力 y_j（固定）
  const W = [0.5, -0.3, 0.8, 0.6, -0.4, 0.9, 0.2, 0.7];   // 重み w_j（固定）
  const C = Y.map((y, j) => y * W[j]);                      // 各ユニットの寄与 w_j y_j
  const TOTAL = C.reduce((s, v) => s + v, 0);
  const NS = 2000, D = Y.length;

  // 再現できる乱数（mulberry32）
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // 2000 回ぶんの一様乱数（p を動かしても乱数は同じ）
  const R = rng(777);
  const UNI = Array.from({ length: NS }, () => Array.from({ length: D }, R));
  const draws = (q) => UNI.map((row) => row.map((v) => (v < q ? 1 : 0)));
  const outputOf = (mask) => mask.reduce((s, r, j) => s + r * C[j], 0);
  E._dropoutStats = (p) => {
    const us = draws(1 - p).map(outputOf);
    return { mean: us.reduce((s, v) => s + v, 0) / NS, test: (1 - p) * TOTAL };
  };

  // 出力の範囲 [-1, 3.2]
  const X0 = 60, X1 = 580, HY0 = 120, HY1 = 235;
  const UMIN = -1, UMAX = 3.2, NB = 42;
  const hx = (u) => X0 + ((u - UMIN) / (UMAX - UMIN)) * (X1 - X0);

  E.register("reg-dropout", (el) => {
    let draw_i = 0; // 今のマスクの番号
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    top.append(E.h("button", { type: "button", class: "ew-btn", onclick: () => { draw_i++; draw(); } }, "マスクを引き直す"));
    const svg = E.svg(el, 640, 295, "ドロップアウトのマスクと、その出力の分布および推論時の出力を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [{ key: "p", label: "p", min: 0, max: 0.9, step: 0.05, value: 0.5 }],
      () => draw()
    );

    function draw() {
      const { p } = sl.get();
      const q = 1 - p;
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 今のマスク
      const g = rng(1000 + draw_i);
      const mask = Y.map(() => (g() < q ? 1 : 0));
      const uCur = outputOf(mask);

      // 上段：ユニット
      Y.forEach((y, j) => {
        const x = 90 + j * 65;
        const at = { "data-k": "unit", cx: x, cy: 40, r: 15, class: mask[j] ? "ew-bar-0" : "ew-node" };
        if (!mask[j]) at["stroke-dasharray"] = "3 3";
        svg.append(E.s("circle", at));
        svg.append(E.s("text", { x, y: 40, class: mask[j] ? "ew-t ew-c" : "ew-s ew-c ew-muted" }, mask[j] ? "1" : "0"));
        svg.append(E.s("text", { x, y: 70, class: "ew-s ew-c ew-muted" }, f(C[j], 2)));
      });
      svg.append(E.s("text", { x: 10, y: 98, class: "ew-s ew-start" }, "円の下：各ユニットの寄与 w·y　下の図：2000 回のマスクでの出力 u の分布"));

      // 下段：u の分布
      const us = draws(q).map(outputOf);
      const cnt = new Array(NB).fill(0);
      us.forEach((u) => {
        const b = Math.floor(((u - UMIN) / (UMAX - UMIN)) * NB);
        cnt[Math.max(0, Math.min(NB - 1, b))]++;
      });
      const cmax = Math.max(...cnt);
      const bw = (X1 - X0) / NB;
      svg.append(E.s("line", { x1: X0, y1: HY1, x2: X1, y2: HY1, class: "ew-edge" }));
      cnt.forEach((c, i) => {
        const h = (c / cmax) * (HY1 - HY0 - 8);
        svg.append(E.s("rect", { x: X0 + i * bw + 0.5, y: HY1 - h, width: bw - 1, height: h, class: "ew-bar-0" }));
      });
      [-1, 0, 1, 2, 3].forEach((v) => {
        svg.append(E.s("text", { x: hx(v), y: HY1 + 14, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      const mean = us.reduce((s, v) => s + v, 0) / NS;
      const test = q * TOTAL;
      svg.append(E.s("line", { x1: hx(mean), y1: HY0, x2: hx(mean), y2: HY1, class: "ew-edge", "stroke-dasharray": "4 3" }));
      svg.append(E.s("line", { "data-k": "test", x1: hx(test), y1: HY0, x2: hx(test), y2: HY1, style: "stroke:var(--main);stroke-width:1.6" }));
      svg.append(E.s("circle", { "data-k": "cur", cx: hx(uCur), cy: HY1, r: 5, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: X0, y: 277, class: "ew-s ew-start" }, "紺の縦線：推論時の出力　灰の点線：訓練時の平均　紺の点：今のマスク"));

      E.formulaRows(rows, svg, [
        { tex: "r_j\\sim\\mathrm{Bernoulli}(q),\\ \\ q=1-p", num: "q=" + f(q) + ",\\ r=(" + mask.join(",") + ")", hl: ["unit"] },
        { tex: "u=\\sum_j w_j\\,y_j\\,r_j\\quad\\text{（訓練時）}", num: "=" + f(uCur), hl: ["cur"] },
        { tex: "u_{\\mathrm{test}}=q\\sum_j w_j\\,y_j\\quad\\text{（推論時）}", num: "=" + f(q) + "\\times" + f(TOTAL) + "=" + f(test), hl: ["test"] },
        { tex: "\\mathbb{E}[u]=q\\sum_j w_j\\,y_j", num: "\\text{平均}\\ " + f(mean) + "\\ \\approx\\ " + f(test), hl: ["test"] },
      ]);
    }

    draw();
  });
})();
