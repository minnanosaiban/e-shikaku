// k-NN：問い合わせ点の近傍 K 個の多数決と、K・距離による決定領域の違い
(function () {
  const E = window.ENote;

  // 2クラスの訓練データ（クラス0＝灰の輪、クラス1＝紺の点）
  const C0 = [[-3.6, 0.8], [-3.0, -0.9], [-2.4, 1.6], [-2.1, 0.1], [-1.7, -1.2], [-1.2, 1.1], [-0.9, -0.4],
    [-4.2, -0.3], [-2.8, 2.1], [-1.4, -2.0], [0.3, 1.5], [-0.2, -1.6], [-3.3, -2.0]];
  const C1 = [[3.4, 0.6], [2.9, -1.0], [2.2, 1.3], [1.9, -0.2], [1.4, -1.5], [0.9, 0.7], [0.6, -0.6],
    [4.0, -0.2], [2.6, 2.0], [1.2, 2.0], [-0.5, 0.5], [3.8, 1.9], [2.7, -2.1], [0.4, -2.0]];
  const PTS = C0.map((p) => ({ x: p[0], y: p[1], c: 0 })).concat(C1.map((p) => ({ x: p[0], y: p[1], c: 1 })));

  const METRICS = {
    l2: { f: (dx, dy) => Math.hypot(dx, dy), unit: (c, s) => 1 / Math.hypot(c, s), tex: "\\|\\mathbf{x}-\\mathbf{x}_n\\|_2" },
    l1: { f: (dx, dy) => Math.abs(dx) + Math.abs(dy), unit: (c, s) => 1 / (Math.abs(c) + Math.abs(s)), tex: "\\|\\mathbf{x}-\\mathbf{x}_n\\|_1" },
    linf: { f: (dx, dy) => Math.max(Math.abs(dx), Math.abs(dy)), unit: (c, s) => 1 / Math.max(Math.abs(c), Math.abs(s)), tex: "\\|\\mathbf{x}-\\mathbf{x}_n\\|_\\infty" },
  };

  // 近傍 K 個の多数決。同数のときは最も近い点のクラスにする
  function classify(qx, qy, K, metric) {
    const d = PTS.map((p, i) => ({ i, d: metric(p.x - qx, p.y - qy) })).sort((a, b) => a.d - b.d || a.i - b.i);
    const nb = d.slice(0, K);
    const cnt = [0, 0];
    nb.forEach((n) => cnt[PTS[n.i].c]++);
    const pred = cnt[1] > cnt[0] ? 1 : cnt[0] > cnt[1] ? 0 : PTS[nb[0].i].c;
    return { nb, cnt, pred, r: nb[K - 1].d };
  }

  if (typeof module !== "undefined") module.exports = { PTS, METRICS, classify };
  if (!E) return;

  const SC = 50, CX = 320, CY = 170;
  const sx = (x) => CX + SC * x;
  const sy = (y) => CY - SC * y;
  const CELL = 10;

  E.register("knn-vote", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "l2", label: "ユークリッド（L2）" }, { value: "l1", label: "マンハッタン（L1）" }, { value: "linf", label: "チェビシェフ（L∞）" }],
      "l2",
      () => draw()
    );
    const svg = E.svg(el, 640, 340, "2クラスの訓練データと問い合わせ点。近傍K個の多数決で背景の領域が決まる図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "K", label: "K", min: 1, max: 15, step: 1, value: 5, digits: 0 },
        { key: "qx", label: "x_1", min: -5, max: 5, step: 0.1, value: 0.3 },
        { key: "qy", label: "x_2", min: -2.5, max: 2.5, step: 0.1, value: 0.4 },
      ],
      () => draw()
    );
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "背景：その位置に問い合わせ点を置いたときの予測クラス（紺＝クラス1、灰＝クラス0）。○＝クラス0、●＝クラス1、＋＝問い合わせ点。"));

    // 背景の格子（位置は固定、色だけ更新する）
    const bg = [];
    const gbg = E.s("g", {});
    for (let ix = 0; ix < 60; ix++) {
      for (let iy = 0; iy < 34; iy++) {
        const x = -6 + (ix + 0.5) * (12 / 60), y = 3.4 - (iy + 0.5) * (6.8 / 34);
        const r = E.s("rect", { x: CX - 300 + ix * CELL, y: CY - 170 + iy * CELL, width: CELL, height: CELL, "shape-rendering": "crispEdges" });
        gbg.append(r);
        bg.push({ r, x, y });
      }
    }
    svg.append(gbg);
    // 訓練点
    PTS.forEach((p) => {
      if (p.c === 0) svg.append(E.s("circle", { cx: sx(p.x), cy: sy(p.y), r: 5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.6" }));
      else svg.append(E.s("circle", { cx: sx(p.x), cy: sy(p.y), r: 5, class: "ew-dot-d" }));
    });
    const dyn = E.s("g", {});
    svg.append(dyn);

    function draw() {
      const { K, qx, qy } = sl.get();
      const m = METRICS[mode.get()];
      const f = E.fmt;
      bg.forEach((c) => {
        const res = classify(c.x, c.y, K, m.f);
        c.r.setAttribute("style", res.pred === 1 ? "fill:var(--main);fill-opacity:.14" : "fill:var(--muted);fill-opacity:.14");
      });
      dyn.innerHTML = "";
      const res = classify(qx, qy, K, m.f);

      // K番目の近傍までの範囲（距離の単位球を拡大したもの）
      const pts = [];
      for (let i = 0; i < 120; i++) {
        const th = (i / 120) * 2 * Math.PI, c = Math.cos(th), s = Math.sin(th);
        const u = m.unit(c, s) * res.r;
        pts.push(sx(qx + u * c).toFixed(1) + "," + sy(qy + u * s).toFixed(1));
      }
      dyn.append(E.s("polygon", { "data-k": "ball", points: pts.join(" "), class: "ew-edge", style: "stroke:var(--main);stroke-width:1.2;stroke-dasharray:4 3" }));
      res.nb.forEach((n) => {
        const p = PTS[n.i];
        dyn.append(E.s("line", { "data-k": "nb", x1: sx(qx), y1: sy(qy), x2: sx(p.x), y2: sy(p.y), class: "ew-edge", style: "stroke:var(--main);stroke-width:1;stroke-opacity:.6" }));
        dyn.append(E.s("circle", { "data-k": "nb", cx: sx(p.x), cy: sy(p.y), r: 9, class: "ew-edge", style: "stroke:var(--main);stroke-width:1.8" }));
      });
      const a = 7;
      dyn.append(E.s("line", { "data-k": "q", x1: sx(qx) - a, y1: sy(qy), x2: sx(qx) + a, y2: sy(qy), class: "ew-curve-f", style: "stroke-width:2.4" }));
      dyn.append(E.s("line", { "data-k": "q", x1: sx(qx), y1: sy(qy) - a, x2: sx(qx), y2: sy(qy) + a, class: "ew-curve-f", style: "stroke-width:2.4" }));

      E.formulaRows(rows, svg, [
        { tex: "d(\\mathbf{x},\\mathbf{x}_n)=" + m.tex, num: "r_K=" + f(res.r), hl: ["ball"] },
        { tex: "\\mathcal{N}_K(\\mathbf{x})=\\{\\text{距離が小さい順に }K\\text{ 個}\\}", num: "K=" + K, hl: ["nb"] },
        { tex: "\\sum_{\\mathbf{x}_n\\in\\mathcal{N}_K}\\mathbb{1}[y_n=c]", num: "c=0:\\ " + res.cnt[0] + ",\\quad c=1:\\ " + res.cnt[1], hl: ["nb"] },
        { tex: "\\hat{y}=\\mathop{\\rm argmax}_c\\ \\sum\\mathbb{1}[y_n=c]", num: "\\hat{y}=" + res.pred, hl: ["q"] },
      ]);
    }

    draw();
  });
})();
