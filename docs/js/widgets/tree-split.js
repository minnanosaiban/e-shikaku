// 決定木：しきい値 τ で 1 次元のデータを 2 つに分けたときの不純度と、不純度減少量 ΔI
(function () {
  const E = window.ENote;

  const XS = [0.5, 1.1, 1.8, 2.4, 3.0, 3.9, 4.4, 5.2, 5.9, 6.7, 7.3, 8.1, 8.8, 9.5];
  const TS = [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1];

  const IMP = {
    gini: (p) => 1 - p * p - (1 - p) * (1 - p),
    ent: (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p)),
    err: (p) => 1 - Math.max(p, 1 - p),
  };
  // しきい値 τ で分けたときの {n, p(クラス1の割合), 不純度}（左：x<=τ、右：x>τ）
  function split(tau, kind) {
    const L = [], R = [];
    XS.forEach((x, i) => (x <= tau ? L : R).push(TS[i]));
    const stat = (a) => {
      const n = a.length, p = n ? a.reduce((s, v) => s + v, 0) / n : 0;
      return { n, p, I: n ? IMP[kind](p) : 0 };
    };
    const P = stat(TS), l = stat(L), r = stat(R);
    const dI = P.I - (l.n / P.n) * l.I - (r.n / P.n) * r.I;
    return { P, l, r, dI };
  }

  if (typeof module !== "undefined") module.exports = { XS, TS, IMP, split };
  if (!E) return;

  const X0 = 40, X1 = 600;
  const sx = (x) => X0 + (x / 10) * (X1 - X0);
  const NAMES = { gini: "ジニ係数", ent: "エントロピー", err: "誤分類率" };
  const TEX = {
    gini: "I_G=1-\\sum_c p_c^2",
    ent: "I_H=-\\sum_c p_c\\log_2 p_c",
    err: "I_E=1-\\max_c p_c",
  };

  E.register("tree-split", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "gini", label: "ジニ係数" }, { value: "ent", label: "エントロピー" }, { value: "err", label: "誤分類率" }],
      "gini",
      () => draw()
    );
    const svg = E.svg(el, 640, 300, "1次元のデータをしきい値で左右に分けたときの不純度減少量を、しきい値ごとに示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(el, [{ key: "tau", label: "\\tau", min: 0.2, max: 9.8, step: 0.05, value: 4.1 }], () => draw());
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "上：データ（●＝クラス1、○＝クラス0）としきい値 τ。下：τ を動かしたときの ΔI（○＝最大の位置）。ΔI が最大になる τ が、そのノードの分岐として選ばれる。"));

    const YA = 70;          // 上段のデータの y 位置
    const B0 = 250, BH = 106; // 下段のグラフ（y=B0 が ΔI=0、BH が高さ）

    function draw() {
      const kind = mode.get();
      const { tau } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 左右の領域（面の濃淡）
      svg.append(E.s("rect", { "data-k": "L", x: X0, y: 28, width: sx(tau) - X0, height: 84, class: "ew-node-loss", style: "stroke:none;fill-opacity:.8" }));
      svg.append(E.s("rect", { "data-k": "R", x: sx(tau), y: 28, width: X1 - sx(tau), height: 84, class: "ew-node", style: "stroke:none" }));
      svg.append(E.s("text", { x: X0 + 6, y: 40, class: "ew-s ew-start" }, "左（x ≤ τ）"));
      svg.append(E.s("text", { x: X1 - 6, y: 40, class: "ew-s ew-end" }, "右（x > τ）"));
      svg.append(E.s("line", { x1: X0, y1: YA + 6, x2: X1, y2: YA + 6, class: "ew-edge" }));
      [0, 2, 4, 6, 8, 10].forEach((v) => svg.append(E.s("text", { x: sx(v), y: 124, class: "ew-s ew-c ew-muted" }, String(v))));
      XS.forEach((x, i) => {
        if (TS[i] === 1) svg.append(E.s("circle", { cx: sx(x), cy: YA, r: 5.5, class: "ew-dot-d" }));
        else svg.append(E.s("circle", { cx: sx(x), cy: YA, r: 5.5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.6" }));
      });
      svg.append(E.s("line", { "data-k": "tau", x1: sx(tau), y1: 28, x2: sx(tau), y2: 112, class: "ew-edge", style: "stroke:var(--main);stroke-width:2" }));

      // ΔI(τ) の曲線（データの位置をまたぐところでだけ値が変わる階段）
      let best = { tau: 0, dI: -1 };
      const all = [];
      for (let t = 0.2; t <= 9.8001; t += 0.05) {
        const d = split(t, kind).dI;
        all.push([t, d]);
        if (d > best.dI + 1e-12) best = { tau: t, dI: d };
      }
      const top2 = Math.max(best.dI * 1.25, 0.05);
      const sy = (d) => B0 - (d / top2) * BH;
      [0, top2 / 2, top2].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 6, y: sy(v), class: "ew-s ew-end ew-muted" }, f(v, 2)));
      });
      [0, 2, 4, 6, 8, 10].forEach((v) => svg.append(E.s("text", { x: sx(v), y: B0 + 16, class: "ew-s ew-c ew-muted" }, String(v))));
      svg.append(E.s("text", { x: X1, y: B0 + 34, class: "ew-s ew-end" }, "τ"));
      svg.append(E.s("text", { x: X0 - 8, y: 122, class: "ew-s ew-end" }, "ΔI"));
      svg.append(E.s("polyline", { "data-k": "curve", points: all.map(([t, d]) => sx(t).toFixed(1) + "," + sy(d).toFixed(1)).join(" "), class: "ew-curve-f" }));
      const cur = split(tau, kind);
      svg.append(E.s("circle", { cx: sx(best.tau), cy: sy(best.dI), r: 8, class: "ew-edge", style: "stroke:var(--muted);stroke-width:1.4" }));
      svg.append(E.s("circle", { "data-k": "dot", cx: sx(tau), cy: sy(cur.dI), r: 4.5, class: "ew-dot-d" }));

      const fmtN = (s) => "n=" + s.n + ",\\ p_1=" + (s.n ? f(s.p, 2) : "—");
      E.formulaRows(rows, svg, [
        { tex: "I(t)\\ ：\\ " + TEX[kind], num: "I(t)=" + f(cur.P.I) + "\\ (n=" + cur.P.n + ",\\ p_1=" + f(cur.P.p, 2) + ")", hl: ["L", "R"] },
        { tex: "I(t_L)", num: f(cur.l.I) + "\\ (" + fmtN(cur.l) + ")", hl: ["L", "tau"] },
        { tex: "I(t_R)", num: f(cur.r.I) + "\\ (" + fmtN(cur.r) + ")", hl: ["R", "tau"] },
        { tex: "\\Delta I=I(t)-\\tfrac{|t_L|}{|t|}I(t_L)-\\tfrac{|t_R|}{|t|}I(t_R)", num: "\\Delta I=" + f(cur.dI) + "\\ \\text{（" + NAMES[kind] + "）}", hl: ["dot", "curve"] },
      ]);
    }

    draw();
  });
})();
