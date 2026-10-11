// 損失関数の形：回帰（loss-regression）、マージン基準の分類（loss-margin）、Focal Loss（loss-focal）
(function () {
  const E = window.ENote;
  const sig = (v) => 1 / (1 + Math.exp(-v));
  // 色は CSS 変数だけを使う（直書きしない）
  const GREY = "fill:none;stroke:var(--muted);stroke-width:1.6";

  function axes(svg, X0, X1, Y0, Y1, sx, sy, xt, yt) {
    xt.forEach((v) => {
      svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: sx(v), y: Y1 + 15, class: "ew-s ew-c ew-muted" }, String(v)));
    });
    yt.forEach((v) => {
      svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
      svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
    });
    svg.append(E.s("line", { x1: X0, y1: sy(0), x2: X1, y2: sy(0), class: "ew-edge" }));
  }
  function curve(svg, g, lo, hi, sx, sy, ymin, ymax, attrs, n = 240) {
    // 範囲外の点は飛ばして、線を途切れさせる
    let cur = [];
    const flush = () => { if (cur.length > 1) svg.append(E.s("polyline", Object.assign({ points: cur.join(" ") }, attrs))); cur = []; };
    for (let i = 0; i <= n; i++) {
      const x = lo + ((hi - lo) * i) / n, y = g(x);
      if (isFinite(y) && y >= ymin && y <= ymax) cur.push(sx(x).toFixed(1) + "," + sy(y).toFixed(1));
      else flush();
    }
    flush();
  }

  // ---------- 回帰：残差 r = y − t（予測 − 正解） ----------
  E.register("loss-regression", (el) => {
    const svg = E.svg(el, 640, 352, "残差 r=y−t に対する MSE・MAE・Huber 損失と、その勾配の曲線");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "r", label: "r=y-t", min: -3, max: 3, step: 0.1, value: 1.5 },
        { key: "d", label: "\\delta", min: 0.2, max: 3, step: 0.1, value: 1 },
      ],
      () => draw()
    );
    const X0 = 70, X1 = 470, RA = 3;
    const sx = (r) => X0 + ((r + RA) / (2 * RA)) * (X1 - X0);
    // 上段：損失 0〜4.5、下段：勾配 -3〜3
    const T0 = 18, T1 = 140, B0 = 190, B1 = 312;
    const syT = (y) => T1 - (y / 4.5) * (T1 - T0);
    const syB = (y) => B1 - ((y + 3) / 6) * (B1 - B0);

    function draw() {
      const { r, d } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      const L = {
        mse: (x) => 0.5 * x * x,
        mae: (x) => Math.abs(x),
        hub: (x) => (Math.abs(x) <= d ? 0.5 * x * x : d * Math.abs(x) - 0.5 * d * d),
      };
      const G = {
        mse: (x) => x,
        mae: (x) => Math.sign(x),
        hub: (x) => Math.max(-d, Math.min(d, x)),
      };
      axes(svg, X0, X1, T0, T1, sx, syT, [-3, -2, -1, 0, 1, 2, 3], [0, 1, 2, 3, 4]);
      axes(svg, X0, X1, B0, B1, sx, syB, [-3, -2, -1, 0, 1, 2, 3], [-2, 0, 2]);
      svg.append(E.s("line", { x1: sx(0), y1: T0, x2: sx(0), y2: T1, class: "ew-edge" }));
      svg.append(E.s("line", { x1: sx(0), y1: B0, x2: sx(0), y2: B1, class: "ew-edge" }));
      svg.append(E.s("text", { x: 20, y: (T0 + T1) / 2, class: "ew-s ew-c" }, "損失"));
      svg.append(E.s("text", { x: 20, y: (B0 + B1) / 2, class: "ew-s ew-c" }, "勾配"));
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: 344, class: "ew-s ew-c ew-muted" }, "残差 r"));

      const style = {
        mse: { class: "ew-curve-f" },
        mae: { style: GREY },
        hub: { class: "ew-curve-d" },
      };
      ["mse", "mae", "hub"].forEach((k) => {
        const a = Object.assign({ "data-k": k }, style[k]);
        curve(svg, L[k], -RA, RA, sx, syT, -1, 4.5, a);
        curve(svg, G[k], -RA, RA, sx, syB, -3, 3, a);
        const dotCls = k === "hub" ? "ew-dot-d" : "ew-dot-f";
        const dot = (y, sy) => svg.append(E.s("circle", Object.assign({ "data-k": k, cx: sx(r), cy: sy(y), r: 4 }, k === "mae" ? { style: "fill:var(--muted)" } : { class: dotCls })));
        dot(L[k](r), syT);
        dot(G[k](r), syB);
      });
      // 凡例（右の余白）
      [["mse", "MSE（½r²）", "ew-curve-f", ""], ["mae", "MAE（|r|）", "", GREY], ["hub", "Huber", "ew-curve-d", ""]].forEach((e, i) => {
        const y = 40 + i * 24;
        const a = { x1: 490, y1: y, x2: 516, y2: y };
        if (e[2]) a.class = e[2]; else a.style = e[3];
        svg.append(E.s("line", a));
        svg.append(E.s("text", { x: 524, y, class: "ew-s ew-start" }, e[1]));
      });

      E.formulaRows(rows, svg, [
        { tex: "\\text{MSE}:\\ \\tfrac12 r^2,\\quad \\partial/\\partial r=r", num: f(L.mse(r)) + ",\\ " + f(G.mse(r)), hl: ["mse"] },
        { tex: "\\text{MAE}:\\ |r|,\\quad \\partial/\\partial r=\\operatorname{sign}(r)", num: f(L.mae(r)) + ",\\ " + f(G.mae(r), 0), hl: ["mae"] },
        { tex: "\\text{Huber}:\\ \\tfrac12 r^2\\ (|r|\\le\\delta),\\ \\delta|r|-\\tfrac12\\delta^2\\ (\\text{それ以外})", num: f(L.hub(r)) + ",\\ " + f(G.hub(r)), hl: ["hub"] },
      ]);
    }
    draw();
  });

  // ---------- 分類：マージン m = t f(x)（t∈{-1,+1}）----------
  const MARGIN = {
    hinge: { label: "ヒンジ", g: (m) => Math.max(0, 1 - m), d: (m) => (m < 1 ? -1 : 0), tex: "\\max(0,\\,1-m)", dtex: "-\\mathbb{1}[m<1]" },
    logistic: { label: "ロジスティック", g: (m) => Math.log(1 + Math.exp(-m)), d: (m) => -sig(-m), tex: "\\ln(1+e^{-m})", dtex: "-\\sigma(-m)" },
    exp: { label: "指数", g: (m) => Math.exp(-m), d: (m) => -Math.exp(-m), tex: "e^{-m}", dtex: "-e^{-m}" },
    sqh: { label: "2乗ヒンジ", g: (m) => Math.max(0, 1 - m) ** 2, d: (m) => -2 * Math.max(0, 1 - m), tex: "\\max(0,\\,1-m)^2", dtex: "-2\\max(0,\\,1-m)" },
  };

  E.register("loss-margin", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      Object.entries(MARGIN).map(([value, t]) => ({ value, label: t.label })),
      "hinge",
      () => draw()
    );
    const svg = E.svg(el, 640, 270, "マージンに対する 0-1 損失と代理損失の曲線");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(el, [{ key: "m", label: "m=t\\,f(\\mathbf{x})", min: -2, max: 3, step: 0.1, value: 0.5 }], () => draw());
    const X0 = 70, X1 = 590, Y0 = 20, Y1 = 225, YM = 3.5;
    const sx = (m) => X0 + ((m + 2) / 5) * (X1 - X0);
    const sy = (y) => Y1 - (y / YM) * (Y1 - Y0);

    function draw() {
      const { m } = sl.get();
      const t = MARGIN[toggle.get()];
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      axes(svg, X0, X1, Y0, Y1, sx, sy, [-2, -1, 0, 1, 2, 3], [0, 1, 2, 3]);
      svg.append(E.s("line", { x1: sx(0), y1: Y0, x2: sx(0), y2: Y1, class: "ew-edge" }));
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: 256, class: "ew-s ew-c ew-muted" }, "マージン m（正：正しく分類、負：誤分類）"));
      // 0-1 損失（階段）
      svg.append(E.s("polyline", { "data-k": "z", points: [[-2, 1], [0, 1]].map((p) => sx(p[0]) + "," + sy(p[1])).join(" "), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "z", points: [[0, 0], [3, 0]].map((p) => sx(p[0]) + "," + sy(p[1])).join(" "), class: "ew-curve-f" }));
      curve(svg, t.g, -2, 3, sx, sy, 0, YM, { "data-k": "s", class: "ew-curve-d" });
      const z = m < 0 ? 1 : 0;
      svg.append(E.s("circle", { "data-k": "z", cx: sx(m), cy: sy(z), r: 4, class: "ew-dot-f" }));
      const sv = t.g(m);
      if (sv <= YM) svg.append(E.s("circle", { "data-k": "s", cx: sx(m), cy: sy(sv), r: 4.5, class: "ew-dot-d" }));
      // 凡例（右上は曲線が通らない）
      svg.append(E.s("line", { x1: 400, y1: 40, x2: 426, y2: 40, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: 434, y: 40, class: "ew-s ew-start" }, "0-1 損失"));
      svg.append(E.s("line", { x1: 400, y1: 62, x2: 426, y2: 62, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: 434, y: 62, class: "ew-s ew-start" }, t.label));

      E.formulaRows(rows, svg, [
        { tex: "\\text{0-1 損失}=\\mathbb{1}[m<0]", num: "=" + f(z, 0), hl: ["z"] },
        { tex: "\\ell(m)=" + t.tex, num: "=" + f(sv), hl: ["s"] },
        { tex: "\\dfrac{d\\ell}{dm}=" + t.dtex, num: "=" + f(t.d(m)), hl: ["s"] },
      ]);
    }
    draw();
  });

  // ---------- Focal Loss ----------
  E.register("loss-focal", (el) => {
    const svg = E.svg(el, 640, 250, "正解クラスの確率に対する交差エントロピーと Focal Loss の曲線");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "p", label: "p_t", min: 0.05, max: 0.99, step: 0.01, value: 0.9 },
        { key: "g", label: "\\gamma", min: 0, max: 5, step: 0.5, value: 2, digits: 1 },
      ],
      () => draw()
    );
    const X0 = 70, X1 = 590, Y0 = 20, Y1 = 205, YM = 3.5;
    const sx = (p) => X0 + p * (X1 - X0);
    const sy = (y) => Y1 - (y / YM) * (Y1 - Y0);

    function draw() {
      const { p, g } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      axes(svg, X0, X1, Y0, Y1, sx, sy, [0, 0.25, 0.5, 0.75, 1], [0, 1, 2, 3]);
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: 238, class: "ew-s ew-c ew-muted" }, "正解クラスの確率（右ほど易しい例）"));
      const ce = (q) => -Math.log(q);
      const fl = (q) => (1 - q) ** g * -Math.log(q);
      curve(svg, ce, 0.03, 1, sx, sy, 0, YM, { "data-k": "ce", class: "ew-curve-f" });
      curve(svg, fl, 0.03, 1, sx, sy, 0, YM, { "data-k": "fl", class: "ew-curve-d" });
      svg.append(E.s("circle", { "data-k": "ce", cx: sx(p), cy: sy(Math.min(ce(p), YM)), r: 4, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "fl", cx: sx(p), cy: sy(fl(p)), r: 4.5, class: "ew-dot-d" }));
      svg.append(E.s("line", { x1: 360, y1: 40, x2: 386, y2: 40, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: 394, y: 40, class: "ew-s ew-start" }, "交差エントロピー"));
      svg.append(E.s("line", { x1: 360, y1: 62, x2: 386, y2: 62, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: 394, y: 62, class: "ew-s ew-start" }, "Focal Loss"));

      const w = (1 - p) ** g;
      E.formulaRows(rows, svg, [
        { tex: "\\text{CE}=-\\ln p_t", num: "=" + f(ce(p)), hl: ["ce"] },
        { tex: "\\text{変調項}\\ (1-p_t)^{\\gamma}", num: "=(1-" + f(p, 2) + ")^{" + f(g, 1) + "}=" + f(w), hl: ["fl"] },
        { tex: "\\text{FL}=(1-p_t)^{\\gamma}\\cdot\\text{CE}", num: "=" + f(w) + "\\times" + f(ce(p)) + "=" + f(fl(p)), hl: ["fl"] },
      ]);
    }
    draw();
  });
})();
