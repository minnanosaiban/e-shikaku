// 最適化手法の軌跡：細長い谷 L = (θ1² + κ θ2²)/2 を、同じ出発点から下る
(function () {
  const E = window.ENote;

  const EPS = 1e-8;
  const MAXV = 1e6; // 発散しても有限の値に収める
  const clip = (v) => Math.max(-MAXV, Math.min(MAXV, v));
  const START = [-4, 1.2];
  const NSTEP = 60;

  // L = (x² + κ y²)/2 の勾配
  const grad = (p, k) => [p[0], k * p[1]];
  const loss = (p, k) => 0.5 * (p[0] * p[0] + k * p[1] * p[1]);

  // 手法ごとの軌跡（NSTEP+1 個の点）。b は Momentum 系の係数・RMSprop の ρ・Adam の β1
  function simulate(kind, eta, b, k) {
    let th = START.slice();
    let v = [0, 0], h = [0, 0], m = [0, 0], s = [0, 0];
    const path = [th.slice()];
    for (let t = 1; t <= NSTEP; t++) {
      let next = th.slice();
      if (kind === "sgd") {
        const g = grad(th, k);
        next = th.map((x, i) => x - eta * g[i]);
      } else if (kind === "momentum") {
        const g = grad(th, k);
        v = v.map((x, i) => b * x - eta * g[i]);
        next = th.map((x, i) => x + v[i]);
      } else if (kind === "nesterov") {
        const g = grad(th.map((x, i) => x + b * v[i]), k); // 先読みした位置の勾配
        v = v.map((x, i) => b * x - eta * g[i]);
        next = th.map((x, i) => x + v[i]);
      } else if (kind === "adagrad") {
        const g = grad(th, k);
        h = h.map((x, i) => x + g[i] * g[i]);
        next = th.map((x, i) => x - (eta * g[i]) / (Math.sqrt(h[i]) + EPS));
      } else if (kind === "rmsprop") {
        const g = grad(th, k);
        h = h.map((x, i) => b * x + (1 - b) * g[i] * g[i]);
        next = th.map((x, i) => x - (eta * g[i]) / (Math.sqrt(h[i]) + EPS));
      } else if (kind === "adam") {
        const g = grad(th, k);
        const b2 = 0.999;
        m = m.map((x, i) => b * x + (1 - b) * g[i]);
        s = s.map((x, i) => b2 * x + (1 - b2) * g[i] * g[i]);
        const mh = m.map((x) => x / (1 - Math.pow(b, t)));
        const sh = s.map((x) => x / (1 - Math.pow(b2, t)));
        next = th.map((x, i) => x - (eta * mh[i]) / (Math.sqrt(sh[i]) + EPS));
      }
      th = next.map(clip);
      path.push(th.slice());
    }
    return path;
  }
  E._optSim = simulate; // 検算用

  const KINDS = [
    { value: "sgd", label: "SGD", tex: "\\theta\\leftarrow\\theta-\\eta\\,g" },
    { value: "momentum", label: "Momentum", tex: "v\\leftarrow\\beta v-\\eta g,\\ \\ \\theta\\leftarrow\\theta+v" },
    {
      value: "nesterov", label: "Nesterov",
      tex: "g=\\nabla L(\\theta+\\beta v),\\ v\\leftarrow\\beta v-\\eta g,\\ \\theta\\leftarrow\\theta+v",
    },
    { value: "adagrad", label: "AdaGrad", tex: "h\\leftarrow h+g\\odot g,\\ \\ \\theta\\leftarrow\\theta-\\eta\\,\\dfrac{g}{\\sqrt{h}+\\epsilon}" },
    {
      value: "rmsprop", label: "RMSprop",
      tex: "h\\leftarrow\\rho h+(1-\\rho)\\,g\\odot g,\\ \\ \\theta\\leftarrow\\theta-\\eta\\,\\dfrac{g}{\\sqrt{h}+\\epsilon}",
    },
    {
      value: "adam", label: "Adam",
      tex: "\\theta\\leftarrow\\theta-\\eta\\,\\dfrac{\\hat{m}}{\\sqrt{\\hat{v}}+\\epsilon}\\ \\ (\\beta_1=\\beta,\\ \\beta_2=0.999)",
    },
  ];

  // 描画範囲：θ1 ∈ [-5, 5]、θ2 ∈ [-2.5, 2.5]
  const X0 = 40, X1 = 600, Y0 = 20, Y1 = 300;
  const sx = (x) => X0 + ((x + 5) / 10) * (X1 - X0);
  const sy = (y) => Y1 - ((y + 2.5) / 5) * (Y1 - Y0);
  const pts = (path, n) =>
    path.slice(0, n + 1).map((p) => sx(Math.max(-9, Math.min(9, p[0]))).toFixed(1) + "," + sy(Math.max(-9, Math.min(9, p[1]))).toFixed(1)).join(" ");

  E.register("opt-descent", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(top, KINDS.map((k) => ({ value: k.value, label: k.label })), "sgd", () => draw());
    const svg = E.svg(el, 640, 345, "細長い谷の形をした損失の等高線の上を、最適化手法ごとに進む軌跡を示す図");
    // 等高線を図の枠で切り取る
    const clipId = "optclip-" + Math.random().toString(36).slice(2, 8);
    const cp = E.s("clipPath", { id: clipId });
    cp.append(E.s("rect", { x: X0, y: Y0, width: X1 - X0, height: Y1 - Y0 }));
    svg.firstChild.nextSibling.append(cp);
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "eta", label: "\\eta", min: 0.01, max: 0.3, step: 0.01, value: 0.12 },
        { key: "b", label: "\\beta", min: 0, max: 0.99, step: 0.01, value: 0.9 },
        { key: "k", label: "\\kappa", min: 1, max: 30, step: 1, value: 10, digits: 0 },
        { key: "t", label: "t", min: 0, max: NSTEP, step: 1, value: 25, digits: 0 },
      ],
      () => draw()
    );

    function draw() {
      const kind = toggle.get();
      const { eta, b, k, t } = sl.get();
      const f = E.fmt;
      const nf = (v) => (Math.abs(v) >= 1000 ? E.fmtSci(v) : f(v));
      while (svg.childNodes.length > 2) svg.lastChild.remove(); // title と defs（clipPath を含む）は残す

      // 等高線（楕円）と軸
      const g = E.s("g", { "clip-path": `url(#${clipId})` });
      [0.3, 1, 2.5, 5, 9, 14].forEach((c) => {
        g.append(E.s("ellipse", {
          cx: sx(0), cy: sy(0), rx: ((X1 - X0) / 10) * Math.sqrt(2 * c),
          ry: ((Y1 - Y0) / 5) * Math.sqrt((2 * c) / k), fill: "none", class: "ew-grid",
        }));
      });
      svg.append(g);
      svg.append(E.s("rect", { x: X0, y: Y0, width: X1 - X0, height: Y1 - Y0, fill: "none", class: "ew-edge" }));
      svg.append(E.s("line", { x1: X0, y1: sy(0), x2: X1, y2: sy(0), class: "ew-grid" }));
      svg.append(E.s("line", { x1: sx(0), y1: Y0, x2: sx(0), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: X1 + 8, y: sy(0), class: "ew-s ew-start ew-muted" }, "θ₁"));
      svg.append(E.s("text", { x: sx(0) + 10, y: Y0 - 8, class: "ew-s ew-start ew-muted" }, "θ₂"));

      // 軌跡：ほかの手法はグレーの細線、選んだ手法は紺
      const sims = {};
      KINDS.forEach((kd) => { sims[kd.value] = simulate(kd.value, eta, b, k); });
      KINDS.forEach((kd) => {
        if (kd.value === kind) return;
        svg.append(E.s("polyline", { points: pts(sims[kd.value], t), fill: "none", class: "ew-edge", "clip-path": `url(#${clipId})` }));
      });
      const path = sims[kind];
      svg.append(E.s("polyline", {
        "data-k": "path", points: pts(path, t), fill: "none", style: "stroke:var(--main);stroke-width:1.8",
        "clip-path": `url(#${clipId})`,
      }));
      // 出発点・現在位置
      svg.append(E.s("circle", { cx: sx(START[0]), cy: sy(START[1]), r: 3.5, class: "ew-node" }));
      svg.append(E.s("text", { x: sx(START[0]) + 10, y: sy(START[1]) - 10, class: "ew-s ew-start" }, "出発点"));
      const cur = path[t];
      const inside = Math.abs(cur[0]) <= 5 && Math.abs(cur[1]) <= 2.5;
      if (inside) svg.append(E.s("circle", { "data-k": "pt", cx: sx(cur[0]), cy: sy(cur[1]), r: 4.5, class: "ew-dot-d" }));

      // 凡例（図の下）
      svg.append(E.s("line", { x1: X0, y1: Y1 + 28, x2: X0 + 30, y2: Y1 + 28, style: "stroke:var(--main);stroke-width:1.8" }));
      svg.append(E.s("text", { x: X0 + 38, y: Y1 + 28, class: "ew-s ew-start" }, "選んだ手法"));
      svg.append(E.s("line", { x1: X0 + 150, y1: Y1 + 28, x2: X0 + 180, y2: Y1 + 28, class: "ew-edge" }));
      svg.append(E.s("text", { x: X0 + 188, y: Y1 + 28, class: "ew-s ew-start" }, "ほかの手法（同じ η, β）"));

      const Lc = loss(cur, k);
      const div = Lc > 1e4;
      const info = KINDS.find((x) => x.value === kind);
      E.formulaRows(rows, svg, [
        { tex: info.tex, num: "\\eta=" + f(eta) + ",\\ \\beta=" + f(b), hl: ["path"] },
        { tex: "\\theta_t=(\\theta_1,\\theta_2)", num: "t=" + t + ":\\ (" + nf(cur[0]) + ",\\ " + nf(cur[1]) + ")", hl: ["pt"] },
        {
          tex: "L(\\theta)=\\tfrac12(\\theta_1^2+\\kappa\\,\\theta_2^2)",
          num: div ? "\\text{発散}" : "=" + f(Lc, 4),
          hl: ["pt"],
        },
      ]);
    }

    draw();
  });
})();
