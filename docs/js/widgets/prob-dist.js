// 主な確率分布：パラメータを動かして、形・期待値・分散を確かめる
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 対数ガンマ関数（Lanczos 近似）
  const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgamma(1 - z);
    z -= 1;
    let x = LG[0];
    for (let i = 1; i < 9; i++) x += LG[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  E.math.lgamma = lgamma;

  const normal = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / Math.sqrt(2 * Math.PI * s * s);
  const tdist = (x, m, s, nu) =>
    Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / Math.sqrt(Math.PI * nu * s * s) *
    Math.pow(1 + ((x - m) ** 2) / (nu * s * s), -(nu + 1) / 2);
  const binom = (x, n, p) =>
    Math.exp(lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * Math.log(p) + (n - x) * Math.log(1 - p));
  const pois = (x, lam) => Math.exp(x * Math.log(lam) - lam - lgamma(x + 1));
  E.math.dist = { normal, tdist, binom, pois };

  // 分布の定義。build(パラメータ) → 描画に必要な情報
  const DISTS = {
    normal: {
      label: "正規",
      sliders: [
        { key: "mu", label: "\\mu", min: -3, max: 3, step: 0.1, value: 0 },
        { key: "sg", label: "\\sigma", min: 0.3, max: 3, step: 0.1, value: 1 },
      ],
      build: (q) => ({
        range: [-8, 8], ticks: [-8, -4, 0, 4, 8],
        pdf: (x) => normal(x, q.mu, q.sg), mean: q.mu, variance: q.sg ** 2,
        fTex: "f(x)=\\dfrac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp\\!\\Bigl(-\\dfrac{(x-\\mu)^2}{2\\sigma^2}\\Bigr)",
        fNum: "f(\\mu)=" + E.fmt(normal(q.mu, q.mu, q.sg), 3),
        eTex: "\\mathbb{E}[X]=\\mu", vTex: "\\mathbb{V}[X]=\\sigma^2",
      }),
    },
    t: {
      label: "t",
      sliders: [
        { key: "mu", label: "\\mu", min: -3, max: 3, step: 0.1, value: 0 },
        { key: "sg", label: "\\sigma", min: 0.3, max: 2, step: 0.1, value: 1 },
        { key: "nu", label: "\\nu", min: 1, max: 30, step: 1, value: 3, digits: 0 },
      ],
      build: (q) => ({
        range: [-8, 8], ticks: [-8, -4, 0, 4, 8],
        pdf: (x) => tdist(x, q.mu, q.sg, q.nu),
        overlay: [{ pdf: (x) => normal(x, q.mu, q.sg) }], overlayNote: "灰：同じ μ・σ の正規分布",
        mean: q.nu > 1 ? q.mu : null,
        variance: q.nu > 2 ? (q.nu * q.sg ** 2) / (q.nu - 2) : null,
        fTex: "f(x)\\propto\\Bigl(1+\\dfrac{(x-\\mu)^2}{\\nu\\sigma^2}\\Bigr)^{-(\\nu+1)/2}",
        fNum: "f(\\mu)=" + E.fmt(tdist(q.mu, q.mu, q.sg, q.nu), 3),
        eTex: "\\mathbb{E}[X]=\\mu\\ (\\nu>1)", vTex: "\\mathbb{V}[X]=\\dfrac{\\nu\\sigma^2}{\\nu-2}\\ (\\nu>2)",
      }),
    },
    gmm: {
      label: "混合ガウス",
      sliders: [
        { key: "w", label: "\\pi_1", min: 0.05, max: 0.95, step: 0.05, value: 0.4 },
        { key: "m1", label: "\\mu_1", min: -5, max: 0, step: 0.1, value: -2 },
        { key: "m2", label: "\\mu_2", min: 0, max: 5, step: 0.1, value: 2.5 },
        { key: "sg", label: "\\sigma", min: 0.3, max: 2, step: 0.1, value: 0.8 },
      ],
      build: (q) => {
        const mean = q.w * q.m1 + (1 - q.w) * q.m2;
        const variance = q.w * (q.sg ** 2 + q.m1 ** 2) + (1 - q.w) * (q.sg ** 2 + q.m2 ** 2) - mean ** 2;
        return {
          range: [-8, 8], ticks: [-8, -4, 0, 4, 8],
          pdf: (x) => q.w * normal(x, q.m1, q.sg) + (1 - q.w) * normal(x, q.m2, q.sg),
          overlay: [
            { pdf: (x) => q.w * normal(x, q.m1, q.sg), dash: true },
            { pdf: (x) => (1 - q.w) * normal(x, q.m2, q.sg), dash: true },
          ],
          overlayNote: "灰：重みをかけた各成分",
          mean, variance,
          fTex: "f(x)=\\sum_{k=1}^{2}\\pi_k\\,\\mathcal{N}(x\\mid\\mu_k,\\sigma^2),\\ \\pi_2=1-\\pi_1",
          fNum: "\\pi_1=" + E.fmt(q.w, 2) + ",\\ \\pi_2=" + E.fmt(1 - q.w, 2),
          eTex: "\\mathbb{E}[X]=\\sum_k\\pi_k\\mu_k",
          vTex: "\\mathbb{V}[X]=\\sum_k\\pi_k(\\sigma^2+\\mu_k^2)-\\mathbb{E}[X]^2",
        };
      },
    },
    binom: {
      label: "二項",
      sliders: [
        { key: "n", label: "n", min: 1, max: 30, step: 1, value: 10, digits: 0 },
        { key: "p", label: "p", min: 0.05, max: 0.95, step: 0.05, value: 0.3 },
      ],
      build: (q) => {
        const mode = Math.min(q.n, Math.floor((q.n + 1) * q.p));
        return {
          discrete: true, range: [0, 30], ticks: [0, 5, 10, 15, 20, 25, 30], kmax: q.n,
          pdf: (x) => binom(x, q.n, q.p), mean: q.n * q.p, variance: q.n * q.p * (1 - q.p),
          fTex: "P(X=x)=\\dbinom{n}{x}p^{x}(1-p)^{n-x}",
          fNum: "P(X=" + mode + ")=" + E.fmt(binom(mode, q.n, q.p), 3),
          eTex: "\\mathbb{E}[X]=np", vTex: "\\mathbb{V}[X]=np(1-p)",
        };
      },
    },
    pois: {
      label: "ポアソン",
      sliders: [{ key: "lam", label: "\\lambda", min: 0.5, max: 15, step: 0.5, value: 4, digits: 1 }],
      build: (q) => {
        const mode = Math.floor(q.lam);
        return {
          discrete: true, range: [0, 30], ticks: [0, 5, 10, 15, 20, 25, 30], kmax: 30,
          pdf: (x) => pois(x, q.lam), mean: q.lam, variance: q.lam,
          fTex: "P(X=x)=\\dfrac{\\lambda^{x}e^{-\\lambda}}{x!}",
          fNum: "P(X=" + mode + ")=" + E.fmt(pois(mode, q.lam), 3),
          eTex: "\\mathbb{E}[X]=\\lambda", vTex: "\\mathbb{V}[X]=\\lambda",
        };
      },
    },
    exp: {
      label: "指数",
      sliders: [{ key: "lam", label: "\\lambda", min: 0.2, max: 3, step: 0.1, value: 1 }],
      build: (q) => ({
        range: [0, 8], ticks: [0, 2, 4, 6, 8],
        pdf: (x) => (x < 0 ? 0 : q.lam * Math.exp(-q.lam * x)), mean: 1 / q.lam, variance: 1 / q.lam ** 2,
        fTex: "f(x)=\\lambda e^{-\\lambda x}\\ (x\\ge0)", fNum: "f(0)=\\lambda=" + E.fmt(q.lam, 2),
        eTex: "\\mathbb{E}[X]=1/\\lambda", vTex: "\\mathbb{V}[X]=1/\\lambda^2",
      }),
    },
    unif: {
      label: "一様",
      sliders: [
        { key: "a", label: "a", min: -4, max: 0, step: 0.1, value: -2 },
        { key: "b", label: "b", min: 0.5, max: 4, step: 0.1, value: 3 },
      ],
      build: (q) => ({
        range: [-5, 5], ticks: [-4, -2, 0, 2, 4],
        pdf: (x) => (x >= q.a && x <= q.b ? 1 / (q.b - q.a) : 0),
        vertices: [[q.a, 0], [q.a, 1 / (q.b - q.a)], [q.b, 1 / (q.b - q.a)], [q.b, 0]],
        mean: (q.a + q.b) / 2, variance: (q.b - q.a) ** 2 / 12,
        fTex: "f(x)=\\dfrac{1}{b-a}\\ (a\\le x\\le b)", fNum: "\\dfrac{1}{b-a}=" + E.fmt(1 / (q.b - q.a), 3),
        eTex: "\\mathbb{E}[X]=\\dfrac{a+b}{2}", vTex: "\\mathbb{V}[X]=\\dfrac{(b-a)^2}{12}",
      }),
    },
  };
  E.math.distDefs = DISTS;

  const X0 = 40, X1 = 610, Y0 = 56, Y1 = 236;
  const NAVY = "stroke:var(--main)";

  E.register("prob-dist", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const svg = E.svg(el, 640, 272, "主な確率分布の確率密度関数または確率質量関数を、期待値と標準偏差とともに示すグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const box = E.h("div");
    el.append(box);
    let sl = null;
    const toggle = E.toggle(
      top,
      Object.entries(DISTS).map(([value, d]) => ({ value, label: d.label })),
      "normal",
      () => { mount(); draw(); }
    );
    function mount() {
      box.innerHTML = "";
      sl = E.sliders(box, DISTS[toggle.get()].sliders, () => draw());
    }

    function draw() {
      const D = DISTS[toggle.get()];
      const q = sl.get();
      const info = D.build(q);
      const f = E.fmt;
      const [xa, xb] = info.range;
      const sx = (x) => X0 + ((x - xa) / (xb - xa)) * (X1 - X0);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 縦軸の最大値
      let ymax = 0;
      const N = 320;
      const xs = [];
      if (info.discrete) for (let k = 0; k <= info.kmax; k++) xs.push(k);
      else for (let i = 0; i <= N; i++) xs.push(xa + ((xb - xa) * i) / N);
      xs.forEach((x) => { ymax = Math.max(ymax, info.pdf(x)); });
      if (!(ymax > 0)) ymax = 1;
      const sy = (v) => Y1 - (v / ymax) * (Y1 - Y0);

      // 目盛
      info.ticks.forEach((t) => {
        svg.append(E.s("line", { x1: sx(t), y1: Y0 - 6, x2: sx(t), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(t), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(t)));
      });
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-edge" }));

      // 重ねて描く参考曲線（灰）
      (info.overlay || []).forEach((o) => {
        const pts = xs.map((x) => sx(x).toFixed(1) + "," + sy(o.pdf(x)).toFixed(1)).join(" ");
        svg.append(E.s("polyline", { points: pts, style: "fill:none;stroke:var(--line-strong);stroke-width:1.8;" + (o.dash ? "stroke-dasharray:4 3" : "") }));
      });

      // 本体
      if (info.discrete) {
        const bw = Math.max(((X1 - X0) / (xb - xa + 1)) * 0.7, 3);
        xs.forEach((x) => {
          const h = Y1 - sy(info.pdf(x));
          svg.append(E.s("rect", { x: sx(x) - bw / 2, y: sy(info.pdf(x)), width: bw, height: Math.max(h, 0), rx: 1.5, class: "ew-bar-0", "data-k": "pdf" }));
        });
      } else {
        const pts = info.vertices
          ? info.vertices.map(([x, v]) => sx(x).toFixed(1) + "," + sy(v).toFixed(1)).join(" ")
          : xs.map((x) => sx(x).toFixed(1) + "," + sy(info.pdf(x)).toFixed(1)).join(" ");
        svg.append(E.s("polyline", { points: pts, class: "ew-curve-f", "data-k": "pdf" }));
      }

      // 期待値（縦線）と ±標準偏差（横棒）
      const m = info.mean, sd = info.variance == null ? null : Math.sqrt(info.variance);
      const clampX = (x) => Math.min(xb, Math.max(xa, x));
      if (m != null) {
        svg.append(E.s("line", { x1: sx(clampX(m)), y1: 36, x2: sx(clampX(m)), y2: Y1, class: "ew-curve-d", "data-k": "mean" }));
        if (sd != null) {
          const l = sx(clampX(m - sd)), r = sx(clampX(m + sd));
          svg.append(E.s("line", { x1: l, y1: 36, x2: r, y2: 36, "data-k": "sd", style: NAVY + ";stroke-width:2.4" }));
          [l, r].forEach((x) => svg.append(E.s("line", { x1: x, y1: 31, x2: x, y2: 41, "data-k": "sd", style: NAVY + ";stroke-width:2.4" })));
        }
      }
      svg.append(E.s("text", { x: 16, y: 12, class: "ew-s ew-start" }, "紺の縦線：期待値　紺の横棒：期待値 ± 標準偏差"));
      if (info.overlayNote) svg.append(E.s("text", { x: 624, y: 12, class: "ew-s ew-end" }, info.overlayNote));

      const eNum = m == null ? "\\text{存在しない}" : "=" + f(m, 3);
      const vNum = info.variance == null ? "\\text{存在しない（裾が重い）}" : "=" + f(info.variance, 3) + ",\\ \\sqrt{\\mathbb{V}}=" + f(sd, 3);
      E.formulaRows(rows, svg, [
        { tex: info.fTex, num: info.fNum, hl: ["pdf"] },
        { tex: info.eTex, num: eNum, hl: ["mean"] },
        { tex: info.vTex, num: vNum, hl: ["sd"] },
      ]);
    }
    mount();
    draw();
  });
})();
