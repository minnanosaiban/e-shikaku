// 拡散モデルの前向き過程：x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε
// データ（2 つの三日月）がノイズに埋もれていく様子と、ᾱ_t のスケジュールを見る
(function () {
  const E = window.ENote;
  const T = 1000;

  // ---- ᾱ_t（t = 0..T）----
  // 線形スケジュール：β_t が 1e-4 から 0.02 まで線形に増える（DDPM）
  const alphaBarLinear = () => {
    const out = [1];
    for (let t = 1; t <= T; t++) {
      const beta = 1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1);
      out.push(out[t - 1] * (1 - beta));
    }
    return out;
  };
  // コサインスケジュール：ᾱ_t = f(t)/f(0)、f(t)=cos²(((t/T+s)/(1+s))·π/2)（Nichol & Dhariwal）
  const alphaBarCosine = () => {
    const s = 0.008;
    const f = (t) => Math.cos((((t / T + s) / (1 + s)) * Math.PI) / 2) ** 2;
    const out = [];
    for (let t = 0; t <= T; t++) out.push(f(t) / f(0));
    return out;
  };
  const AB = { linear: alphaBarLinear(), cosine: alphaBarCosine() };
  E.diffusionAlphaBar = AB;

  // ---- 固定のデータとノイズ ----
  const rng = (seed) => () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const rand = rng(7);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

  const N = 160;
  let X0 = [];
  for (let i = 0; i < N / 2; i++) {
    const a = (Math.PI * i) / (N / 2 - 1);
    X0.push([Math.cos(a), Math.sin(a)]);
    X0.push([1 - Math.cos(a), 0.5 - Math.sin(a)]);
  }
  // 平均 0・各軸の分散がおよそ 1 になるように規格化（終端の N(0, I) と同じ尺度にする）
  const mx = X0.reduce((s, p) => s + p[0], 0) / N, my = X0.reduce((s, p) => s + p[1], 0) / N;
  const sd = Math.sqrt(X0.reduce((s, p) => s + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0) / (2 * N));
  X0 = X0.map((p) => [(p[0] - mx) / sd, (p[1] - my) / sd]);
  const EPS = X0.map(() => [gauss(), gauss()]);

  // ---- 描画 ----
  const PC = { x: 160, y: 140, k: 44, half: 130 }; // 散布図の中心と 1 単位あたりの px
  const GX0 = 370, GX1 = 610, GY0 = 80, GY1 = 230; // 曲線のグラフ
  const gx = (t) => GX0 + (t / T) * (GX1 - GX0);
  const gy = (v) => GY1 - v * (GY1 - GY0);
  const clamp = (v, a) => Math.max(-a, Math.min(a, v));

  E.register("gen-diffusion", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const sched = E.toggle(
      top,
      [{ value: "linear", label: "線形スケジュール" }, { value: "cosine", label: "コサインスケジュール" }],
      "linear",
      () => draw()
    );
    const svg = E.svg(el, 640, 305, "2 つの三日月の形をしたデータに、時刻 t に応じたノイズが混ざっていく散布図と、信号の割合 ᾱ_t の曲線");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [{ key: "t", label: "t", min: 0, max: T, step: 10, value: 300, digits: 0 }],
      () => draw()
    );

    function draw() {
      const { t } = sl.get();
      const ab = AB[sched.get()];
      const a = ab[t];
      const sa = Math.sqrt(Math.max(0, a)), sn = Math.sqrt(Math.max(0, 1 - a));
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 散布図
      svg.append(E.s("rect", { "data-k": "panel", x: PC.x - PC.half, y: PC.y - PC.half, width: 2 * PC.half, height: 2 * PC.half, rx: 4, class: "ew-node" }));
      svg.append(E.s("line", { x1: PC.x - PC.half, y1: PC.y, x2: PC.x + PC.half, y2: PC.y, class: "ew-grid" }));
      svg.append(E.s("line", { x1: PC.x, y1: PC.y - PC.half, x2: PC.x, y2: PC.y + PC.half, class: "ew-grid" }));
      X0.forEach((p) => {
        svg.append(E.s("circle", { cx: PC.x + clamp(p[0], 2.9) * PC.k, cy: PC.y - clamp(p[1], 2.9) * PC.k, r: 1.8, class: "ew-bar-0" }));
      });
      X0.forEach((p, i) => {
        const x = sa * p[0] + sn * EPS[i][0], y = sa * p[1] + sn * EPS[i][1];
        svg.append(E.s("circle", { cx: PC.x + clamp(x, 2.9) * PC.k, cy: PC.y - clamp(y, 2.9) * PC.k, r: 2.6, class: "ew-dot-d" }));
      });

      // ᾱ_t のグラフ
      [0, 0.5, 1].forEach((v) => {
        svg.append(E.s("line", { x1: GX0, y1: gy(v), x2: GX1, y2: gy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: GX0 - 8, y: gy(v), class: "ew-s ew-end ew-muted" }, String(v)));
      });
      [0, 500, 1000].forEach((v) => {
        svg.append(E.s("text", { x: gx(v), y: GY1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      const pts = (g) => {
        const out = [];
        for (let i = 0; i <= T; i += 10) out.push(gx(i).toFixed(1) + "," + gy(g(ab[i])).toFixed(1));
        return out.join(" ");
      };
      svg.append(E.s("polyline", { "data-k": "cur", points: pts((v) => v), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "cur", points: pts((v) => 1 - v), class: "ew-curve-d" }));
      svg.append(E.s("line", { x1: gx(t), y1: GY0, x2: gx(t), y2: GY1, class: "ew-edge" }));
      svg.append(E.s("circle", { "data-k": "mk", cx: gx(t), cy: gy(a), r: 4, class: "ew-dot-f" }));
      svg.append(E.s("circle", { "data-k": "mk", cx: gx(t), cy: gy(1 - a), r: 4, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: GX0, y: 22, class: "ew-s ew-start" }, "実線：ᾱ(t)（元のデータの割合）"));
      svg.append(E.s("text", { x: GX0, y: 44, class: "ew-s ew-start" }, "破線：1−ᾱ(t)（ノイズの割合）"));
      svg.append(E.s("text", { x: GX1, y: 262, class: "ew-s ew-end ew-muted" }, "t"));
      svg.append(E.s("text", { x: PC.x, y: 289, class: "ew-s ew-c ew-muted" }, "灰：元のデータ　紺：ノイズ後"));

      const snr = 1 - a > 1e-12 ? f(a / (1 - a), 3) : "\\infty";
      E.formulaRows(rows, svg, [
        { tex: "\\bar\\alpha_t=\\prod_{s=1}^{t}(1-\\beta_s)", num: "\\bar\\alpha_{" + t + "}=" + f(a, 4), hl: ["cur", "mk"] },
        { tex: "\\mathbf{x}_t=\\sqrt{\\bar\\alpha_t}\\,\\mathbf{x}_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\varepsilon}", num: "=" + f(sa, 3) + "\\,\\mathbf{x}_0+" + f(sn, 3) + "\\,\\boldsymbol{\\varepsilon}", hl: ["panel"] },
        { tex: "\\mathrm{SNR}(t)=\\dfrac{\\bar\\alpha_t}{1-\\bar\\alpha_t}", num: "=\\dfrac{" + f(a, 4) + "}{" + f(1 - a, 4) + "}=" + snr, hl: ["cur", "mk"] },
      ]);
    }

    draw();
  });
})();
