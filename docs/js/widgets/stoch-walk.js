// ランダムウォークとウィーナー過程：刻みを細かくしても X_T の分布は変わらない（平均 μT、分散 σ²T）
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 種つきの標準正規乱数（Box–Muller）
  function normals(n, seed) {
    let s = seed >>> 0;
    const u = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
    const out = [];
    for (let i = 0; i < n; i += 2) {
      const r = Math.sqrt(-2 * Math.log(u())), t = 2 * Math.PI * u();
      out.push(r * Math.cos(t), r * Math.sin(t));
    }
    return out.slice(0, n);
  }

  const T = 10, NF = 100, NP = 24; // 全体の時間、細かい刻みの数、標本路の数
  // 細かい刻みで作ったウィーナー過程 W(t)（W(0)=0、増分は N(0, Δt)）。粗い刻みはここから間引くので、n を変えても同じ路を見られる
  E.math.wienerPaths = function () {
    const dt = T / NF;
    const z = normals(NP * NF, 2024);
    return Array.from({ length: NP }, (_, p) => {
      const w = [0];
      for (let i = 0; i < NF; i++) w.push(w[i] + Math.sqrt(dt) * z[p * NF + i]);
      return w;
    });
  };

  // 目盛りのきりのよい上限
  function niceCeil(x) {
    const e = Math.pow(10, Math.floor(Math.log10(x)));
    for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * e >= x) return m * e;
    return 10 * e;
  }
  E.math.niceCeil = niceCeil;

  const W = E.math.wienerPaths();
  const X0 = 60, X1 = 600, Y0 = 44, Y1 = 236;
  const sx = (t) => X0 + (t / T) * (X1 - X0);

  E.register("stoch-walk", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const svg = E.svg(el, 640, 276, "ドリフト付きのランダムウォークの標本路を24本重ねた図。期待値の直線と、標準偏差の幅の帯を合わせて示す");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const toggle = E.toggle(
      top,
      [5, 10, 20, 50, 100].map((v) => ({ value: v, label: "n = " + v })),
      10,
      () => draw()
    );
    const sl = E.sliders(
      el,
      [
        { key: "mu", label: "\\mu", min: -1, max: 1, step: 0.1, value: 0.3 },
        { key: "sg", label: "\\sigma", min: 0.2, max: 2, step: 0.1, value: 1 },
      ],
      () => draw()
    );

    function draw() {
      const n = toggle.get();
      const { mu, sg } = sl.get();
      const f = E.fmt;
      const R = niceCeil(Math.max(Math.abs(mu) * T + 2.5 * sg * Math.sqrt(T), 1));
      const sy = (v) => (Y0 + Y1) / 2 - (Math.max(-1.08 * R, Math.min(1.08 * R, v)) / R) * ((Y1 - Y0) / 2); // 枠をはみ出す路は端で止める
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [-R, -R / 2, 0, R / 2, R].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: v === 0 ? "ew-edge" : "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 8, y: sy(v), class: "ew-s ew-end ew-muted" }, f(v, R < 10 ? 1 : 0)));
      });
      [0, 5, 10].forEach((t) => svg.append(E.s("text", { x: sx(t), y: Y1 + 28, class: "ew-s ew-c ew-muted" }, "t = " + t)));

      // 標準偏差の帯 μt ± σ√t
      const up = [], dn = [];
      for (let i = 0; i <= 50; i++) {
        const t = (T * i) / 50;
        up.push(sx(t).toFixed(1) + "," + sy(mu * t + sg * Math.sqrt(t)).toFixed(1));
        dn.unshift(sx(t).toFixed(1) + "," + sy(mu * t - sg * Math.sqrt(t)).toFixed(1));
      }
      svg.append(E.s("polygon", { points: up.concat(dn).join(" "), "data-k": "band", style: "fill:var(--main);fill-opacity:0.1;stroke:var(--main);stroke-width:1;stroke-dasharray:2 3" }));

      // 標本路（灰）
      let sum = 0, sum2 = 0;
      W.forEach((w) => {
        const pts = [];
        for (let i = 0; i <= n; i++) {
          const t = (T * i) / n, j = Math.round((i * NF) / n);
          pts.push(sx(t).toFixed(1) + "," + sy(mu * t + sg * w[j]).toFixed(1));
        }
        svg.append(E.s("polyline", { points: pts.join(" "), class: "ew-edge", style: "stroke-width:1", "data-k": "paths" }));
        const xT = mu * T + sg * w[NF];
        sum += xT; sum2 += xT * xT;
      });
      // 期待値 μt
      svg.append(E.s("line", { x1: sx(0), y1: sy(0), x2: sx(T), y2: sy(mu * T), class: "ew-curve-d", "data-k": "mean" }));

      svg.append(E.s("text", { x: 16, y: 14, class: "ew-s ew-start" }, "灰：標本路 " + NP + " 本　紺の破線：期待値 μt　帯：μt ± σ√t"));

      const m = sum / NP, sd = Math.sqrt(Math.max(sum2 / NP - m * m, 0));
      const dt = T / n;
      E.formulaRows(rows, svg, [
        { tex: "X_{t+\\Delta t}-X_t\\sim\\mathcal{N}(\\mu\\Delta t,\\ \\sigma^2\\Delta t)", num: "\\Delta t=" + f(dt, 2) + ":\\ \\mathcal{N}(" + f(mu * dt, 3) + ",\\ " + f(sg * sg * dt, 3) + ")", hl: ["paths"] },
        { tex: "\\mathbb{E}[X_T]=\\mu T", num: "=" + f(mu * T, 2), hl: ["mean"] },
        { tex: "\\mathbb{V}[X_T]=\\sigma^2T", num: "=" + f(sg * sg * T, 2) + ",\\ \\sigma\\sqrt{T}=" + f(sg * Math.sqrt(T), 2), hl: ["band"] },
        { tex: "\\text{標本路 " + NP + " 本の }X_T\\text{ の平均・標準偏差}", num: f(m, 2) + ",\\ " + f(sd, 2), hl: ["paths"] },
      ]);
    }
    draw();
  });
})();
