// k-means：割り当てと更新を1手ずつ進め、J が単調に減ること・初期値で結果が変わることを見る
(function () {
  // ---- 計算（node でも検算できるように、描画から切り離す）----
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
  // 3つの塊（各 30 点）、領域は [0,10]²
  const DATA = (() => {
    const r = rng(7), pts = [];
    const centers = [[2.2, 2.6], [7.6, 2.8], [4.9, 7.4]];
    centers.forEach(([cx, cy]) => {
      for (let i = 0; i < 30; i++) {
        const u = Math.max(r(), 1e-9), v = r();
        const m = Math.sqrt(-2 * Math.log(u)) * 0.85;
        pts.push([
          Math.min(9.8, Math.max(0.2, cx + m * Math.cos(2 * Math.PI * v))),
          Math.min(9.8, Math.max(0.2, cy + m * Math.sin(2 * Math.PI * v))),
        ]);
      }
    });
    return pts;
  })();
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

  function initRandom(K, seed) {
    const r = rng(seed * 101 + 3), idx = [];
    while (idx.length < K) {
      const i = Math.floor(r() * DATA.length);
      if (!idx.includes(i)) idx.push(i);
    }
    return idx.map((i) => DATA[i].slice());
  }
  function initPP(K, seed) {
    const r = rng(seed * 101 + 3);
    const mu = [DATA[Math.floor(r() * DATA.length)].slice()];
    while (mu.length < K) {
      const d = DATA.map((p) => Math.min(...mu.map((m) => d2(p, m)))); // D(x)²
      const tot = d.reduce((s, v) => s + v, 0);
      let u = r() * tot, i = 0;
      while (i < d.length - 1 && u > d[i]) { u -= d[i]; i++; }
      mu.push(DATA[i].slice());
    }
    return mu;
  }
  const assign = (mu) => DATA.map((p) => {
    let b = 0;
    mu.forEach((m, k) => { if (d2(p, m) < d2(p, mu[b])) b = k; });
    return b;
  });
  function update(mu, c) {
    return mu.map((m, k) => {
      const mem = DATA.filter((_, i) => c[i] === k);
      if (!mem.length) return m.slice(); // 空のクラスタは動かさない
      return [mem.reduce((s, p) => s + p[0], 0) / mem.length, mem.reduce((s, p) => s + p[1], 0) / mem.length];
    });
  }
  const cost = (mu, c) => DATA.reduce((s, p, i) => s + d2(p, mu[c[i]]), 0);

  if (typeof module !== "undefined") module.exports = { DATA, initRandom, initPP, assign, update, cost };
  if (typeof window === "undefined") return;

  // ---- 描画 ----
  const E = window.ENote;
  // クラスタごとの印の形（色ではなく形で区別する）
  const SHAPES = [
    (x, y, r) => `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0z`,
    (x, y, r) => `M${x - r} ${y - r}h${2 * r}v${2 * r}h${-2 * r}z`,
    (x, y, r) => `M${x} ${y - 1.2 * r}L${x + 1.2 * r} ${y + 0.9 * r}L${x - 1.2 * r} ${y + 0.9 * r}z`,
    (x, y, r) => `M${x} ${y - 1.3 * r}L${x + 1.3 * r} ${y}L${x} ${y + 1.3 * r}L${x - 1.3 * r} ${y}z`,
    (x, y, r) => `M${x} ${y + 1.2 * r}L${x + 1.2 * r} ${y - 0.9 * r}L${x - 1.2 * r} ${y - 0.9 * r}z`,
    (x, y, r) => `M${x - r} ${y - 0.5 * r}L${x} ${y - 1.1 * r}L${x + r} ${y - 0.5 * r}L${x + r} ${y + 0.5 * r}L${x} ${y + 1.1 * r}L${x - r} ${y + 0.5 * r}z`,
  ];
  const MAXH = 14;

  E.register("clustering-kmeans", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const initT = E.toggle(
      top,
      [{ value: "rand", label: "ランダム初期化" }, { value: "pp", label: "k-means++" }],
      "rand",
      () => reset()
    );
    const btnStep = E.h("button", { type: "button", class: "ew-btn", onclick: () => step() }, "1手進める");
    const btnRun = E.h("button", { type: "button", class: "ew-btn", onclick: () => run() }, "収束まで");
    top.append(btnStep, btnRun);

    const svg = E.svg(el, 640, 300, "2次元の点群をk-meansで分けていく過程と、目的関数Jの推移を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "K", label: "K", min: 1, max: 6, step: 1, value: 3, digits: 0 },
        { key: "seed", label: "\\text{乱数}", min: 1, max: 20, step: 1, value: 6, digits: 0 },
      ],
      () => reset()
    );

    // 状態：mu（重心）、c（割り当て。最初は null）、phase（次にやること）、hist（J の履歴）
    let st;
    function reset() {
      const { K, seed } = sl.get();
      const mu = initT.get() === "pp" ? initPP(K, seed) : initRandom(K, seed);
      st = { mu, prev: null, c: null, phase: "assign", hist: [], changed: null, done: false };
      draw();
    }
    function step() {
      if (st.done) return;
      if (st.phase === "assign") {
        const c = assign(st.mu);
        st.changed = st.c ? c.filter((v, i) => v !== st.c[i]).length : DATA.length;
        st.c = c;
        st.prev = null;
        if (st.changed === 0) st.done = true; // 変化なし＝収束
        else st.hist.push(cost(st.mu, c));
        st.phase = "update";
      } else {
        st.prev = st.mu;
        st.mu = update(st.mu, st.c);
        st.hist.push(cost(st.mu, st.c));
        st.phase = "assign";
      }
      if (st.hist.length > MAXH) st.hist.shift();
      draw();
    }
    function run() {
      for (let i = 0; i < 200 && !st.done; i++) step();
    }

    const PX = (x) => 20 + x * 38, PY = (y) => 285 - y * 27;
    function draw() {
      const f = E.fmt;
      const { K } = sl.get();
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      btnStep.disabled = btnRun.disabled = st.done;
      btnStep.style.opacity = btnRun.style.opacity = st.done ? 0.4 : 1;

      // 左：点と重心
      svg.append(E.s("rect", { x: 20, y: 15, width: 380, height: 270, rx: 3, class: "ew-node", style: "fill:none" }));
      DATA.forEach((p, i) => {
        const k = st.c ? st.c[i] : -1;
        const x = PX(p[0]), y = PY(p[1]);
        if (k < 0) svg.append(E.s("circle", { "data-k": "pts", cx: x, cy: y, r: 3.4, class: "ew-dot-f", style: "opacity:0.35" }));
        else svg.append(E.s("path", { "data-k": "pts", d: SHAPES[k % 6](x, y, 3.4), class: "ew-dot-f", style: "opacity:0.6" }));
      });
      if (st.prev) {
        st.prev.forEach((m, k) => {
          svg.append(E.s("line", {
            "data-k": "cent", x1: PX(m[0]), y1: PY(m[1]), x2: PX(st.mu[k][0]), y2: PY(st.mu[k][1]),
            class: "ew-edge", style: "stroke:var(--main);stroke-dasharray:3 3",
          }));
        });
      }
      st.mu.forEach((m, k) => {
        svg.append(E.s("path", {
          "data-k": "cent", d: SHAPES[k % 6](PX(m[0]), PY(m[1]), 7), class: "ew-dot-d", style: "stroke:var(--surface);stroke-width:1.5",
        }));
      });

      // 右：J の履歴
      const X0 = 440, X1 = 620, Y0 = 40, Y1 = 250;
      svg.append(E.s("text", { x: X0, y: 16, class: "ew-s ew-start" }, "目的関数 J の推移"));
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-grid" }));
      const jmax = Math.max(...st.hist, 1);
      const bw = (X1 - X0) / MAXH;
      st.hist.forEach((j, i) => {
        const h = (j / jmax) * (Y1 - Y0);
        const last = i === st.hist.length - 1;
        svg.append(E.s("rect", {
          "data-k": "bars", x: X0 + i * bw + 1.5, y: Y1 - h, width: bw - 3, height: h, rx: 2,
          class: last ? "ew-dot-d" : "ew-bar-0",
        }));
      });
      svg.append(E.s("text", { x: (X0 + X1) / 2, y: Y1 + 20, class: "ew-s ew-c" }, "手数（割り当て・更新を交互に）"));

      const sizes = st.c ? st.mu.map((_, k) => st.c.filter((v) => v === k).length) : null;
      const jNow = st.c ? cost(st.mu, st.c) : null;
      const nextTxt = st.done ? "収束（割り当てが変化しない）" : st.phase === "assign" ? "次は割り当て" : "次は更新";
      E.formulaRows(rows, svg, [
        {
          tex: "c^{(i)}=\\operatorname*{arg\\,min}_k\\|\\mathbf{x}^{(i)}-\\boldsymbol{\\mu}_k\\|^2",
          num: st.changed == null ? "\\text{まだ割り当てていません}" : "\\text{直近の割り当てで変化した点：}" + st.changed + "\\,/\\," + DATA.length,
          hl: ["pts"],
        },
        {
          tex: "\\boldsymbol{\\mu}_k=\\frac{1}{|C_k|}\\sum_{i\\in C_k}\\mathbf{x}^{(i)}",
          num: sizes ? "|C_k|=" + sizes.join(",\\ ") : "K=" + K,
          hl: ["cent"],
        },
        {
          tex: "J=\\sum_{k}\\sum_{i\\in C_k}\\|\\mathbf{x}^{(i)}-\\boldsymbol{\\mu}_k\\|^2",
          num: (jNow == null ? "—" : "=" + f(jNow, 1)) + "\\quad\\text{（" + nextTxt + "）}",
          hl: ["bars"],
        },
      ]);
    }
    reset();
  });
})();
