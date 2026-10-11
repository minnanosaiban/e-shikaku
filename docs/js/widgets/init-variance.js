// 重みの初期化：10 層の全結合ネットワークに乱数の入力を通し、層ごとの出力の大きさと分布を見る
(function () {
  const E = window.ENote;

  const L = 10;   // 層の数
  const S = 50;   // 入力サンプル数
  const NB = 24;  // ヒストグラムの棒の数

  // 再現できる乱数（mulberry32）と標準正規乱数（Box-Muller）
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
  }

  const ACTS = {
    tanh: { label: "tanh", f: (u) => Math.tanh(u), lo: -1, hi: 1 },
    relu: { label: "ReLU", f: (u) => Math.max(0, u), lo: 0, hi: 4 },
    sigmoid: { label: "シグモイド", f: (u) => 1 / (1 + Math.exp(-u)), lo: 0, hi: 1 },
  };
  // 重みの標準偏差（入出力のノード数はどちらも n）
  const INITS = [
    { value: "s001", label: "σ = 0.01", tex: "w\\sim\\mathcal{N}(0,\\,0.01^2)", std: () => 0.01 },
    { value: "s1", label: "σ = 1", tex: "w\\sim\\mathcal{N}(0,\\,1^2)", std: () => 1 },
    { value: "xavier", label: "Xavier", tex: "\\sigma^2=\\dfrac{2}{n_{\\mathrm{in}}+n_{\\mathrm{out}}}", std: (n) => Math.sqrt(2 / (n + n)) },
    { value: "he", label: "He", tex: "\\sigma^2=\\dfrac{2}{n_{\\mathrm{in}}}", std: (n) => Math.sqrt(2 / n) },
  ];

  const cache = {};
  const unit = {}; // n ごとの標準正規乱数（スライダーで σ を変えても乱数は同じ）

  function unitNormals(n) {
    if (unit[n]) return unit[n];
    const g = rng(20240 + n);
    const X = new Float64Array(S * n).map(() => g());
    const W = [];
    for (let l = 0; l < L; l++) W.push(new Float64Array(n * n).map(() => g()));
    return (unit[n] = { X, W });
  }

  // 戻り値：init ごとの { rms: [L+1 個], z: [L+1 個の配列] }
  function compute(n, actKey) {
    const key = n + actKey;
    if (cache[key]) return cache[key];
    const act = ACTS[actKey].f;
    const { X, W } = unitNormals(n);
    const out = {};
    INITS.forEach((init) => {
      const sd = init.std(n);
      let z = X;
      const zs = [z], rms = [Math.sqrt(z.reduce((s, v) => s + v * v, 0) / z.length)];
      for (let l = 0; l < L; l++) {
        const w = W[l];
        const nz = new Float64Array(S * n);
        for (let s = 0; s < S; s++) {
          for (let j = 0; j < n; j++) {
            let u = 0;
            for (let i = 0; i < n; i++) u += z[s * n + i] * w[i * n + j];
            nz[s * n + j] = act(u * sd);
          }
        }
        z = nz;
        zs.push(z);
        rms.push(Math.sqrt(z.reduce((s, v) => s + v * v, 0) / z.length));
      }
      out[init.value] = { rms, z: zs };
    });
    return (cache[key] = out);
  }
  E._initSim = compute; // 検算用

  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c]).join("");

  // 大きな値も小さな値も指数表記で表示する
  const sci = (v) => {
    if (v === 0) return "0";
    if (v >= 0.001 && v < 1000) return E.fmt(v, 3);
    const [m, e] = v.toExponential(2).split("e");
    return m + "\\times 10^{" + Number(e) + "}";
  };

  // 左：層ごとの RMS（対数目盛）
  const LX0 = 60, LX1 = 300, Y0 = 34, Y1 = 220;
  const EMIN = -8, EMAX = 2;
  const lx = (l) => LX0 + (l / L) * (LX1 - LX0);
  const ly = (v) => {
    const e = v > 0 ? Math.log10(v) : EMIN;
    return Y1 - ((Math.max(EMIN, Math.min(EMAX, e)) - EMIN) / (EMAX - EMIN)) * (Y1 - Y0);
  };
  // 右：層 l の出力の分布
  const HX0 = 380, HX1 = 610;

  E.register("init-variance", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const tAct = E.toggle(top, Object.entries(ACTS).map(([value, a]) => ({ value, label: a.label })), "tanh", () => draw());
    const top2 = E.h("div", { class: "ew-bar" });
    el.append(top2);
    const tInit = E.toggle(top2, INITS.map((i) => ({ value: i.value, label: i.label })), "xavier", () => draw());
    const svg = E.svg(el, 640, 290, "10層のネットワークで、初期化の違いによる層ごとの出力の大きさと分布を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "n", label: "n", min: 20, max: 150, step: 10, value: 100, digits: 0 },
        { key: "l", label: "l", min: 1, max: L, step: 1, value: 5, digits: 0 },
      ],
      () => draw()
    );

    function draw() {
      const actKey = tAct.get(), initKey = tInit.get();
      const { n, l } = sl.get();
      const act = ACTS[actKey];
      const f = E.fmt;
      const res = compute(n, actKey);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 左の枠
      svg.append(E.s("text", { x: LX0, y: 14, class: "ew-s ew-start" }, "層ごとの出力の大きさ（RMS）"));
      [2, 0, -2, -4, -6, -8].forEach((e) => {
        const y = ly(Math.pow(10, e));
        svg.append(E.s("line", { x1: LX0, y1: y, x2: LX1, y2: y, class: "ew-grid" }));
        svg.append(E.s("text", { x: LX0 - 8, y, class: "ew-s ew-end ew-muted" }, e === 0 ? "1" : "10" + sup(e)));
      });
      [0, 2, 4, 6, 8, 10].forEach((v) => {
        svg.append(E.s("text", { x: lx(v), y: Y1 + 14, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      svg.append(E.s("text", { x: (LX0 + LX1) / 2, y: Y1 + 34, class: "ew-s ew-c" }, "層の番号 l"));
      // 入力の大きさ 1 の基準線
      svg.append(E.s("line", { x1: LX0, y1: ly(1), x2: LX1, y2: ly(1), class: "ew-edge", "stroke-dasharray": "3 3" }));

      // 折れ線：ほかの初期化はグレー、選んだものは紺
      INITS.forEach((init) => {
        if (init.value === initKey) return;
        const p = res[init.value].rms.map((v, i) => lx(i).toFixed(1) + "," + ly(v).toFixed(1)).join(" ");
        svg.append(E.s("polyline", { points: p, fill: "none", class: "ew-edge" }));
      });
      const sel = res[initKey];
      svg.append(E.s("polyline", {
        "data-k": "line", fill: "none", style: "stroke:var(--main);stroke-width:1.8",
        points: sel.rms.map((v, i) => lx(i).toFixed(1) + "," + ly(v).toFixed(1)).join(" "),
      }));
      svg.append(E.s("circle", { "data-k": "cur", cx: lx(l), cy: ly(sel.rms[l]), r: 4.5, class: "ew-dot-d" }));

      // 右：層 l の出力の分布（高さは最大の棒を基準）
      svg.append(E.s("text", { x: HX0, y: 14, class: "ew-s ew-start" }, "層 l の出力 z の分布"));
      const cnt = new Array(NB).fill(0);
      const z = sel.z[l];
      for (let i = 0; i < z.length; i++) {
        const b = Math.floor(((z[i] - act.lo) / (act.hi - act.lo)) * NB);
        cnt[Math.max(0, Math.min(NB - 1, b))]++;
      }
      const cmax = Math.max(...cnt);
      const bw = (HX1 - HX0) / NB;
      svg.append(E.s("line", { x1: HX0, y1: Y1, x2: HX1, y2: Y1, class: "ew-edge" }));
      cnt.forEach((c, i) => {
        const h = (c / cmax) * (Y1 - Y0 - 6);
        svg.append(E.s("rect", {
          "data-k": "hist", x: HX0 + i * bw + 1, y: Y1 - h, width: bw - 2, height: Math.max(h, 0), class: "ew-bar-0",
        }));
      });
      const lo = String(act.lo), hi = actKey === "relu" ? "4 以上" : String(act.hi);
      svg.append(E.s("text", { x: HX0, y: Y1 + 14, class: "ew-s ew-c ew-muted" }, lo));
      svg.append(E.s("text", { x: HX1, y: Y1 + 14, class: "ew-s ew-c ew-muted" }, hi));
      svg.append(E.s("text", { x: (HX0 + HX1) / 2, y: Y1 + 34, class: "ew-s ew-c" }, "z の値"));
      svg.append(E.s("text", { x: LX0, y: 276, class: "ew-s ew-start" }, "紺：選んだ初期化　グレー：ほかの初期化　点線：入力の大きさ 1"));

      const init = INITS.find((i) => i.value === initKey);
      E.formulaRows(rows, svg, [
        { tex: init.tex, num: "n=" + n + ",\\ \\sigma=" + f(init.std(n), 3), hl: [] },
        {
          tex: "\\mathrm{RMS}=\\sqrt{\\overline{\\bigl(z^{(l)}\\bigr)^2}}",
          num: "l=" + l + ":\\ " + sci(sel.rms[l]),
          hl: ["line", "cur"],
        },
        { tex: "z^{(l)}\\ \\text{の分布}", num: S * n + "\\ \\text{個の}\\ z\\ \\text{の度数}", hl: ["hist"] },
      ]);
    }

    draw();
  });
})();
