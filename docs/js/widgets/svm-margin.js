// SVM：マージン・サポートベクター・ソフトマージン（C）・カーネル（RBF）
// 双対問題を SMO 法（2 つの α を同時に動かす）で解き、結果を図と数式の行に出す
(function () {
  const E = window.ENote;

  const SEP_P = [[1.0, 0.6], [1.8, 1.6], [2.4, 0.2], [3.0, 1.4], [3.6, -0.4], [2.0, -1.0], [1.4, -1.8]];
  const SEP_N = [[-1.0, -0.4], [-1.8, 0.8], [-2.4, -0.6], [-3.0, 1.0], [-3.6, -1.2], [-2.0, -1.6], [-1.5, 1.9]];
  const ring = (n, rx, ry, off, jit) =>
    Array.from({ length: n }, (_, k) => {
      const a = off + (2 * Math.PI * k) / n, j = 1 + jit[k % jit.length];
      return [rx * j * Math.cos(a), ry * j * Math.sin(a)];
    });
  const DATASETS = {
    sep: { P: SEP_P, N: SEP_N, kernel: "lin" },
    out: { P: SEP_P.concat([[-1.6, 0.2]]), N: SEP_N, kernel: "lin" },
    ringlin: { P: ring(8, 1.0, 1.0, 0.2, [0, 0.12, -0.1, 0.06]), N: ring(12, 3.2, 2.3, 0.3, [0, 0.05, -0.05]), kernel: "lin" },
    ringrbf: { P: ring(8, 1.0, 1.0, 0.2, [0, 0.12, -0.1, 0.06]), N: ring(12, 3.2, 2.3, 0.3, [0, 0.05, -0.05]), kernel: "rbf" },
  };

  const kern = (type, g) =>
    type === "lin"
      ? (a, b) => a[0] * b[0] + a[1] * b[1]
      : (a, b) => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));

  // 簡易 SMO（Platt, 1998 の考え方）。j は |E_i-E_j| が大きい順に試す
  function train(X, t, K, C) {
    const n = X.length;
    const Km = X.map((a) => X.map((b) => K(a, b)));
    const a = new Array(n).fill(0);
    let b = 0;
    const fx = (i) => {
      let s = b;
      for (let k = 0; k < n; k++) if (a[k] !== 0) s += a[k] * t[k] * Km[k][i];
      return s;
    };
    const tol = 1e-7;
    for (let pass = 0; pass < 2000; pass++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = fx(i) - t[i];
        if (!((t[i] * Ei < -tol && a[i] < C - 1e-12) || (t[i] * Ei > tol && a[i] > 1e-12))) continue;
        const order = [...Array(n).keys()].filter((j) => j !== i)
          .sort((p, q) => Math.abs(Ei - (fx(q) - t[q])) - Math.abs(Ei - (fx(p) - t[p])));
        for (const j of order) {
          const Ej = fx(j) - t[j];
          let L, H;
          if (t[i] !== t[j]) { L = Math.max(0, a[j] - a[i]); H = Math.min(C, C + a[j] - a[i]); }
          else { L = Math.max(0, a[i] + a[j] - C); H = Math.min(C, a[i] + a[j]); }
          if (H - L < 1e-12) continue;
          const eta = 2 * Km[i][j] - Km[i][i] - Km[j][j];
          if (eta >= -1e-12) continue;
          const aj = Math.min(H, Math.max(L, a[j] - (t[j] * (Ei - Ej)) / eta));
          if (Math.abs(aj - a[j]) < 1e-9) continue;
          const ai = a[i] + t[i] * t[j] * (a[j] - aj);
          const b1 = b - Ei - t[i] * (ai - a[i]) * Km[i][i] - t[j] * (aj - a[j]) * Km[i][j];
          const b2 = b - Ej - t[i] * (ai - a[i]) * Km[i][j] - t[j] * (aj - a[j]) * Km[j][j];
          a[i] = ai; a[j] = aj;
          b = ai > 1e-12 && ai < C - 1e-12 ? b1 : aj > 1e-12 && aj < C - 1e-12 ? b2 : (b1 + b2) / 2;
          changed++;
          break;
        }
      }
      if (changed === 0) break;
    }
    // b の決め直し：マージン上の点（0<α<C）があれば t_i - f0(x_i) の平均、
    // なければ主問題（ヒンジ損失の和）を最小にする b（最小値の区間の中点）
    {
      const f0 = [];
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) if (a[k] !== 0) s += a[k] * t[k] * Km[k][i];
        f0.push(s);
      }
      const free = [];
      for (let i = 0; i < n; i++) if (a[i] > 1e-6 && a[i] < C - 1e-6) free.push(i);
      if (free.length) b = free.reduce((s, i) => s + t[i] - f0[i], 0) / free.length;
      else {
        const H = (bb) => f0.reduce((s, v, i) => s + Math.max(0, 1 - t[i] * (v + bb)), 0);
        const cand = f0.map((v, i) => t[i] - v);
        const best = Math.min(...cand.map(H));
        const ok = cand.filter((c) => H(c) <= best + 1e-9);
        b = (Math.min(...ok) + Math.max(...ok)) / 2;
      }
    }
    // 双対・主問題の値
    let q = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q += a[i] * a[j] * t[i] * t[j] * Km[i][j];
    let xi = 0;
    for (let i = 0; i < n; i++) xi += Math.max(0, 1 - t[i] * fx(i));
    const sumA = a.reduce((s, v) => s + v, 0);
    const f = (x) => {
      let s = b;
      for (let k = 0; k < n; k++) if (a[k] > 1e-12) s += a[k] * t[k] * K(X[k], x);
      return s;
    };
    return { a, b, f, primal: 0.5 * q + C * xi, dual: sumA - 0.5 * q, q, sumAT: a.reduce((s, v, i) => s + v * t[i], 0) };
  }

  if (typeof module !== "undefined") module.exports = { DATASETS, kern, train };
  if (!E) return;

  const SC = 54, CX = 320, CY = 170;
  const sx = (x) => CX + SC * x;
  const sy = (y) => CY - SC * y;
  const CELL = 9;

  E.register("svm-margin", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "sep", label: "線形・分離可能" }, { value: "out", label: "線形・外れ値あり" },
        { value: "ringlin", label: "円状・線形カーネル" }, { value: "ringrbf", label: "円状・RBFカーネル" }],
      "sep",
      () => draw()
    );
    const svg = E.svg(el, 640, 340, "2クラスのデータに対する SVM の決定境界、マージン、サポートベクターを示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "lc", label: "\\log_{10}C", min: -2, max: 2, step: 0.1, value: 0, digits: 1 },
        { key: "g", label: "\\gamma", min: 0.05, max: 2, step: 0.05, value: 0.5 },
      ],
      () => draw()
    );
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "●＝t=+1、○＝t=−1。実線の輪＝サポートベクター（0<α<C、マージン上）、破線の輪＝α=C（マージン内か誤分類）。背景の濃い領域は f≥1 または f≤−1 で、マージンの外側。γ は RBF のときだけ効く。"));

    const clipId = "ew-clip-" + Math.random().toString(36).slice(2, 8);
    const cp = E.s("clipPath", { id: clipId });
    cp.append(E.s("rect", { x: 23, y: 8, width: 594, height: 324 }));
    svg.childNodes[1].append(cp);

    // 背景の格子
    const cells = [];
    const gbg = E.s("g", {});
    for (let ix = 0; ix < 66; ix++) {
      for (let iy = 0; iy < 36; iy++) {
        const r = E.s("rect", { x: 23 + ix * CELL, y: 8 + iy * CELL, width: CELL, height: CELL, "shape-rendering": "crispEdges" });
        gbg.append(r);
        cells.push({ r, x: (23 + ix * CELL + CELL / 2 - CX) / SC, y: (CY - (8 + iy * CELL + CELL / 2)) / SC });
      }
    }
    svg.append(gbg);
    const dyn = E.s("g", {});
    svg.append(dyn);

    function draw() {
      const key = mode.get();
      const ds = DATASETS[key];
      const { lc, g } = sl.get();
      const C = Math.pow(10, lc);
      const f = E.fmt;
      const X = ds.P.concat(ds.N);
      const t = ds.P.map(() => 1).concat(ds.N.map(() => -1));
      const K = kern(ds.kernel, g);
      const m = train(X, t, K, C);

      cells.forEach((c) => {
        const v = m.f([c.x, c.y]);
        const e = 1e-6; // f がちょうど ±1 の定数になるときの、丸め誤差によるちらつきを防ぐ
        const st = v >= 1 - e ? "fill:var(--main);fill-opacity:.20" : v >= 0 ? "fill:var(--main);fill-opacity:.07"
          : v > -1 + e ? "fill:var(--muted);fill-opacity:.07" : "fill:var(--muted);fill-opacity:.20";
        c.r.setAttribute("style", st);
      });
      dyn.innerHTML = "";

      // 線形カーネルなら w が求まり、f=0、f=±1 を直線で描ける
      let w = null;
      if (ds.kernel === "lin") {
        w = [0, 0];
        X.forEach((x, i) => { w[0] += m.a[i] * t[i] * x[0]; w[1] += m.a[i] * t[i] * x[1]; });
        const line = (c, k, style) => {
          // w·x + b = c
          let p1, p2;
          if (Math.abs(w[1]) > Math.abs(w[0])) {
            const y = (x) => (c - m.b - w[0] * x) / w[1];
            p1 = [-6, y(-6)]; p2 = [6, y(6)];
          } else {
            const x = (y) => (c - m.b - w[1] * y) / w[0];
            p1 = [x(-3.5), -3.5]; p2 = [x(3.5), 3.5];
          }
          dyn.append(E.s("line", { "data-k": k, x1: sx(p1[0]), y1: sy(p1[1]), x2: sx(p2[0]), y2: sy(p2[1]), class: "ew-edge", style, "clip-path": "url(#" + clipId + ")" }));
        };
        if (Math.hypot(w[0], w[1]) > 1e-6) {
          line(0, "line", "stroke:var(--main);stroke-width:2.2");
          line(1, "marg", "stroke:var(--main);stroke-width:1.2;stroke-dasharray:5 3");
          line(-1, "marg", "stroke:var(--main);stroke-width:1.2;stroke-dasharray:5 3");
        }
      }

      let nMargin = 0, nBound = 0;
      X.forEach((x, i) => {
        const cx = sx(x[0]), cy = sy(x[1]);
        if (m.a[i] > 1e-6) {
          if (m.a[i] > C - 1e-6) {
            nBound++;
            dyn.append(E.s("circle", { "data-k": "sv", cx, cy, r: 9.5, class: "ew-edge", style: "stroke:var(--main);stroke-width:1.4;stroke-dasharray:3 2" }));
          } else {
            nMargin++;
            dyn.append(E.s("circle", { "data-k": "sv", cx, cy, r: 9.5, class: "ew-edge", style: "stroke:var(--main);stroke-width:1.8" }));
          }
        }
        if (t[i] === 1) dyn.append(E.s("circle", { "data-k": "pts", cx, cy, r: 5, class: "ew-dot-d" }));
        else dyn.append(E.s("circle", { "data-k": "pts", cx, cy, r: 5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.6" }));
      });

      const rowsDef = [];
      if (w) {
        const wn = Math.hypot(w[0], w[1]);
        rowsDef.push({ tex: "\\mathbf{w}=\\sum_n\\alpha_n t_n\\mathbf{x}_n", num: "\\mathbf{w}=(" + f(w[0]) + ",\\ " + f(w[1]) + "),\\ b=" + f(m.b), hl: ["line", "sv"] });
        rowsDef.push({ tex: "\\text{マージン幅}\\ \\dfrac{2}{\\|\\mathbf{w}\\|}", num: wn > 1e-6 ? "2/" + f(wn) + "=" + f(2 / wn) : "\\mathbf{w}\\approx\\mathbf{0}\\ \\text{（直線では分けられない）}", hl: ["marg"] });
      } else {
        rowsDef.push({ tex: "f(\\mathbf{x})=\\sum_n\\alpha_n t_n K(\\mathbf{x}_n,\\mathbf{x})+b", num: "b=" + f(m.b), hl: ["sv"] });
        rowsDef.push({ tex: "K(\\mathbf{x},\\mathbf{x}')=\\exp\\bigl(-\\gamma\\|\\mathbf{x}-\\mathbf{x}'\\|^2\\bigr)", num: "\\gamma=" + f(g, 2), hl: ["sv"] });
      }
      rowsDef.push({ tex: "\\sum_n\\alpha_n t_n=0", num: f(m.sumAT), hl: ["sv"] });
      rowsDef.push({ tex: "\\text{主問題}\\ \\tfrac12\\|\\mathbf{w}\\|^2+C\\sum_n\\xi_n", num: f(m.primal), hl: ["pts", "sv"] });
      rowsDef.push({ tex: "\\text{双対問題}\\ \\sum_n\\alpha_n-\\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j t_it_jK_{ij}", num: f(m.dual) + "\\ \\text{（最適なら主問題と等しい）}", hl: ["sv"] });
      rowsDef.push({ tex: "0<\\alpha_n<C\\ \\text{／}\\ \\alpha_n=C\\ \\text{の個数}", num: nMargin + "\\ \\text{／}\\ " + nBound + "\\quad (C=" + f(C, 2) + ")", hl: ["sv"] });
      E.formulaRows(rows, svg, rowsDef);
    }

    draw();
  });
})();
