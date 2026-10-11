// 情報理論：KL・交差エントロピー・JS（info-kl）、相互情報量とエントロピーの関係（info-mi）
// 単位はビット（底 2）。機械学習では自然対数を使うことが多いが、大小関係は変わらない。
(function () {
  const E = window.ENote;
  const lg = (x) => (x > 0 ? Math.log2(x) : 0);
  const xlogx = (x) => (x > 0 ? x * Math.log2(x) : 0);
  const norm = (a) => {
    const s = a.reduce((t, v) => t + v, 0);
    return a.map((v) => v / s);
  };
  const H = (p) => -p.reduce((s, v) => s + xlogx(v), 0);
  const CE = (p, q) => -p.reduce((s, v, i) => s + (v > 0 ? v * lg(q[i]) : 0), 0);
  const KL = (p, q) => CE(p, q) - H(p);

  // ---------- 2つの分布 P, Q の比較 ----------
  E.register("info-kl", (el) => {
    const svg = E.svg(el, 640, 190, "3つの値をとる分布 P と Q を並べた棒グラフ。M は P と Q の平均");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "p1", label: "P_1", min: 0.02, max: 1, step: 0.01, value: 0.5 },
        { key: "p2", label: "P_2", min: 0.02, max: 1, step: 0.01, value: 0.3 },
        { key: "p3", label: "P_3", min: 0.02, max: 1, step: 0.01, value: 0.2 },
        { key: "q1", label: "Q_1", min: 0.02, max: 1, step: 0.01, value: 0.25 },
        { key: "q2", label: "Q_2", min: 0.02, max: 1, step: 0.01, value: 0.25 },
        { key: "q3", label: "Q_3", min: 0.02, max: 1, step: 0.01, value: 0.5 },
      ],
      () => draw()
    );

    const BASE = 150, HT = 110, GX = [110, 270, 430], BW = 48;

    function draw() {
      const v = sl.get();
      const P = norm([v.p1, v.p2, v.p3]);
      const Q = norm([v.q1, v.q2, v.q3]);
      const M = P.map((x, i) => (x + Q[i]) / 2);
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [0, 0.5, 1].forEach((g) => {
        const y = BASE - HT * g;
        svg.append(E.s("line", { x1: 60, y1: y, x2: 560, y2: y, class: "ew-grid" }));
        svg.append(E.s("text", { x: 52, y, class: "ew-s ew-end ew-muted" }, String(g)));
      });
      for (let i = 0; i < 3; i++) {
        const x = GX[i];
        const hp = HT * P[i], hq = HT * Q[i];
        svg.append(E.s("rect", { "data-k": "P", x, y: BASE - hp, width: BW, height: hp, rx: 2, class: "ew-bar-0" }));
        svg.append(E.s("rect", { "data-k": "Q", x: x + BW + 6, y: BASE - hq, width: BW, height: hq, rx: 2, class: "ew-bar-1" }));
        svg.append(E.s("line", { "data-k": "M", x1: x + 2 * BW + 14, y1: BASE - HT * M[i], x2: x + 2 * BW + 34, y2: BASE - HT * M[i], class: "ew-edge", "stroke-width": 2 }));
        svg.append(E.s("text", { x: x + BW / 2, y: BASE - hp - 10, class: "ew-s ew-c" }, f(P[i], 2)));
        svg.append(E.s("text", { x: x + BW + 6 + BW / 2, y: BASE - hq - 10, class: "ew-s ew-c" }, f(Q[i], 2)));
        svg.append(E.s("text", { x: x + BW + 3, y: BASE + 16, class: "ew-s ew-c ew-muted" }, "x = " + (i + 1)));
      }
      // 凡例
      svg.append(E.s("text", { x: 580, y: 24, class: "ew-s ew-start" }, "左：P"));
      svg.append(E.s("text", { x: 580, y: 44, class: "ew-s ew-start" }, "右：Q"));
      svg.append(E.s("text", { x: 580, y: 64, class: "ew-s ew-start" }, "短線：M"));

      const hP = H(P), ce = CE(P, Q), klpq = KL(P, Q), klqp = KL(Q, P);
      const js = (KL(P, M) + KL(Q, M)) / 2;
      E.formulaRows(rows, svg, [
        { tex: "H(P)=-\\sum_x P(x)\\log_2 P(x)", num: "=" + f(hP), hl: ["P"] },
        { tex: "H(P,Q)=-\\sum_x P(x)\\log_2 Q(x)", num: "=" + f(ce), hl: ["P", "Q"] },
        { tex: "D_{\\mathrm{KL}}(P\\|Q)=H(P,Q)-H(P)", num: "=" + f(ce) + "-" + f(hP) + "=" + f(klpq), hl: ["P", "Q"] },
        { tex: "D_{\\mathrm{KL}}(Q\\|P)", num: "=" + f(klqp), hl: ["P", "Q"] },
        { tex: "D_{\\mathrm{JS}}=\\tfrac12\\bigl(D_{\\mathrm{KL}}(P\\|M)+D_{\\mathrm{KL}}(Q\\|M)\\bigr)", num: "=" + f(js), hl: ["P", "Q", "M"] },
      ]);
    }
    draw();
  });

  // ---------- 相互情報量 ----------
  E.register("info-mi", (el) => {
    const svg = E.svg(el, 640, 190, "H(X) と H(Y) の重なりが相互情報量であることを示す3本の棒");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "px", label: "P(X{=}1)", min: 0.05, max: 0.95, step: 0.01, value: 0.5 },
        { key: "c", label: "c", min: 0, max: 1, step: 0.01, value: 0.6 },
      ],
      () => draw()
    );

    const X0 = 140, W = 200; // 1 ビットあたりの幅

    function draw() {
      const { px, c } = sl.get();
      const f = E.fmt;
      const pm = [1 - px, px];
      // 結合分布：独立な分布と「X=Y」の分布の混合
      const J = [0, 1].map((x) => [0, 1].map((y) => (1 - c) * pm[x] * pm[y] + (x === y ? c * pm[x] : 0)));
      const hx = H(pm);
      const hxy = -J.flat().reduce((s, v) => s + xlogx(v), 0);
      let mi = 0;
      for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) if (J[x][y] > 0) mi += J[x][y] * lg(J[x][y] / (pm[x] * pm[y]));
      const hxGy = hx - mi; // H(X|Y)
      const hyGx = hx - mi; // H(Y|X)（周辺分布が同じなので等しい）
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [0, 1, 2].forEach((b) => {
        const x = X0 + W * b;
        svg.append(E.s("line", { x1: x, y1: 14, x2: x, y2: 150, class: "ew-grid" }));
        svg.append(E.s("text", { x, y: 168, class: "ew-s ew-c ew-muted" }, b + " bit"));
      });
      const seg = (x, w, y, k, cls) =>
        svg.append(E.s("rect", { "data-k": k, x: X0 + W * x, y, width: Math.max(W * w - 2, 0.5), height: 24, rx: 3, class: cls }));
      const label = (text, y) => svg.append(E.s("text", { x: X0 - 12, y: y + 12, class: "ew-t ew-end" }, text));
      const val = (x, y, v) => svg.append(E.s("text", { x: X0 + W * x + 8, y: y + 12, class: "ew-s ew-start" }, f(v)));

      // H(X) = H(X|Y) + I
      label("H(X)", 24);
      seg(0, hxGy, 24, "hxgy", "ew-bar-0");
      seg(hxGy, mi, 24, "mi", "ew-bar-1");
      val(hx, 24, hx);
      // H(X,Y) = H(X|Y) + I + H(Y|X)
      label("H(X,Y)", 62);
      seg(0, hxGy, 62, "hxgy", "ew-bar-0");
      seg(hxGy, mi, 62, "mi", "ew-bar-1");
      seg(hxGy + mi, hyGx, 62, "hygx", "ew-bar-2");
      val(hxy, 62, hxy);
      // H(Y) = I + H(Y|X)（H(X,Y) の右側に揃える）
      label("H(Y)", 100);
      seg(hxGy, mi, 100, "mi", "ew-bar-1");
      seg(hxGy + mi, hyGx, 100, "hygx", "ew-bar-2");
      val(hxGy + hx, 100, hx);

      E.formulaRows(rows, svg, [
        { tex: "I(X;Y)=\\sum_{x,y}p(x,y)\\log_2\\frac{p(x,y)}{p(x)p(y)}", num: "=" + f(mi), hl: ["mi"] },
        { tex: "I(X;Y)=H(X)-H(X\\mid Y)", num: "=" + f(hx) + "-" + f(hxGy) + "=" + f(hx - hxGy), hl: ["mi", "hxgy"] },
        { tex: "I(X;Y)=H(X)+H(Y)-H(X,Y)", num: "=" + f(2 * hx) + "-" + f(hxy) + "=" + f(2 * hx - hxy), hl: ["mi"] },
        { tex: "H(X\\mid Y)=H(X,Y)-H(Y)", num: "=" + f(hxy) + "-" + f(hx) + "=" + f(hxy - hx), hl: ["hxgy"] },
      ]);
    }
    draw();
  });
})();
