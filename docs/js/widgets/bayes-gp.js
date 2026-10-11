// ガウス過程回帰：RBF カーネルの長さ尺度 ℓ とノイズ σ_n を変え、予測の平均と不確かさ、周辺尤度を見る
(function () {
  // ---- 計算 ----
  const XS = [1.0, 2.2, 4.0, 5.1, 7.0, 8.8];
  const YS = [0.6, 1.1, -0.2, -0.9, 0.5, 1.0];
  const SF2 = 1; // 信号の分散 σ_f²（固定）
  const kern = (a, b, ell) => SF2 * Math.exp(-((a - b) ** 2) / (2 * ell * ell));

  // コレスキー分解 A = L Lᵀ
  function chol(A) {
    const n = A.length, L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
      }
    }
    return L;
  }
  const fwd = (L, b) => { // L v = b
    const v = [];
    for (let i = 0; i < b.length; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * v[k];
      v.push(s / L[i][i]);
    }
    return v;
  };
  const bwd = (L, b) => { // Lᵀ x = b
    const n = b.length, x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  };
  function model(ell, sn) {
    const n = XS.length;
    const K = XS.map((a, i) => XS.map((b, j) => kern(a, b, ell) + (i === j ? sn * sn : 0)));
    const L = chol(K);
    const alpha = bwd(L, fwd(L, YS));
    const fit = -0.5 * YS.reduce((s, y, i) => s + y * alpha[i], 0);
    const cplx = -L.reduce((s, row, i) => s + Math.log(row[i]), 0); // -½ log|K| = -Σ log L_ii
    const c = -0.5 * n * Math.log(2 * Math.PI);
    return {
      fit, cplx, c, lml: fit + cplx + c,
      at(x) {
        const ks = XS.map((a) => kern(a, x, ell));
        const v = fwd(L, ks);
        return {
          mean: ks.reduce((s, k, i) => s + k * alpha[i], 0),
          varf: Math.max(0, SF2 - v.reduce((s, t) => s + t * t, 0)),
        };
      },
    };
  }
  if (typeof module !== "undefined") module.exports = { model, XS, YS };
  if (typeof window === "undefined") return;

  // ---- 描画 ----
  const E = window.ENote;
  const X0 = 50, X1 = 610, Y0 = 20, Y1 = 250;
  const sx = (x) => X0 + (x / 10) * (X1 - X0);
  const sy = (y) => Y1 - ((Math.min(3, Math.max(-3, y)) + 3) / 6) * (Y1 - Y0);

  E.register("bayes-gp", (el) => {
    const svg = E.svg(el, 640, 290, "ガウス過程回帰の予測平均と95%区間を、観測点とともに示すグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "ell", label: "\\ell", min: 0.3, max: 4, step: 0.05, value: 1.2 },
        { key: "sn", label: "\\sigma_n", min: 0.02, max: 1, step: 0.01, value: 0.15 },
        { key: "xs", label: "x_*", min: 0, max: 10, step: 0.1, value: 3.1 },
      ],
      () => draw()
    );

    function draw() {
      const { ell, sn, xs } = sl.get();
      const f = E.fmt;
      const M = model(ell, sn);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [0, 2, 4, 6, 8, 10].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c" }, String(v)));
      });
      [-2, 0, 2].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end" }, String(v)));
      });

      // 95% 区間（±2σ）の帯と平均
      const up = [], lo = [], mid = [];
      for (let i = 0; i <= 140; i++) {
        const x = (i / 140) * 10, p = M.at(x), sd = Math.sqrt(p.varf);
        up.push(sx(x).toFixed(1) + "," + sy(p.mean + 2 * sd).toFixed(1));
        lo.unshift(sx(x).toFixed(1) + "," + sy(p.mean - 2 * sd).toFixed(1));
        mid.push(sx(x).toFixed(1) + "," + sy(p.mean).toFixed(1));
      }
      svg.append(E.s("polygon", { "data-k": "band", points: up.concat(lo).join(" "), style: "fill:var(--line);fill-opacity:0.7;stroke:none" }));
      svg.append(E.s("polyline", { "data-k": "mean", points: mid.join(" "), class: "ew-curve-f", style: "stroke:var(--main)" }));
      XS.forEach((x, i) => svg.append(E.s("circle", { cx: sx(x), cy: sy(YS[i]), r: 4.5, class: "ew-dot-f" })));

      // 予測したい点 x*
      const p = M.at(xs), sd = Math.sqrt(p.varf);
      svg.append(E.s("line", { x1: sx(xs), y1: Y0, x2: sx(xs), y2: Y1, class: "ew-edge", "stroke-dasharray": "3 3" }));
      svg.append(E.s("line", { "data-k": "band", x1: sx(xs), y1: sy(p.mean - 2 * sd), x2: sx(xs), y2: sy(p.mean + 2 * sd), class: "ew-edge", style: "stroke:var(--main);stroke-width:2" }));
      svg.append(E.s("circle", { "data-k": "mean", cx: sx(xs), cy: sy(p.mean), r: 5, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: X0, y: 8, class: "ew-s ew-start" }, "●：観測データ　紺：予測平均　帯：平均 ± 2σ"));

      E.formulaRows(rows, svg, [
        { tex: "\\mu_*=\\mathbf{k}_*^{\\top}(\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y}", num: "=" + f(p.mean), hl: ["mean"] },
        { tex: "\\sigma_*^2=k(x_*,x_*)-\\mathbf{k}_*^{\\top}(\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{k}_*", num: "=" + f(p.varf) + "\\quad(\\sigma_*=" + f(sd) + ")", hl: ["band"] },
        {
          tex: "\\log p(\\mathbf{y}\\mid\\mathbf{X})=\\underbrace{-\\tfrac12\\mathbf{y}^{\\top}(\\cdot)^{-1}\\mathbf{y}}_{\\text{データへの適合}}\\underbrace{-\\tfrac12\\log|\\cdot|}_{\\text{複雑さ}}-\\tfrac{N}{2}\\log2\\pi",
          num: "=" + f(M.fit, 2) + "\\ " + (M.cplx < 0 ? "-" : "+") + "\\ " + f(Math.abs(M.cplx), 2) + "\\ -\\ " + f(-M.c, 2) + "\\ =\\ " + f(M.lml, 2),
          hl: [],
        },
      ]);
    }
    draw();
  });
})();
