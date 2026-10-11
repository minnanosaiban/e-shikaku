// 低ランク近似：特異値を大きい順に k 個だけ残して、12×12 の行列（濃淡の画像）を近似する
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 片側ヤコビ法による特異値分解。A（m×n、m≥n）= U diag(s) V^T。s は大きい順
  E.math.svdJacobi = function (A) {
    const m = A.length, n = A[0].length;
    const U = A.map((r) => r.slice());
    const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let rotated = false;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          let al = 0, be = 0, ga = 0;
          for (let i = 0; i < m; i++) { al += U[i][p] * U[i][p]; be += U[i][q] * U[i][q]; ga += U[i][p] * U[i][q]; }
          if (Math.abs(ga) <= 1e-14 * Math.sqrt(al * be) || ga === 0) continue;
          rotated = true;
          const zeta = (be - al) / (2 * ga);
          const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (let i = 0; i < m; i++) {
            const up = U[i][p], uq = U[i][q];
            U[i][p] = c * up - s * uq; U[i][q] = s * up + c * uq;
          }
          for (let i = 0; i < n; i++) {
            const vp = V[i][p], vq = V[i][q];
            V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq;
          }
        }
      }
      if (!rotated) break;
    }
    const sv = [];
    for (let j = 0; j < n; j++) {
      let nr = 0;
      for (let i = 0; i < m; i++) nr += U[i][j] * U[i][j];
      sv.push(Math.sqrt(nr));
    }
    const order = sv.map((_, j) => j).sort((x, y) => sv[y] - sv[x]);
    return {
      s: order.map((j) => sv[j]),
      U: U.map((row) => order.map((j) => (sv[j] > 1e-12 ? row[j] / sv[j] : 0))),
      V: V.map((row) => order.map((j) => row[j])),
    };
  };

  // ランク k の近似 A_k = Σ_{i<k} σ_i u_i v_i^T
  E.math.lowrank = function (svd, k) {
    const m = svd.U.length, n = svd.V.length;
    const R = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let r = 0; r < k; r++)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) R[i][j] += svd.s[r] * svd.U[i][r] * svd.V[j][r];
    return R;
  };

  // 例にする 12×12 の濃淡：中央の丸み＋左右のなだらかな傾き＋細かい模様
  const N = 12;
  E.math.sampleImage = function () {
    return Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => {
        const y = i / (N - 1), x = j / (N - 1);
        const blob = 0.55 * Math.exp(-((x - 0.5) ** 2 + (y - 0.45) ** 2) / 0.06);
        const ramp = 0.22 * x;
        const fine = 0.07 * Math.sin(7.3 * i + 3.1 * j) * Math.cos(4.1 * i - 2.2 * j);
        return Math.min(1, Math.max(0, 0.1 + blob + ramp + fine));
      })
    );
  };

  const CELL = 14;
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  E.register("eigen-svd-lowrank", (el) => {
    const svg = E.svg(el, 640, 250, "12×12 の濃淡の行列を、特異値の大きい順に k 個だけ残して近似する図。右は特異値の大きさ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(el, [{ key: "k", label: "k", min: 1, max: N, step: 1, value: 2, digits: 0 }], () => draw());

    const A = E.math.sampleImage();
    const svd = E.math.svdJacobi(A);
    const total = svd.s.reduce((s, v) => s + v * v, 0);

    function heat(M, x0, y0, key) {
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const attrs = { x: x0 + j * CELL, y: y0 + i * CELL, width: CELL, height: CELL, class: "ew-dot-f", "fill-opacity": clamp01(M[i][j]).toFixed(3) };
          if (key) attrs["data-k"] = key;
          svg.append(E.s("rect", attrs));
        }
      svg.append(E.s("rect", { x: x0, y: y0, width: N * CELL, height: N * CELL, class: "ew-edge" }));
    }

    function draw() {
      const k = sl.get().k;
      const f = E.fmt;
      const Ak = E.math.lowrank(svd, k);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      const GY = 28;
      heat(A, 20, GY, null);
      heat(Ak, 222, GY, "mid");
      svg.append(E.s("text", { x: 20 + (N * CELL) / 2, y: 232, class: "ew-s ew-c" }, "A（12×12）"));
      svg.append(E.s("text", { x: 222 + (N * CELL) / 2, y: 232, class: "ew-s ew-c" }, "近似 A_k（k=" + k + "）"));
      svg.append(E.s("text", { x: 520, y: 232, class: "ew-s ew-c" }, "特異値 σ_i"));

      // 特異値の棒グラフ（残す分だけ紺）
      const BX = 430, BASE = GY + N * CELL, H = 150;
      svg.append(E.s("line", { x1: BX - 4, y1: BASE, x2: BX + N * 15 + 1, y2: BASE, class: "ew-edge" }));
      svg.append(E.s("text", { x: BX + 6, y: BASE + 14, class: "ew-s ew-c ew-muted" }, "1"));
      svg.append(E.s("text", { x: BX + (N - 1) * 15 + 6, y: BASE + 14, class: "ew-s ew-c ew-muted" }, String(N)));
      svd.s.forEach((v, i) => {
        const h = Math.max((v / svd.s[0]) * H, 1);
        const attrs = { x: BX + i * 15, y: BASE - h, width: 12, height: h, rx: 2, class: "ew-bar-0", "data-k": i < k ? "kept" : "cut" };
        if (i < k) attrs.style = "fill:var(--main)";
        svg.append(E.s("rect", attrs));
      });

      let tail = 0, kept = 0, direct = 0;
      svd.s.forEach((v, i) => { if (i < k) kept += v * v; else tail += v * v; });
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) direct += (A[i][j] - Ak[i][j]) ** 2;
      E.formulaRows(rows, svg, [
        { tex: "A_k=\\sum_{i=1}^{k}\\sigma_i\\mathbf{u}_i\\mathbf{v}_i^{\\top}", num: "k=" + k + ",\\ \\sigma_1=" + f(svd.s[0], 2) + ",\\ \\sigma_{" + k + "}=" + f(svd.s[k - 1], 3), hl: ["kept", "mid"] },
        { tex: "\\|A-A_k\\|_F=\\sqrt{\\sum_{i>k}\\sigma_i^2}", num: "=" + f(Math.sqrt(tail), 3) + "\\ (\\text{直接計算 }" + f(Math.sqrt(direct), 3) + ")", hl: ["cut"] },
        { tex: "\\text{累積寄与率}=\\dfrac{\\sum_{i\\le k}\\sigma_i^2}{\\sum_i\\sigma_i^2}", num: "=" + f(kept / total, 4), hl: ["kept"] },
        { tex: "\\text{保存する数 }k(m+n+1)\\ \\text{と}\\ mn", num: k + "\\cdot25=" + k * 25 + "\\ \\text{と}\\ 144", hl: ["kept", "mid"] },
      ]);
    }
    draw();
  });
})();
