// Mixup / CutMix / Cutout：画素とラベルがどう混ざる（消える）かを見る
(function () {
  const E = window.ENote;
  const N = 10, PX = 14; // 10×10 画素、1画素 14px

  // 画像 A（バツ印）と画像 B（枠）
  const A = [], B = [];
  for (let r = 0; r < N; r++) {
    A.push([]); B.push([]);
    for (let c = 0; c < N; c++) {
      A[r].push(r === c || r + c === N - 1 ? 1 : 0);
      B[r].push(Math.max(Math.abs(r - 4.5), Math.abs(c - 4.5)) === 3.5 ? 1 : 0);
    }
  }

  E.register("aug-mix", (el) => {
    const mode = E.toggle(
      el,
      [{ value: "mixup", label: "Mixup" }, { value: "cutmix", label: "CutMix" }, { value: "cutout", label: "Cutout" }],
      "mixup",
      () => draw()
    );
    const svg = E.svg(el, 640, 252, "画像Aと画像Bから、Mixup・CutMix・Cutout で作った画像とラベルの比較");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(el, [{ key: "lam", label: "\\lambda", min: 0.05, max: 0.95, step: 0.05, value: 0.6 }], () => draw());

    const PXS = [40, 250, 460], PY = 40;

    function panel(x0, key, title, grid, dim) {
      const g = E.s("g", dim ? { opacity: 0.35 } : {});
      g.append(E.s("rect", { "data-k": key, x: x0 - 1, y: PY - 1, width: N * PX + 2, height: N * PX + 2, class: "ew-node" }));
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          if (grid[r][c] <= 0) continue;
          g.append(E.s("rect", { x: x0 + c * PX, y: PY + r * PX, width: PX, height: PX, class: "ew-back", "fill-opacity": grid[r][c] }));
        }
      }
      svg.append(g);
      svg.append(E.s("text", { x: x0 + (N * PX) / 2, y: 26, class: "ew-s ew-c" }, title));
    }

    function draw() {
      const m = mode.get();
      const { lam } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 結果画像とラベル
      let R = [], y1, y2, side = 0, r0 = 0, tex, num;
      if (m === "mixup") {
        R = A.map((row, r) => row.map((v, c) => lam * v + (1 - lam) * B[r][c]));
        y1 = lam; y2 = 1 - lam;
      } else {
        side = Math.round(N * Math.sqrt(1 - lam)); // 貼る（消す）正方形の一辺
        r0 = Math.floor((N - side) / 2);
        const inBox = (r, c) => r >= r0 && r < r0 + side && c >= r0 && c < r0 + side;
        if (m === "cutmix") {
          R = A.map((row, r) => row.map((v, c) => (inBox(r, c) ? B[r][c] : v)));
          y1 = 1 - (side * side) / (N * N); y2 = 1 - y1;
        } else {
          R = A.map((row, r) => row.map((v, c) => (inBox(r, c) ? 0 : v)));
          y1 = 1; y2 = 0;
        }
      }

      panel(PXS[0], "A", "画像 A（クラス1）", A, false);
      panel(PXS[1], "B", m === "cutout" ? "画像 B（使わない）" : "画像 B（クラス2）", B, m === "cutout");
      panel(PXS[2], "R", "作った画像", R, false);
      svg.append(E.s("text", { x: 215, y: PY + (N * PX) / 2 + 5, class: "ew-t ew-c" }, m === "mixup" ? "+" : "→"));
      svg.append(E.s("text", { x: 425, y: PY + (N * PX) / 2 + 5, class: "ew-t ew-c" }, "⇒"));
      if (m !== "mixup" && side > 0) {
        const keyPanel = m === "cutmix" ? PXS[1] : PXS[0];
        svg.append(E.s("rect", { x: keyPanel + r0 * PX, y: PY + r0 * PX, width: side * PX, height: side * PX, class: "ew-curve-d" }));
      }

      // ラベルの棒
      [["クラス1", y1, "y1"], ["クラス2", y2, "y2"]].forEach(([name, v, k], idx) => {
        const y = 196 + idx * 26;
        svg.append(E.s("text", { x: PXS[2] + 40, y: y + 12, class: "ew-s ew-end" }, name));
        svg.append(E.s("rect", { "data-k": "y", x: PXS[2] + 48, y, width: Math.max(80 * v, 1), height: 16, rx: 3, class: "ew-bar-" + idx }));
        svg.append(E.s("text", { x: PXS[2] + 48 + 80 * v + 6, y: y + 12, class: "ew-s ew-start" }, f(v, 2)));
      });
      svg.append(E.s("text", { x: PXS[2] + 40, y: 188, class: "ew-s ew-end ew-muted" }, "ラベル"));

      if (m === "mixup") {
        tex = ["\\tilde{\\mathbf{x}}=\\lambda\\,\\mathbf{x}_A+(1-\\lambda)\\,\\mathbf{x}_B", "\\tilde{\\mathbf{y}}=\\lambda\\,\\mathbf{y}_A+(1-\\lambda)\\,\\mathbf{y}_B"];
        num = ["=" + f(lam, 2) + "\\,\\mathbf{x}_A+" + f(1 - lam, 2) + "\\,\\mathbf{x}_B", "=(" + f(y1, 2) + ",\\ " + f(y2, 2) + ")"];
      } else if (m === "cutmix") {
        tex = ["\\tilde{\\mathbf{x}}=\\mathbf{M}\\odot\\mathbf{x}_A+(\\mathbf{1}-\\mathbf{M})\\odot\\mathbf{x}_B", "\\tilde{\\lambda}=1-\\dfrac{\\text{貼った面積}}{WH},\\ \\ \\tilde{\\mathbf{y}}=\\tilde\\lambda\\,\\mathbf{y}_A+(1-\\tilde\\lambda)\\,\\mathbf{y}_B"];
        num = ["\\text{一辺 }" + side + "\\text{ 画素の正方形を B から貼る}", "\\tilde\\lambda=1-\\dfrac{" + side * side + "}{100}=" + f(y1, 2)];
      } else {
        tex = ["\\tilde{\\mathbf{x}}=\\mathbf{M}\\odot\\mathbf{x}_A", "\\tilde{\\mathbf{y}}=\\mathbf{y}_A"];
        num = ["\\text{一辺 }" + side + "\\text{ 画素の正方形を 0 にする}", "=(1,\\ 0)\\ \\text{（ラベルは変えない）}"];
      }
      const hlx = m === "mixup" ? ["A", "B", "R"] : m === "cutmix" ? ["A", "B", "R"] : ["A", "R"];
      E.formulaRows(rows, svg, [
        { tex: tex[0], num: num[0], hl: hlx },
        { tex: tex[1], num: num[1], hl: ["y"] },
      ]);
    }
    draw();
  });
})();
