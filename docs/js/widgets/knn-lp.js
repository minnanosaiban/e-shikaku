// 距離の形：Lp 距離・マハラノビス距離の「距離 1 の点の集まり」と、コサイン距離の角度
(function () {
  const E = window.ENote;

  const lp = (x1, x2, p) => Math.pow(Math.pow(Math.abs(x1), p) + Math.pow(Math.abs(x2), p), 1 / p);
  // 共分散行列 Σ=[[1,ρ],[ρ,1]] のマハラノビス距離（原点から）
  const maha = (x1, x2, rho) => Math.sqrt((x1 * x1 - 2 * rho * x1 * x2 + x2 * x2) / (1 - rho * rho));
  // r=(rx,ry) と x のコサイン距離。零ベクトルがあると定義できない
  const cosDist = (x1, x2, rx, ry) => {
    const n = Math.hypot(x1, x2) * Math.hypot(rx, ry);
    return n === 0 ? NaN : 1 - (x1 * rx + x2 * ry) / n;
  };
  // 距離 1 の曲線上の点（角度 θ の方向）
  const ballLp = (th, p) => {
    const c = Math.cos(th), s = Math.sin(th);
    const r = 1 / Math.pow(Math.pow(Math.abs(c), p) + Math.pow(Math.abs(s), p), 1 / p);
    return [r * c, r * s];
  };
  const ballMaha = (th, rho) => {
    const a = Math.cos(th) * Math.sqrt(1 + rho), b = Math.sin(th) * Math.sqrt(1 - rho);
    return [(a + b) / Math.SQRT2, (a - b) / Math.SQRT2];
  };

  if (typeof module !== "undefined") module.exports = { lp, maha, cosDist, ballLp, ballMaha };
  if (!E) return;

  const SC = 62, CX = 320, CY = 150;
  const sx = (x) => CX + SC * x;
  const sy = (y) => CY - SC * y;
  const R_VEC = [1.6, 0.5]; // コサイン距離の基準ベクトル

  E.register("knn-lp", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "lp", label: "Lp距離" }, { value: "maha", label: "マハラノビス距離" }, { value: "cos", label: "コサイン距離" }],
      "lp",
      () => draw()
    );
    const svg = E.svg(el, 640, 300, "原点からの距離が 1 になる点の集まり（Lp 距離・マハラノビス距離）と、コサイン距離で使う角度を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const slX = E.sliders(
      el,
      [
        { key: "x1", label: "x_1", min: -2, max: 2, step: 0.1, value: 1.2 },
        { key: "x2", label: "x_2", min: -2, max: 2, step: 0.1, value: 0.8 },
      ],
      () => draw()
    );
    const boxX = el.lastChild;
    const slP = E.sliders(el, [{ key: "p", label: "p", min: 0.5, max: 8, step: 0.1, value: 3 }], () => draw());
    const boxP = el.lastChild;
    const slR = E.sliders(el, [{ key: "rho", label: "\\rho", min: -0.9, max: 0.9, step: 0.05, value: 0.7 }], () => draw());
    const boxR = el.lastChild;
    boxP.style.borderTop = boxR.style.borderTop = "none";
    boxP.style.paddingTop = boxR.style.paddingTop = "0";
    const note = E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" });
    el.append(note);

    function draw() {
      const m = mode.get();
      const { x1, x2 } = slX.get();
      const { p } = slP.get();
      const { rho } = slR.get();
      const f = E.fmt;
      boxP.style.display = m === "lp" ? "" : "none";
      boxR.style.display = m === "maha" ? "" : "none";
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 座標軸と目盛
      [-2, -1, 0, 1, 2].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: sy(-2.3), x2: sx(v), y2: sy(2.3), class: "ew-grid" }));
        svg.append(E.s("line", { x1: sx(-2.3), y1: sy(v), x2: sx(2.3), y2: sy(v), class: "ew-grid" }));
      });
      svg.append(E.s("line", { x1: sx(-2.3), y1: sy(0), x2: sx(2.3), y2: sy(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: sx(0), y1: sy(-2.3), x2: sx(0), y2: sy(2.3), class: "ew-edge" }));
      svg.append(E.s("text", { x: sx(2.3) + 8, y: sy(0), class: "ew-s ew-start" }, "x₁"));
      svg.append(E.s("text", { x: sx(0) + 8, y: sy(2.3) - 4, class: "ew-s ew-start" }, "x₂"));

      const curve = (g, k, style) => {
        const pts = [];
        for (let i = 0; i <= 200; i++) {
          const [a, b] = g((i / 200) * 2 * Math.PI);
          pts.push(sx(a).toFixed(1) + "," + sy(b).toFixed(1));
        }
        svg.append(E.s("polyline", { "data-k": k, points: pts.join(" "), class: "ew-edge", style }));
      };
      const ref = "stroke:var(--line-strong);stroke-width:1.2;stroke-dasharray:4 3";
      const cur = "stroke:var(--main);stroke-width:2.2";
      let rowsDef;

      if (m === "lp") {
        curve((t) => ballLp(t, 2), "ref", ref); // 比較用に L2 の円
        curve((t) => ballLp(t, p), "ball", cur);
        svg.append(E.s("circle", { "data-k": "pt", cx: sx(x1), cy: sy(x2), r: 5, class: "ew-dot-f" }));
        rowsDef = [
          { tex: "\\|\\mathbf{x}\\|_p=\\Bigl(\\sum_i|x_i|^p\\Bigr)^{1/p}", num: "\\|\\mathbf{x}\\|_{" + f(p, 1) + "}=" + f(lp(x1, x2, p)), hl: ["ball", "pt"] },
          { tex: "p=1:\\ \\sum_i|x_i|,\\quad p=2:\\ \\sqrt{\\textstyle\\sum_i x_i^2}", num: "L_1=" + f(lp(x1, x2, 1)) + ",\\ L_2=" + f(lp(x1, x2, 2)), hl: ["ref", "pt"] },
          { tex: "p\\to\\infty:\\ \\max_i|x_i|", num: "L_\\infty=" + f(Math.max(Math.abs(x1), Math.abs(x2))), hl: ["pt"] },
        ];
        note.textContent = "紺の線＝選んだ p で距離が 1 になる点。灰の破線＝ユークリッド距離（p=2）の円。p<1 では凹み、三角不等式を満たさず距離にならない。";
      } else if (m === "maha") {
        curve((t) => ballLp(t, 2), "ref", ref);
        curve((t) => ballMaha(t, rho), "ball", cur);
        svg.append(E.s("circle", { "data-k": "pt", cx: sx(x1), cy: sy(x2), r: 5, class: "ew-dot-f" }));
        const dm = maha(x1, x2, rho), de = lp(x1, x2, 2);
        rowsDef = [
          { tex: "d_M=\\sqrt{(\\mathbf{x}-\\boldsymbol{\\mu})^\\top\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})}", num: "d_M=" + f(dm), hl: ["ball", "pt"] },
          { tex: "\\boldsymbol{\\Sigma}=\\begin{pmatrix}1&\\rho\\\\\\rho&1\\end{pmatrix},\\ \\boldsymbol{\\mu}=\\mathbf{0}", num: "\\rho=" + f(rho), hl: ["ball"] },
          { tex: "\\boldsymbol{\\Sigma}=\\mathbf{I}\\ \\Rightarrow\\ d_M=\\|\\mathbf{x}-\\boldsymbol{\\mu}\\|_2", num: "d_2=" + f(de), hl: ["ref", "pt"] },
        ];
        note.textContent = "紺の楕円＝マハラノビス距離が 1 の点。データが相関する方向（ρ>0 なら右上がり）には遠くても距離が小さく、逆方向は小さい移動でも距離が大きい。";
      } else {
        const [rx, ry] = R_VEC;
        const arrow = (a, b, k, style) =>
          svg.append(E.s("line", { "data-k": k, x1: sx(0), y1: sy(0), x2: sx(a), y2: sy(b), class: "ew-edge", style, "marker-end": svg.arrow }));
        arrow(rx, ry, "ref", "stroke:var(--muted);stroke-width:2");
        arrow(x1, x2, "pt", "stroke:var(--main);stroke-width:2");
        svg.append(E.s("text", { x: sx(rx) + 8, y: sy(ry) + 14, class: "ew-s ew-start" }, "基準ベクトル"));
        // 2つのベクトルのあいだの角を円弧で示す
        const a0 = Math.atan2(ry, rx), a1 = Math.atan2(x2, x1);
        let da = a1 - a0;
        while (da > Math.PI) da -= 2 * Math.PI;
        while (da < -Math.PI) da += 2 * Math.PI;
        if (Math.hypot(x1, x2) > 0) {
          const pts = [];
          for (let i = 0; i <= 30; i++) {
            const a = a0 + (da * i) / 30;
            pts.push(sx(0.55 * Math.cos(a)).toFixed(1) + "," + sy(0.55 * Math.sin(a)).toFixed(1));
          }
          svg.append(E.s("polyline", { "data-k": "ang", points: pts.join(" "), class: "ew-edge", style: "stroke:var(--main);stroke-width:2" }));
        }
        const cd = cosDist(x1, x2, rx, ry);
        rowsDef = [
          { tex: "\\cos\\theta=\\dfrac{\\mathbf{x}\\cdot\\mathbf{r}}{\\|\\mathbf{x}\\|\\,\\|\\mathbf{r}\\|}", num: "\\cos\\theta=" + f(1 - cd), hl: ["ang"] },
          { tex: "d_{\\cos}=1-\\cos\\theta", num: "d_{\\cos}=" + f(cd), hl: ["ang", "pt"] },
          { tex: "\\|\\mathbf{x}-\\mathbf{r}\\|_2\\ （比較）", num: "d_2=" + f(Math.hypot(x1 - rx, x2 - ry)), hl: ["pt", "ref"] },
        ];
        note.textContent = "x を同じ向きのまま伸び縮みさせても、コサイン距離は変わらない（ユークリッド距離は変わる）。x が原点だと向きがなく、定義できない。";
      }
      E.formulaRows(rows, svg, rowsDef);
    }

    draw();
  });
})();
