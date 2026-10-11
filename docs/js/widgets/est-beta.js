// コインの表の確率 θ の推定：最尤推定・MAP推定・ベイズ推定（Beta 分布が共役事前分布）
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // N 回中 k 回表、事前分布 Beta(α,β) のときの、最尤・MAP・事後平均・事後分布のパラメータ
  E.math.betaBernoulli = function (N, k, al, be) {
    const post = [al + k, be + N - k];
    const dMap = N + al + be - 2;
    return {
      mle: N > 0 ? k / N : NaN,
      map: dMap > 0 ? (k + al - 1) / dMap : NaN, // α+β=2 かつ N=0 のときは山がなく、定まらない
      mean: (k + al) / (N + al + be),
      post,
    };
  };

  const X0 = 40, X1 = 600, Y0 = 62, Y1 = 232;
  const sx = (th) => X0 + th * (X1 - X0);
  const sy = (v) => Y1 - v * (Y1 - Y0);
  const NAVY = "stroke:var(--main)";
  const num = (v) => (Number.isInteger(v) ? String(v) : E.fmt(v, 1));

  // θ^p (1-θ)^q を、最大値が 1 になるようにそろえて 0〜1 を 400 分割で返す
  function shape(p, q) {
    const v = [];
    let m = 0;
    for (let i = 0; i <= 400; i++) {
      const th = i / 400;
      const y = Math.pow(th, p) * Math.pow(1 - th, q);
      v.push(y);
      if (y > m) m = y;
    }
    return v.map((y) => y / m);
  }
  const poly = (v) => v.map((y, i) => sx(i / 400).toFixed(1) + "," + sy(y).toFixed(1)).join(" ");

  E.register("est-beta", (el) => {
    const svg = E.svg(el, 640, 276, "コインの表の確率 θ について、事前分布・尤度・事後分布の形と、最尤推定・MAP推定・事後平均の位置を重ねて示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "N", label: "N", min: 0, max: 50, step: 1, value: 10, digits: 0 },
        { key: "k", label: "k", min: 0, max: 50, step: 1, value: 7, digits: 0 },
        { key: "al", label: "\\alpha", min: 1, max: 10, step: 0.5, value: 3, digits: 1 },
        { key: "be", label: "\\beta", min: 1, max: 10, step: 0.5, value: 3, digits: 1 },
      ],
      () => draw()
    );

    function draw() {
      const q = sl.get();
      const N = q.N, k = Math.min(q.k, N), al = q.al, be = q.be;
      const f = E.fmt;
      const r = E.math.betaBernoulli(N, k, al, be);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
        svg.append(E.s("line", { x1: sx(t), y1: Y0 - 4, x2: sx(t), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(t), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(t)));
      });
      svg.append(E.s("text", { x: X1 + 6, y: Y1 + 16, class: "ew-s ew-start" }, "θ"));
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-edge" }));

      // 曲線（高さは最大値が 1 になるようにそろえてある）
      svg.append(E.s("polyline", { points: poly(shape(al - 1, be - 1)), "data-k": "pri", style: "fill:none;stroke:var(--line-strong);stroke-width:2;stroke-dasharray:5 3" }));
      svg.append(E.s("polyline", { points: poly(shape(k, N - k)), "data-k": "lik", class: "ew-curve-f" }));
      svg.append(E.s("polyline", { points: poly(shape(k + al - 1, N - k + be - 1)), "data-k": "post", style: NAVY + ";fill:none;stroke-width:2.6" }));

      // 推定値の縦線
      if (isFinite(r.mle)) svg.append(E.s("line", { x1: sx(r.mle), y1: Y0 - 4, x2: sx(r.mle), y2: Y1, "data-k": "mle", style: "stroke:var(--muted);stroke-width:1.6" }));
      if (isFinite(r.map)) svg.append(E.s("line", { x1: sx(r.map), y1: Y0 - 4, x2: sx(r.map), y2: Y1, "data-k": "map", class: "ew-curve-d" }));
      svg.append(E.s("line", { x1: sx(r.mean), y1: Y0 - 4, x2: sx(r.mean), y2: Y1, "data-k": "avg", style: NAVY + ";stroke-width:2;stroke-dasharray:1 3" }));

      // 凡例（上段：曲線、下段：縦線）
      const leg = (x, y, line, label) => {
        svg.append(E.s("line", Object.assign({ x1: x, y1: y, x2: x + 26, y2: y }, line)));
        svg.append(E.s("text", { x: x + 34, y, class: "ew-s ew-start" }, label));
      };
      leg(16, 14, { style: "stroke:var(--line-strong);stroke-width:2;stroke-dasharray:5 3" }, "事前分布");
      leg(176, 14, { class: "ew-curve-f" }, "尤度（形だけ）");
      leg(336, 14, { style: NAVY + ";stroke-width:2.6" }, "事後分布");
      leg(16, 36, { style: "stroke:var(--muted);stroke-width:1.6" }, "最尤推定");
      leg(176, 36, { class: "ew-curve-d" }, "MAP推定");
      leg(336, 36, { style: NAVY + ";stroke-width:2;stroke-dasharray:1 3" }, "事後平均");
      svg.append(E.s("text", { x: 624, y: 266, class: "ew-s ew-end" }, "曲線の高さは、最大値を 1 にそろえてある"));

      const dash = "\\text{定まらない}";
      E.formulaRows(rows, svg, [
        { tex: "p(\\mathcal{D}\\mid\\theta)\\propto\\theta^{k}(1-\\theta)^{N-k}", num: "N=" + N + ",\\ k=" + k, hl: ["lik"] },
        { tex: "\\hat\\theta_{\\mathrm{MLE}}=\\dfrac{k}{N}", num: isFinite(r.mle) ? "=" + f(r.mle, 3) : dash, hl: ["mle"] },
        { tex: "p(\\theta)\\propto\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}", num: "\\mathrm{Beta}(" + num(al) + ",\\," + num(be) + ")", hl: ["pri"] },
        { tex: "p(\\theta\\mid\\mathcal{D})=\\mathrm{Beta}(\\alpha+k,\\ \\beta+N-k)", num: "=\\mathrm{Beta}(" + num(r.post[0]) + ",\\," + num(r.post[1]) + ")", hl: ["post"] },
        { tex: "\\hat\\theta_{\\mathrm{MAP}}=\\dfrac{k+\\alpha-1}{N+\\alpha+\\beta-2}", num: isFinite(r.map) ? "=" + f(r.map, 3) : dash, hl: ["map"] },
        { tex: "\\mathbb{E}[\\theta\\mid\\mathcal{D}]=\\dfrac{k+\\alpha}{N+\\alpha+\\beta}", num: "=" + f(r.mean, 3), hl: ["avg"] },
      ]);
    }
    draw();
  });
})();
