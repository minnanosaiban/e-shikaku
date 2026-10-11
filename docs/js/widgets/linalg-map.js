// 行列は平面の写像：単位正方形が A でどう変わるか（行列式＝面積の倍率、ランク＝像の次元）
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 2×2 行列 A=[[a,b],[c,d]] のランク（許容誤差つき）
  E.math.rank2 = function (a, b, c, d) {
    const eps = 1e-9;
    if (Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d)) < eps) return 0;
    return Math.abs(a * d - b * c) < eps ? 1 : 2;
  };

  const CX = 320, CY = 160, SC = 44, EXT = 3.2; // 1 目盛 = 44px、描画範囲は ±3.2
  const px = (x) => CX + SC * x;
  const py = (y) => CY - SC * y;
  const NAVY = "stroke:var(--main)";

  E.register("linalg-map", (el) => {
    const svg = E.svg(el, 640, 320, "2×2 行列で単位正方形がどう写るかを示す図。行列式は面積の倍率、ランクは像の次元を表す");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "a", label: "a", min: -1.5, max: 1.5, step: 0.1, value: 1.2 },
        { key: "b", label: "b", min: -1.5, max: 1.5, step: 0.1, value: 0.5 },
        { key: "c", label: "c", min: -1.5, max: 1.5, step: 0.1, value: 0.3 },
        { key: "d", label: "d", min: -1.5, max: 1.5, step: 0.1, value: 1.0 },
      ],
      () => draw()
    );

    function draw() {
      const { a, b, c, d } = sl.get();
      const f = E.fmt;
      const det = a * d - b * c;
      const r = E.math.rank2(a, b, c, d);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 格子と軸
      for (let k = -3; k <= 3; k++) {
        svg.append(E.s("line", { x1: px(k), y1: py(-EXT), x2: px(k), y2: py(EXT), class: "ew-grid" }));
        svg.append(E.s("line", { x1: px(-EXT), y1: py(k), x2: px(EXT), y2: py(k), class: "ew-grid" }));
      }
      svg.append(E.s("line", { x1: px(-EXT), y1: py(0), x2: px(EXT), y2: py(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: px(0), y1: py(-EXT), x2: px(0), y2: py(EXT), class: "ew-edge" }));

      // 元の単位正方形（グレー）
      svg.append(E.s("polygon", {
        points: [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => px(x) + "," + py(y)).join(" "),
        "data-k": "sq", style: "fill:var(--surface-gray);stroke:var(--muted);stroke-width:1.5",
      }));
      // A で写した像（紺）
      const q = [[0, 0], [a, c], [a + b, c + d], [b, d]];
      svg.append(E.s("polygon", {
        points: q.map(([x, y]) => px(x).toFixed(1) + "," + py(y).toFixed(1)).join(" "),
        "data-k": "img", style: NAVY + ";fill:var(--main);fill-opacity:0.14;stroke-width:1.6",
      }));
      // 列ベクトル Ae1=(a,c), Ae2=(b,d)
      [[a, c, "e1", "Ae₁"], [b, d, "e2", "Ae₂"]].forEach(([x, y, k, lab]) => {
        const len = Math.hypot(x, y);
        if (len < 0.05) return;
        svg.append(E.s("line", {
          x1: px(0), y1: py(0), x2: px(x), y2: py(y), "data-k": k,
          style: NAVY + ";stroke-width:2", "marker-end": svg.arrow,
        }));
        svg.append(E.s("text", { x: px(x) + (x / len) * 16, y: py(y) - (y / len) * 16, class: "ew-t ew-c", style: "fill:var(--main)" }, lab));
      });
      // 凡例
      svg.append(E.s("text", { x: 16, y: 30, class: "ew-s ew-start" }, "灰：単位正方形"));
      svg.append(E.s("text", { x: 16, y: 52, class: "ew-s ew-start", style: "fill:var(--main)" }, "紺：A で写した像"));
      svg.append(E.s("text", { x: 624, y: 30, class: "ew-s ew-end" }, "面積 = |det A| = " + f(Math.abs(det), 2)));
      svg.append(E.s("text", { x: 624, y: 52, class: "ew-s ew-end" }, "像の次元 = rank A = " + r));

      E.formulaRows(rows, svg, [
        { tex: "A\\mathbf{e}_1=\\begin{pmatrix}a\\\\c\\end{pmatrix},\\ A\\mathbf{e}_2=\\begin{pmatrix}b\\\\d\\end{pmatrix}", num: "=\\begin{pmatrix}" + f(a, 1) + "\\\\" + f(c, 1) + "\\end{pmatrix},\\ \\begin{pmatrix}" + f(b, 1) + "\\\\" + f(d, 1) + "\\end{pmatrix}", hl: ["e1", "e2"] },
        { tex: "\\det A=ad-bc", num: "=" + f(a, 1) + "\\cdot" + f(d, 1) + "-" + f(b, 1) + "\\cdot" + f(c, 1) + "=" + f(det, 2), hl: ["sq", "img"] },
        { tex: "\\operatorname{rank}A", num: "=" + r + (r === 2 ? "\\ (\\text{正則})" : r === 1 ? "\\ (\\det A=0:\\text{線に潰れる})" : "\\ (\\text{原点に潰れる})"), hl: ["img"] },
      ]);
    }
    draw();
  });
})();
