// 固有値分解と特異値分解：単位円が A でどう写るか（2×2）
//   特異値分解 … 円が楕円になる。楕円の軸の長さが特異値
//   固有値分解 … 向きが変わらない方向（固有ベクトル）が、λ 倍になる
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 2×2 行列 A=[[a,b],[c,d]] の特異値分解（A^T A の固有値分解から求める）
  E.math.svd2 = function (a, b, c, d) {
    const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d;
    const half = (p + r) / 2, rad = Math.sqrt(((p - r) / 2) ** 2 + q * q);
    const s1 = Math.sqrt(Math.max(half + rad, 0));
    const s2 = Math.sqrt(Math.max(half - rad, 0));
    const phi = 0.5 * Math.atan2(2 * q, p - r);
    const v1 = [Math.cos(phi), Math.sin(phi)];
    const v2 = [-Math.sin(phi), Math.cos(phi)];
    const A = (v) => [a * v[0] + b * v[1], c * v[0] + d * v[1]];
    const u1 = s1 > 1e-7 ? A(v1).map((t) => t / s1) : [1, 0];
    const u2 = s2 > 1e-7 ? A(v2).map((t) => t / s2) : [-u1[1], u1[0]];
    return { s1, s2, v1, v2, u1, u2 };
  };

  // 2×2 行列の固有値・固有ベクトル（長さ 1 に正規化）
  // type: real（相異なる実数）／scalar（単位行列の定数倍）／defective（対角化できない）／complex（複素数）
  E.math.eig2 = function (a, b, c, d) {
    const tr = a + d, det = a * d - b * c;
    const disc = (tr * tr) / 4 - det;
    const unit = (v) => { const n = Math.hypot(v[0], v[1]); return [v[0] / n, v[1] / n]; };
    const vec = (l) => {
      if (Math.abs(b) > 1e-9) return unit([b, l - a]);
      if (Math.abs(c) > 1e-9) return unit([l - d, c]);
      return Math.abs(l - a) < 1e-9 ? [1, 0] : [0, 1];
    };
    if (disc < -1e-9) return { type: "complex", re: tr / 2, im: Math.sqrt(-disc), tr, det };
    if (disc <= 1e-9) {
      const l = tr / 2;
      if (Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9) return { type: "scalar", l1: l, l2: l, v1: [1, 0], v2: [0, 1], tr, det };
      return { type: "defective", l1: l, l2: l, v1: vec(l), tr, det };
    }
    const sq = Math.sqrt(disc);
    const l1 = tr / 2 + sq, l2 = tr / 2 - sq;
    return { type: "real", l1, l2, v1: vec(l1), v2: vec(l2), tr, det };
  };

  const CX = 320, CY = 176, SC = 40, EXT = 3.2; // 1 目盛 = 40px、範囲 ±3.2（上の 50px は凡例用）
  const px = (x) => CX + SC * x;
  const py = (y) => CY - SC * y;
  const NAVY = "stroke:var(--main)";
  const SUB = "\u2081\u2082";

  E.register("eigen-svd", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      [{ value: "svd", label: "特異値分解" }, { value: "eig", label: "固有値分解" }],
      "svd",
      () => draw()
    );
    const svg = E.svg(el, 640, 350, "2×2 行列で単位円がどう写るかを示す図。特異値分解では楕円の軸、固有値分解では向きの変わらない方向を見る");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "a", label: "a", min: -1.5, max: 1.5, step: 0.1, value: 1.2 },
        { key: "b", label: "b", min: -1.5, max: 1.5, step: 0.1, value: 0.8 },
        { key: "c", label: "c", min: -1.5, max: 1.5, step: 0.1, value: 0.2 },
        { key: "d", label: "d", min: -1.5, max: 1.5, step: 0.1, value: 1.0 },
        { key: "t", label: "\\theta^\\circ", min: 0, max: 360, step: 5, value: 120, digits: 0 },
      ],
      () => draw()
    );

    let labels = [];
    // 原点から (x,y) への矢印と、先端の外側のラベル
    function arrow(x, y, key, style, label, labelStyle) {
      const len = Math.hypot(x, y);
      if (len < 0.05) return;
      svg.append(E.s("line", { x1: px(0), y1: py(0), x2: px(x), y2: py(y), "data-k": key, style, "marker-end": svg.arrow }));
      if (label) {
        // 先端の外側に置く。ほかのラベルと近いときは、さらに外側へずらす
        let off = 17, lx, ly;
        for (let n = 0; n < 5; n++, off += 14) {
          lx = px(x) + (x / len) * off + (y / len) * 12; ly = py(y) - (y / len) * off + (x / len) * 12;
          if (!labels.some(([ax, ay]) => Math.hypot(ax - lx, ay - ly) < 26)) break;
        }
        labels.push([lx, ly]);
        svg.append(E.s("text", { x: lx, y: ly, class: "ew-t ew-c", style: labelStyle || "" }, label));
      }
    }

    function draw() {
      const { a, b, c, d, t } = sl.get();
      const mode = toggle.get();
      const f = E.fmt;
      const th = (t * Math.PI) / 180;
      const A = (v) => [a * v[0] + b * v[1], c * v[0] + d * v[1]];
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      labels = [];

      for (let k = -3; k <= 3; k++) {
        svg.append(E.s("line", { x1: px(k), y1: py(-EXT), x2: px(k), y2: py(EXT), class: "ew-grid" }));
        svg.append(E.s("line", { x1: px(-EXT), y1: py(k), x2: px(EXT), y2: py(k), class: "ew-grid" }));
      }
      svg.append(E.s("line", { x1: px(-EXT), y1: py(0), x2: px(EXT), y2: py(0), class: "ew-edge" }));
      svg.append(E.s("line", { x1: px(0), y1: py(-EXT), x2: px(0), y2: py(EXT), class: "ew-edge" }));

      // 単位円（灰）と、その像（紺）
      const circ = [], ell = [];
      for (let i = 0; i <= 120; i++) {
        const u = (2 * Math.PI * i) / 120;
        const v = [Math.cos(u), Math.sin(u)], w = A(v);
        circ.push(px(v[0]).toFixed(1) + "," + py(v[1]).toFixed(1));
        ell.push(px(w[0]).toFixed(1) + "," + py(w[1]).toFixed(1));
      }
      svg.append(E.s("polyline", { points: circ.join(" "), class: "ew-edge", style: "stroke-width:1.4", "data-k": "circ" }));
      svg.append(E.s("polyline", { points: ell.join(" "), "data-k": "ell", style: NAVY + ";fill:var(--main);fill-opacity:0.1;stroke-width:1.6" }));

      // 円周上の点 x と、その像 Ax
      const pt = [Math.cos(th), Math.sin(th)], ap = A(pt);
      svg.append(E.s("circle", { cx: px(pt[0]), cy: py(pt[1]), r: 4.5, class: "ew-dot-f", "data-k": "pt" }));
      svg.append(E.s("circle", { cx: px(ap[0]), cy: py(ap[1]), r: 4.5, class: "ew-dot-d", "data-k": "ap" }));

      let r;
      if (mode === "svd") {
        const sv = E.math.svd2(a, b, c, d);
        arrow(sv.v1[0], sv.v1[1], "v1", "stroke:var(--muted);stroke-width:2", "v\u2081", "fill:var(--muted)");
        arrow(sv.v2[0], sv.v2[1], "v2", "stroke:var(--muted);stroke-width:2", "v\u2082", "fill:var(--muted)");
        arrow(sv.s1 * sv.u1[0], sv.s1 * sv.u1[1], "u1", NAVY + ";stroke-width:2.2", "\u03c3\u2081u\u2081", "fill:var(--main)");
        arrow(sv.s2 * sv.u2[0], sv.s2 * sv.u2[1], "u2", NAVY + ";stroke-width:2.2", "\u03c3\u2082u\u2082", "fill:var(--main)");
        svg.append(E.s("text", { x: 16, y: 16, class: "ew-s ew-start" }, "灰：単位円と右特異ベクトル v"));
        svg.append(E.s("text", { x: 16, y: 36, class: "ew-s ew-start", style: "fill:var(--main)" }, "紺：像（楕円）と σu"));
        const A1 = A(sv.v1), A2 = A(sv.v2);
        const rk = (sv.s1 > 1e-7 ? 1 : 0) + (sv.s2 > 1e-7 ? 1 : 0);
        r = [
          { tex: "\\sigma_i=\\sqrt{\\lambda_i(A^{\\top}A)}", num: "\\sigma_1=" + f(sv.s1, 3) + ",\\ \\sigma_2=" + f(sv.s2, 3), hl: ["u1", "u2"] },
          { tex: "A\\mathbf{v}_i=\\sigma_i\\mathbf{u}_i", num: "A\\mathbf{v}_1=(" + f(A1[0], 2) + ",\\," + f(A1[1], 2) + "),\\ A\\mathbf{v}_2=(" + f(A2[0], 2) + ",\\," + f(A2[1], 2) + ")", hl: ["v1", "v2", "u1", "u2"] },
          { tex: "|\\det A|=\\sigma_1\\sigma_2", num: "|" + f(a * d - b * c, 3) + "|=" + f(sv.s1 * sv.s2, 3), hl: ["ell"] },
          { tex: "\\operatorname{rank}A=\\#\\{i:\\sigma_i>0\\}", num: "=" + rk, hl: ["ell"] },
        ];
      } else {
        const ev = E.math.eig2(a, b, c, d);
        svg.append(E.s("text", { x: 16, y: 16, class: "ew-s ew-start" }, "灰：単位円と固有ベクトル v（直線は固有ベクトルの方向）"));
        svg.append(E.s("text", { x: 16, y: 36, class: "ew-s ew-start", style: "fill:var(--main)" }, "紺：像（楕円）と λv"));
        const eigLine = (v, key) => svg.append(E.s("line", {
          x1: px(-EXT * v[0]), y1: py(-EXT * v[1]), x2: px(EXT * v[0]), y2: py(EXT * v[1]),
          class: "ew-edge", style: "stroke-dasharray:4 3", "data-k": key,
        }));
        const char = [
          { tex: "\\lambda^2-\\operatorname{tr}(A)\\lambda+\\det A=0", num: "\\lambda^2-" + f(ev.tr, 2) + "\\lambda+" + f(ev.det, 2) + "=0", hl: [] },
        ];
        if (ev.type === "real" || ev.type === "scalar") {
          [[ev.l1, ev.v1, "1"], [ev.l2, ev.v2, "2"]].forEach(([l, v, i]) => {
            eigLine(v, "w" + i);
            arrow(v[0], v[1], "w" + i, "stroke:var(--muted);stroke-width:2", "");
            arrow(l * v[0], l * v[1], "w" + i, NAVY + ";stroke-width:2.2", "\u03bb" + SUB[i - 1] + "v" + SUB[i - 1], "fill:var(--main)");
          });
          const Av1 = A(ev.v1);
          r = char.concat([
            { tex: "\\lambda_1,\\ \\lambda_2", num: "=" + f(ev.l1, 3) + ",\\ " + f(ev.l2, 3) + (ev.type === "scalar" ? "\\ (\\text{どの向きも固有ベクトル})" : ""), hl: ["w1", "w2"] },
            { tex: "\\operatorname{tr}A=\\lambda_1+\\lambda_2,\\ \\det A=\\lambda_1\\lambda_2", num: f(ev.tr, 2) + "=" + f(ev.l1 + ev.l2, 2) + ",\\ " + f(ev.det, 2) + "=" + f(ev.l1 * ev.l2, 2), hl: [] },
            { tex: "A\\mathbf{v}_1=\\lambda_1\\mathbf{v}_1", num: "(" + f(Av1[0], 2) + ",\\," + f(Av1[1], 2) + ")=" + f(ev.l1, 2) + "(" + f(ev.v1[0], 2) + ",\\," + f(ev.v1[1], 2) + ")", hl: ["w1"] },
          ]);
        } else if (ev.type === "defective") {
          eigLine(ev.v1, "w1");
          arrow(ev.v1[0], ev.v1[1], "w1", "stroke:var(--muted);stroke-width:2", "");
          arrow(ev.l1 * ev.v1[0], ev.l1 * ev.v1[1], "w1", NAVY + ";stroke-width:2.2", "\u03bbv", "fill:var(--main)");
          r = char.concat([
            { tex: "\\lambda_1=\\lambda_2", num: "=" + f(ev.l1, 3) + "\\ (\\text{重解})", hl: ["w1"] },
            { tex: "\\text{固有ベクトル}", num: "\\text{1 本だけ → 対角化できない}", hl: ["w1"] },
          ]);
        } else {
          r = char.concat([
            { tex: "\\lambda=\\alpha\\pm i\\beta", num: "=" + f(ev.re, 3) + "\\pm " + f(ev.im, 3) + "i", hl: ["ell"] },
            { tex: "\\text{実の固有ベクトル}", num: "\\text{なし（回転を含むので、向きの変わらない直線がない）}", hl: ["ell"] },
          ]);
        }
      }
      E.formulaRows(rows, svg, r);
    }
    draw();
  });
})();
