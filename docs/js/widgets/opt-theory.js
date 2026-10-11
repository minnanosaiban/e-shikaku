// 最適化：勾配降下法とニュートン法の比較（opt-gd-newton）、KKT 条件と相補スラック性（opt-kkt）
(function () {
  const E = window.ENote;

  // ---------- 1変数の2次関数 f(w)=a w²/2 で、勾配降下法とニュートン法を比べる ----------
  E.register("opt-gd-newton", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(
      top,
      [
        { value: "gd", label: "勾配降下法" },
        { value: "newton", label: "ニュートン法" },
      ],
      "gd",
      () => draw()
    );
    const svg = E.svg(el, 640, 280, "2次関数の曲線の上に、更新を繰り返した点を順に示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "a", label: "a", min: 0.2, max: 2, step: 0.05, value: 1 },
        { key: "eta", label: "\\eta", min: 0.05, max: 2.5, step: 0.05, value: 0.3 },
        { key: "w0", label: "w_0", min: -2.5, max: 2.5, step: 0.1, value: 2 },
      ],
      () => draw()
    );

    const X0 = 70, X1 = 590, Y0 = 20, Y1 = 225, XR = 2.5, YR = 6.5, K = 8;
    const sx = (w) => X0 + ((w + XR) / (2 * XR)) * (X1 - X0);
    const sy = (y) => Y1 - (y / YR) * (Y1 - Y0);

    function draw() {
      const { a, eta, w0 } = sl.get();
      const mode = toggle.get();
      const f = E.fmt;
      const fn = (w) => 0.5 * a * w * w;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      [-2, -1, 0, 1, 2].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-edge" }));
      const pts = [];
      for (let i = 0; i <= 100; i++) {
        const w = -XR + (i * 2 * XR) / 100;
        const y = fn(w);
        if (y <= YR) pts.push(sx(w).toFixed(1) + "," + sy(y).toFixed(1));
      }
      svg.append(E.s("polyline", { points: pts.join(" "), class: "ew-curve-f" }));

      // 更新列 w_k
      const ws = [w0];
      for (let k = 0; k < K; k++) {
        const w = ws[k];
        ws.push(mode === "gd" ? w - eta * a * w : w - (a * w) / a);
      }
      let out = 0;
      for (let k = 0; k < K; k++) {
        const p = ws[k], q = ws[k + 1];
        const inP = Math.abs(p) <= XR && fn(p) <= YR;
        const inQ = Math.abs(q) <= XR && fn(q) <= YR;
        if (inP && inQ) {
          svg.append(E.s("line", { "data-k": "path", x1: sx(p), y1: sy(fn(p)), x2: sx(q), y2: sy(fn(q)), class: "ew-edge", "stroke-dasharray": "4 3" }));
        }
      }
      ws.slice(0, K + 1).forEach((w, k) => {
        if (Math.abs(w) <= XR && fn(w) <= YR) {
          svg.append(E.s("circle", { "data-k": "path", cx: sx(w), cy: sy(fn(w)), r: k === 0 ? 5.5 : 4, class: k === 0 ? "ew-dot-d" : "ew-dot-f" }));
          if (k === 0) svg.append(E.s("text", { x: sx(w) + (w >= 0 ? 10 : -10), y: sy(fn(w)) - 12, class: "ew-s " + (w >= 0 ? "ew-start" : "ew-end") }, "w₀"));
        } else out++;
      });
      if (out > 0) svg.append(E.s("text", { x: 320, y: 262, class: "ew-s ew-c" }, "範囲外に出た点が " + out + " 個（発散している）"));
      else svg.append(E.s("text", { x: 320, y: 262, class: "ew-s ew-c ew-muted" }, "紺の大きい点が出発点、黒い点は更新後（k=1〜" + K + "）"));

      const rate = 1 - eta * a;
      const list = [0, 1, 2, 3, K].map((k) => "w_{" + k + "}=" + f(ws[k])).join(",\\ ");
      const r = [{ tex: "f(w)=\\tfrac12 a w^2,\\quad f'(w)=aw,\\quad f''(w)=a", num: "f'(w_0)=" + f(a * w0), hl: [] }];
      if (mode === "gd") {
        r.push({ tex: "w_{k+1}=w_k-\\eta f'(w_k)=(1-\\eta a)\\,w_k", num: "1-\\eta a=" + f(rate), hl: ["path"] });
        r.push({ tex: "|1-\\eta a|<1\\iff 0<\\eta<2/a", num: "2/a=" + f(2 / a) + (Math.abs(rate) < 1 ? "\\ \\text{（収束）}" : "\\ \\text{（発散）}"), hl: [] });
      } else {
        r.push({ tex: "w_{k+1}=w_k-\\dfrac{f'(w_k)}{f''(w_k)}=w_k-\\dfrac{a w_k}{a}", num: "w_1=" + f(ws[1]), hl: ["path"] });
      }
      r.push({ tex: "\\text{更新列}", num: list, hl: ["path"] });
      E.formulaRows(rows, svg, r);
    }
    draw();
  });

  // ---------- 制約 x1 + x2 ≤ c つきの最小化：min ½‖x−x0‖² ----------
  E.register("opt-kkt", (el) => {
    const svg = E.svg(el, 640, 330, "目的関数の等高線と制約領域、最適解の位置を示す平面図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "c", label: "c", min: 0, max: 5, step: 0.1, value: 2 },
        { key: "u1", label: "x_{0,1}", min: 0, max: 3.5, step: 0.1, value: 2 },
        { key: "u2", label: "x_{0,2}", min: 0, max: 3.5, step: 0.1, value: 2 },
      ],
      () => draw()
    );

    const X0 = 60, X1 = 320, Y0 = 20, Y1 = 280, R0 = -0.5, R1 = 4.5;
    const sx = (v) => X0 + ((v - R0) / (R1 - R0)) * (X1 - X0);
    const sy = (v) => Y1 - ((v - R0) / (R1 - R0)) * (Y1 - Y0);
    const clip = (v) => Math.min(R1, Math.max(R0, v));

    function draw() {
      const { c, u1, u2 } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 制約領域 x1+x2 ≤ c を多角形で塗る（描画範囲内で切り取る）
      const poly = [];
      const corners = [[R0, R0], [R1, R0], [R1, R1], [R0, R1]];
      const inside = (p) => p[0] + p[1] <= c + 1e-9;
      for (let i = 0; i < 4; i++) {
        const A = corners[i], B = corners[(i + 1) % 4];
        if (inside(A)) poly.push(A);
        if (inside(A) !== inside(B)) {
          const t = (c - A[0] - A[1]) / (B[0] + B[1] - A[0] - A[1]);
          poly.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]);
        }
      }
      if (poly.length >= 3) {
        svg.append(E.s("polygon", { points: poly.map((p) => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" "), class: "ew-node-loss", stroke: "none" }));
      }
      [0, 1, 2, 3, 4].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
      });
      // 制約の境界線 x1 + x2 = c
      const ca = clip(c - R1), cb = clip(c - R0);
      svg.append(E.s("line", { "data-k": "g", x1: sx(c - cb), y1: sy(cb), x2: sx(c - ca), y2: sy(ca), class: "ew-edge", "stroke-width": 2 }));

      // 解
      const s = u1 + u2;
      const active = s > c;
      const lam = active ? (s - c) / 2 : 0;
      const x1 = u1 - lam, x2 = u2 - lam;
      const dist = Math.hypot(x1 - u1, x2 - u2);
      // 等高線（最適解を通る円）。円の外側に出る部分は clipPath ではなく、点数を絞って描く
      const rr = dist > 0.05 ? dist : 0;
      if (rr > 0) {
        const cp = [];
        for (let i = 0; i <= 120; i++) {
          const th = (i / 120) * 2 * Math.PI;
          const px = u1 + rr * Math.cos(th), py = u2 + rr * Math.sin(th);
          if (px >= R0 && px <= R1 && py >= R0 && py <= R1) cp.push(sx(px).toFixed(1) + "," + sy(py).toFixed(1));
          else if (cp.length) { svg.append(E.s("polyline", { "data-k": "f", points: cp.join(" "), class: "ew-curve-f" })); cp.length = 0; }
        }
        if (cp.length > 1) svg.append(E.s("polyline", { "data-k": "f", points: cp.join(" "), class: "ew-curve-f" }));
      }
      // 無制約の最小点 x0
      svg.append(E.s("circle", { "data-k": "f", cx: sx(u1), cy: sy(u2), r: 4.5, class: "ew-dot-f" }));
      svg.append(E.s("text", { x: sx(u1) + 9, y: sy(u2) - 11, class: "ew-s ew-start" }, "x₀"));
      // 最適解
      svg.append(E.s("circle", { "data-k": "opt", cx: sx(x1), cy: sy(x2), r: 6, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: sx(x1) - 10, y: sy(x2) + 14, class: "ew-s ew-end" }, "x*"));
      // 凡例
      svg.append(E.s("text", { x: 360, y: 40, class: "ew-s ew-start" }, "灰の面：g(x)≤0（許される側）"));
      svg.append(E.s("text", { x: 360, y: 62, class: "ew-s ew-start" }, "円：f の等高線"));
      svg.append(E.s("text", { x: 360, y: 84, class: "ew-s ew-start" }, "太線：g(x)=0"));
      svg.append(E.s("text", { x: 190, y: 312, class: "ew-s ew-c ew-muted" }, "横軸 x₁、縦軸 x₂"));

      const g = x1 + x2 - c;
      E.formulaRows(rows, svg, [
        { tex: "f(\\mathbf{x})=\\tfrac12\\|\\mathbf{x}-\\mathbf{x}_0\\|^2,\\quad g(\\mathbf{x})=x_1+x_2-c\\le 0", num: "\\mathbf{x}_0\\text{ は" + (active ? "領域の外" : "領域の中") + "}", hl: ["f", "g"] },
        { tex: "\\nabla f+\\lambda\\nabla g=\\mathbf{x}-\\mathbf{x}_0+\\lambda\\mathbf{1}=\\mathbf{0}", num: "\\mathbf{x}^*=(" + f(x1) + ",\\ " + f(x2) + ")", hl: ["opt"] },
        { tex: "g(\\mathbf{x}^*)\\le 0", num: "g=" + f(g), hl: ["g", "opt"] },
        { tex: "\\lambda\\ge 0", num: "\\lambda=" + f(lam), hl: ["opt"] },
        { tex: "\\lambda\\,g(\\mathbf{x}^*)=0", num: f(lam) + "\\times" + f(g) + "=" + f(lam * g) + (active ? "\\ \\text{（制約が効いている：}g=0\\text{）}" : "\\ \\text{（制約は効かない：}\\lambda=0\\text{）}"), hl: ["g", "opt"] },
      ]);
    }
    draw();
  });
})();
