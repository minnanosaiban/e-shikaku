// L1 / L2 正則化：データ損失の等高線とペナルティの等高線が接する点が解。L1 では解が軸の上に乗りやすい
(function () {
  const E = window.ENote;

  const H = [3, 1]; // データ損失 ½ Σ h_i (w_i - w*_i)² の曲率

  // 閉じた式の解（H が対角なので、成分ごとに独立に解ける）
  function solve(kind, lam, a, b) {
    return [a, b].map((ws, i) => {
      if (kind === "l2") return (H[i] / (H[i] + lam)) * ws;
      const m = Math.abs(ws) - lam / H[i];
      return m > 0 ? Math.sign(ws) * m : 0;
    });
  }
  E._regSolve = solve; // 検算用

  // 描画範囲：w1, w2 ∈ [-3, 3]（正方形）
  const X0 = 185, X1 = 455, Y0 = 20, Y1 = 290;
  const sx = (v) => X0 + ((v + 3) / 6) * (X1 - X0);
  const sy = (v) => Y1 - ((v + 3) / 6) * (Y1 - Y0);
  const U = (X1 - X0) / 6; // 1 単位あたりの画素数

  E.register("reg-l1l2", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const toggle = E.toggle(top, [{ value: "l2", label: "L2（weight decay）" }, { value: "l1", label: "L1" }], "l2", () => draw());
    const svg = E.svg(el, 640, 330, "重み空間で、データ損失の等高線とペナルティの等高線が接する点を示す図");
    const clipId = "regclip-" + Math.random().toString(36).slice(2, 8);
    const cp = E.s("clipPath", { id: clipId });
    cp.append(E.s("rect", { x: X0, y: Y0, width: X1 - X0, height: Y1 - Y0 }));
    svg.firstChild.nextSibling.append(cp);
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "lam", label: "\\lambda", min: 0, max: 3, step: 0.05, value: 1 },
        { key: "a", label: "w_1^*", min: -2.8, max: 2.8, step: 0.1, value: 1.8 },
        { key: "b", label: "w_2^*", min: -2.8, max: 2.8, step: 0.1, value: 0.8 },
      ],
      () => draw()
    );

    function draw() {
      const kind = toggle.get();
      const { lam, a, b } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      const clip = `url(#${clipId})`;

      // 枠・軸
      svg.append(E.s("rect", { x: X0, y: Y0, width: X1 - X0, height: Y1 - Y0, fill: "none", class: "ew-edge" }));
      svg.append(E.s("line", { x1: X0, y1: sy(0), x2: X1, y2: sy(0), class: "ew-grid" }));
      svg.append(E.s("line", { x1: sx(0), y1: Y0, x2: sx(0), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: X1 + 8, y: sy(0) - 10, class: "ew-s ew-start ew-muted" }, "w₁"));
      svg.append(E.s("text", { x: sx(0) + 8, y: Y0 - 8, class: "ew-s ew-start ew-muted" }, "w₂"));

      const w = solve(kind, lam, a, b);
      const cSol = 0.5 * (H[0] * (w[0] - a) ** 2 + H[1] * (w[1] - b) ** 2);
      const ellipse = (c, cls, extra) => {
        if (c < 1e-9) return null;
        const e = E.s("ellipse", Object.assign({
          cx: sx(a), cy: sy(b), rx: U * Math.sqrt((2 * c) / H[0]), ry: U * Math.sqrt((2 * c) / H[1]),
          fill: "none", class: cls, "clip-path": clip,
        }, extra));
        svg.append(e);
        return e;
      };

      // データ損失の等高線（固定の数本＋解を通る 1 本）
      [0.3, 1, 2.2, 4].forEach((c) => ellipse(c, "ew-grid"));
      ellipse(cSol, "ew-edge");

      // λ を 0 から 3 まで動かしたときの解の軌跡
      const trace = [];
      for (let i = 0; i <= 60; i++) {
        const p = solve(kind, i * 0.05, a, b);
        trace.push(sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1));
      }
      svg.append(E.s("polyline", { points: trace.join(" "), fill: "none", class: "ew-edge", "stroke-dasharray": "2 3" }));

      // ペナルティの等高線（解を通るもの）
      const navy = "stroke:var(--main);stroke-width:1.8";
      if (kind === "l2") {
        const r = Math.hypot(w[0], w[1]);
        if (r > 1e-9) svg.append(E.s("circle", { "data-k": "pen", cx: sx(0), cy: sy(0), r: U * r, fill: "none", style: navy, "clip-path": clip }));
      } else {
        const s = Math.abs(w[0]) + Math.abs(w[1]);
        if (s > 1e-9) {
          const d = `M${sx(s)} ${sy(0)} L${sx(0)} ${sy(s)} L${sx(-s)} ${sy(0)} L${sx(0)} ${sy(-s)} Z`;
          svg.append(E.s("path", { "data-k": "pen", d, fill: "none", style: navy, "clip-path": clip }));
        }
      }

      // 点
      svg.append(E.s("circle", { cx: sx(a), cy: sy(b), r: 4, class: "ew-node" }));
      svg.append(E.s("circle", { "data-k": "sol", cx: sx(w[0]), cy: sy(w[1]), r: 5, class: "ew-dot-d" }));

      // 凡例（右の余白）
      const LXp = X1 + 40;
      svg.append(E.s("circle", { cx: LXp, cy: 60, r: 4, class: "ew-node" }));
      svg.append(E.s("text", { x: LXp + 12, y: 60, class: "ew-s ew-start" }, "w*（損失の最小）"));
      svg.append(E.s("circle", { cx: LXp, cy: 84, r: 5, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: LXp + 12, y: 84, class: "ew-s ew-start" }, "解"));
      svg.append(E.s("line", { x1: LXp - 8, y1: 108, x2: LXp + 8, y2: 108, style: navy }));
      svg.append(E.s("text", { x: LXp + 12, y: 108, class: "ew-s ew-start" }, "ペナルティ"));
      svg.append(E.s("line", { x1: LXp - 8, y1: 132, x2: LXp + 8, y2: 132, class: "ew-edge", "stroke-dasharray": "2 3" }));
      svg.append(E.s("text", { x: LXp + 12, y: 132, class: "ew-s ew-start" }, "λ を動かした軌跡"));

      const head = kind === "l2"
        ? { tex: "L_{\\mathrm{total}}=L+\\dfrac{\\lambda}{2}\\|\\mathbf{w}\\|_2^2", sol: (k) => "w_" + k + "=\\dfrac{h_" + k + "}{h_" + k + "+\\lambda}\\,w_" + k + "^*" }
        : { tex: "L_{\\mathrm{total}}=L+\\lambda\\|\\mathbf{w}\\|_1", sol: (k) => "w_" + k + "=\\mathrm{sign}(w_" + k + "^*)\\max\\bigl(0,\\,\\lvert w_" + k + "^*\\rvert-\\lambda/h_" + k + "\\bigr)" };
      const num = (i) => {
        const ws = i === 0 ? a : b, h = H[i];
        return kind === "l2"
          ? "w_" + (i + 1) + "=\\dfrac{" + h + "}{" + h + "+" + f(lam) + "}\\times" + f(ws, 1) + "=" + f(w[i])
          : "w_" + (i + 1) + "=" + (ws < 0 ? "-" : "") + "\\max\\bigl(0,\\," + f(Math.abs(ws), 1) + "-" + f(lam / h) + "\\bigr)=" + f(w[i]);
      };
      E.formulaRows(rows, svg, [
        { tex: head.tex, num: "h_1=3,\\ h_2=1", hl: ["pen"] },
        { tex: head.sol(1), num: num(0), hl: ["sol"] },
        { tex: head.sol(2), num: num(1), hl: ["sol"] },
      ]);
    }

    draw();
  });
})();
