// ロジスティック回帰（1 次元）：σ(wx+b)・オッズ・ロジット・交差エントロピーと勾配
(function () {
  const E = window.ENote;

  // (x, t)：t=0 と t=1 が中央付近で重なるデータ（完全には分離できない）
  const DATA = [[-3.5, 0], [-2.8, 0], [-2.2, 0], [-1.5, 0], [-1.0, 0], [-0.2, 0], [0.8, 0],
    [-0.6, 1], [0.3, 1], [1.0, 1], [1.6, 1], [2.2, 1], [2.9, 1], [3.6, 1]];

  const sig = (z) => 1 / (1 + Math.exp(-z));
  // 負の対数尤度 L と勾配（Σ は全サンプル）
  function stats(w, b) {
    let L = 0, gw = 0, gb = 0;
    DATA.forEach(([x, t]) => {
      const z = w * x + b, y = sig(z);
      // log(1+e^z) の安定な計算：L_n = softplus(z) - t z
      L += Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - t * z;
      gw += (y - t) * x;
      gb += y - t;
    });
    return { L, gw, gb };
  }
  // ニュートン法（IRLS）で最尤解を求める
  function newton(w = 0, b = 0) {
    for (let i = 0; i < 100; i++) {
      let hww = 0, hwb = 0, hbb = 0;
      DATA.forEach(([x]) => {
        const y = sig(w * x + b), r = y * (1 - y);
        hww += r * x * x; hwb += r * x; hbb += r;
      });
      const g = stats(w, b), det = hww * hbb - hwb * hwb;
      w -= (hbb * g.gw - hwb * g.gb) / det;
      b -= (-hwb * g.gw + hww * g.gb) / det;
    }
    return { w, b };
  }

  if (typeof module !== "undefined") module.exports = { DATA, stats, newton, sig };
  if (!E) return;

  const X0 = 50, X1 = 610, Y0 = 20, Y1 = 220;
  const sx = (x) => X0 + ((x + 4.5) / 9) * (X1 - X0);
  const sy = (p) => Y1 - p * (Y1 - Y0);

  E.register("logistic-fit", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const svg = E.svg(el, 640, 252, "1次元の2クラスのデータとロジスティック曲線。重みとバイアスで曲線が動く図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "w", label: "w", min: -3, max: 3, step: 0.05, value: 0.5 },
        { key: "b", label: "b", min: -3, max: 3, step: 0.05, value: -1 },
        { key: "x", label: "x", min: -4, max: 4, step: 0.1, value: 1 },
      ],
      () => draw()
    );
    const inputs = [...el.querySelectorAll(".ew-sliders input")];
    top.append(E.h("button", {
      type: "button", class: "ew-btn",
      onclick: () => {
        const s = newton();
        [s.w, s.b].forEach((v, i) => {
          inputs[i].value = v;
          inputs[i].dispatchEvent(new Event("input"));
        });
      },
    }, "最尤解にする"));
    el.append(E.h("div", { style: "font-size:.68rem;color:var(--muted);margin-top:.3rem" },
      "●＝t=1、○＝t=0 のデータ。縦の点線が x の位置、紺の曲線が y=σ(wx+b)。"));

    // 目盛
    [-4, -2, 0, 2, 4].forEach((v) => {
      svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
      svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
    });
    [0, 0.5, 1].forEach((v) => {
      svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
      svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
    });
    DATA.forEach(([x, t]) => {
      if (t === 1) svg.append(E.s("circle", { "data-k": "pts", cx: sx(x), cy: sy(1), r: 5, class: "ew-dot-d" }));
      else svg.append(E.s("circle", { "data-k": "pts", cx: sx(x), cy: sy(0), r: 5, class: "ew-node", style: "stroke:var(--muted);stroke-width:1.6" }));
    });
    const dyn = E.s("g", {});
    svg.append(dyn);

    function draw() {
      const { w, b, x } = sl.get();
      const f = E.fmt;
      dyn.innerHTML = "";
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const v = -4.5 + (i / 200) * 9;
        pts.push(sx(v).toFixed(1) + "," + sy(sig(w * v + b)).toFixed(1));
      }
      dyn.append(E.s("polyline", { "data-k": "curve", points: pts.join(" "), class: "ew-edge", style: "stroke:var(--main);stroke-width:2.2" }));
      // 判別境界 wx+b=0（p=0.5）
      if (Math.abs(w) > 1e-6) {
        const xb = -b / w;
        if (xb > -4.5 && xb < 4.5) dyn.append(E.s("line", { "data-k": "bd", x1: sx(xb), y1: Y0, x2: sx(xb), y2: Y1, class: "ew-edge", style: "stroke:var(--main);stroke-width:1;stroke-dasharray:2 3" }));
      }
      const z = w * x + b, p = sig(z);
      dyn.append(E.s("line", { "data-k": "q", x1: sx(x), y1: Y0, x2: sx(x), y2: Y1, class: "ew-edge", style: "stroke:var(--muted);stroke-dasharray:3 3" }));
      dyn.append(E.s("circle", { "data-k": "q", cx: sx(x), cy: sy(p), r: 5, class: "ew-dot-f" }));

      const st = stats(w, b);
      E.formulaRows(rows, svg, [
        { tex: "z=wx+b\\ （\\text{ロジット}=\\log\\tfrac{y}{1-y}）", num: "z=" + f(z), hl: ["q"] },
        { tex: "y=\\sigma(z)=\\dfrac{1}{1+e^{-z}}", num: "y=" + f(p), hl: ["q", "curve"] },
        { tex: "\\text{オッズ}\\ \\dfrac{y}{1-y}=e^{z}", num: "e^{" + f(z) + "}=" + E.fmtSci(Math.exp(z)), hl: ["q"] },
        { tex: "L=-\\sum_n\\bigl[t_n\\log y_n+(1-t_n)\\log(1-y_n)\\bigr]", num: "L=" + f(st.L), hl: ["pts", "curve"] },
        { tex: "\\dfrac{\\partial L}{\\partial w}=\\sum_n(y_n-t_n)x_n,\\ \\ \\dfrac{\\partial L}{\\partial b}=\\sum_n(y_n-t_n)", num: "(" + f(st.gw) + ",\\ " + f(st.gb) + ")", hl: ["pts", "curve"] },
        { tex: "\\text{判別境界}\\ \\ wx+b=0\\ \\Leftrightarrow\\ y=0.5", num: Math.abs(w) > 1e-6 ? "x=" + f(-b / w) : "\\text{（w=0 では境界なし）}", hl: ["bd"] },
      ]);
    }

    draw();
  });
})();
