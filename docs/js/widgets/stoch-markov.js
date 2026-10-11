// 2状態のマルコフ連鎖：分布 p_t が定常分布 p* に近づく様子（収束の速さは第2固有値 1−α−β で決まる）
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 遷移行列 P=[[1-α, α],[β, 1-β]] のもとで、時刻 t に状態 A にいる確率（行列のべき乗を使わない閉じた形）
  E.math.markov2 = function (alpha, beta, p0, t) {
    const s = alpha + beta;
    const pi = beta / s;
    return pi + (p0 - pi) * Math.pow(1 - s, t);
  };

  const T_MAX = 20;
  const PX0 = 404, PX1 = 584, PY0 = 56, PY1 = 236; // グラフの枠
  const sx = (t) => PX0 + (t / T_MAX) * (PX1 - PX0);
  const sy = (p) => PY1 - p * (PY1 - PY0);

  E.register("stoch-markov", (el) => {
    const svg = E.svg(el, 640, 290, "2つの状態 A と B を行き来するマルコフ連鎖の状態遷移図と、状態 A にいる確率が時間とともに定常分布へ近づくグラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "alpha", label: "\\alpha", min: 0.02, max: 1, step: 0.01, value: 0.3 },
        { key: "beta", label: "\\beta", min: 0.02, max: 1, step: 0.01, value: 0.1 },
        { key: "p0", label: "p_0", min: 0, max: 1, step: 0.05, value: 0.9 },
        { key: "t", label: "t", min: 0, max: T_MAX, step: 1, value: 3, digits: 0 },
      ],
      () => draw()
    );

    const NAVY = "stroke:var(--main)";
    function draw() {
      const { alpha, beta, p0, t } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // --- 状態遷移図 ---
      const AX = 100, BX = 250, NY = 140, R = 28;
      const arrow = (d, key, label, lx, ly) => {
        svg.append(E.s("path", { d, class: "ew-edge", "data-k": key, "marker-end": svg.arrow, style: "stroke-width:1.4" }));
        svg.append(E.s("text", { x: lx[0], y: lx[1], class: "ew-s " + (lx[2] || "ew-c") }, label));
      };
      arrow("M75.7,126 C30,100 30,180 75.7,154", "sa", "1−α", [34, 140, "ew-end"]);
      arrow("M274.3,126 C320,100 320,180 274.3,154", "sb", "1−β", [330, 140, "ew-start"]);
      arrow("M119.8,120.2 Q175,80 230.2,120.2", "ab", "α", [175, 86]);
      arrow("M230.2,159.8 Q175,200 119.8,159.8", "ba", "β", [175, 194]);
      [[AX, "A"], [BX, "B"]].forEach(([x, name]) => {
        svg.append(E.s("circle", { cx: x, cy: NY, r: R, class: "ew-node" }));
        svg.append(E.s("text", { x, y: NY, class: "ew-t ew-c" }, name));
      });
      const pt = E.math.markov2(alpha, beta, p0, t);
      svg.append(E.s("text", { x: AX, y: 236, class: "ew-t ew-c", style: "fill:var(--main)", "data-k": "cur" }, f(pt, 3)));
      svg.append(E.s("text", { x: BX, y: 236, class: "ew-t ew-c", style: "fill:var(--main)", "data-k": "cur" }, f(1 - pt, 3)));
      svg.append(E.s("text", { x: 175, y: 262, class: "ew-s ew-c" }, "t = " + t + " のときに A、B にいる確率"));

      // --- グラフ：P(X_t = A) ---
      svg.append(E.s("text", { x: PX0, y: 18, class: "ew-s ew-start" }, "状態 A にいる確率 P(X_t = A)"));
      svg.append(E.s("line", { x1: PX0, y1: 38, x2: PX0 + 30, y2: 38, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: PX0 + 38, y: 38, class: "ew-s ew-start" }, "定常分布 p*"));
      [0, 0.5, 1].forEach((v) => {
        svg.append(E.s("line", { x1: PX0, y1: sy(v), x2: PX1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: PX0 - 8, y: sy(v), class: "ew-s ew-end ew-muted" }, String(v)));
      });
      [0, 5, 10, 15, 20].forEach((k) => {
        svg.append(E.s("text", { x: sx(k), y: PY1 + 16, class: "ew-s ew-c ew-muted" }, String(k)));
      });
      const pstar = beta / (alpha + beta);
      svg.append(E.s("line", { x1: PX0, y1: sy(pstar), x2: PX1, y2: sy(pstar), class: "ew-curve-d", "data-k": "pi" }));
      svg.append(E.s("text", { x: PX1 + 8, y: sy(pstar), class: "ew-s ew-start" }, f(pstar, 2)));
      const pts = [];
      for (let k = 0; k <= T_MAX; k++) pts.push(sx(k).toFixed(1) + "," + sy(E.math.markov2(alpha, beta, p0, k)).toFixed(1));
      svg.append(E.s("polyline", { points: pts.join(" "), class: "ew-curve-f", "data-k": "traj" }));
      for (let k = 0; k <= T_MAX; k++) {
        svg.append(E.s("circle", { cx: sx(k), cy: sy(E.math.markov2(alpha, beta, p0, k)), r: 2.4, class: "ew-dot-f", "data-k": "traj" }));
      }
      svg.append(E.s("circle", { cx: sx(t), cy: sy(pt), r: 5.5, class: "ew-dot-d", "data-k": "cur" }));

      const lam2 = 1 - alpha - beta;
      E.formulaRows(rows, svg, [
        { tex: "P=\\begin{pmatrix}1-\\alpha&\\alpha\\\\\\beta&1-\\beta\\end{pmatrix}", num: "=\\begin{pmatrix}" + f(1 - alpha, 2) + "&" + f(alpha, 2) + "\\\\" + f(beta, 2) + "&" + f(1 - beta, 2) + "\\end{pmatrix}", hl: ["sa", "sb", "ab", "ba"] },
        { tex: "\\mathbf{p}_t=\\mathbf{p}_0P^{t}", num: "\\mathbf{p}_{" + t + "}=(" + f(pt, 3) + ",\\ " + f(1 - pt, 3) + ")", hl: ["cur", "traj"] },
        { tex: "\\mathbf{p}^{*}=\\mathbf{p}^{*}P\\ \\Rightarrow\\ \\mathbf{p}^{*}=\\Bigl(\\dfrac{\\beta}{\\alpha+\\beta},\\dfrac{\\alpha}{\\alpha+\\beta}\\Bigr)", num: "=(" + f(pstar, 3) + ",\\ " + f(1 - pstar, 3) + ")", hl: ["pi"] },
        { tex: "\\lambda_2(P)=1-\\alpha-\\beta", num: "=" + f(lam2, 2) + (Math.abs(lam2) > 0.999 ? "\\ (|\\lambda_2|=1:\\text{収束しない})" : "\\ (|\\lambda_2|<1:\\text{収束する})"), hl: ["traj"] },
      ]);
    }
    draw();
  });
})();
