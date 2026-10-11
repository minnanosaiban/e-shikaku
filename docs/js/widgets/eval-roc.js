// 2クラス分類：しきい値 t を動かすと、混同行列・各指標・ROC 曲線上の点が連動して動く
// スコアはネガティブ N(0,1)、ポジティブ N(d,1) に従うと仮定する（等分散の2正規モデル）
(function () {
  // ---- 計算 ----
  // 標準正規分布の累積分布関数（erf は Abramowitz-Stegun 7.1.26、誤差 1.5e-7）
  const erf = (v) => {
    const s = Math.sign(v), a = Math.abs(v);
    const t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  };
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const Phi = (v) => clamp01(0.5 * (1 + erf(v / Math.SQRT2)));
  const pdf = (v) => Math.exp(-v * v / 2) / Math.sqrt(2 * Math.PI);

  // しきい値 t 以上をポジティブと予測するときの指標
  function metrics(d, t, pi, n = 1000) {
    const tpr = 1 - Phi(t - d), fpr = 1 - Phi(t);
    const TP = n * pi * tpr, FN = n * pi * (1 - tpr), FP = n * (1 - pi) * fpr, TN = n * (1 - pi) * (1 - fpr);
    return {
      tpr, fpr, TP, FN, FP, TN,
      precision: TP / (TP + FP),
      f1: (2 * TP) / (2 * TP + FP + FN),
      acc: (TP + TN) / n,
      auc: Phi(d / Math.SQRT2),
    };
  }
  if (typeof module !== "undefined") module.exports = { Phi, metrics };
  if (typeof window === "undefined") return;

  // ---- 描画 ----
  const E = window.ENote;

  E.register("eval-roc", (el) => {
    const svg = E.svg(el, 640, 290, "2つのスコア分布としきい値、およびROC曲線上の動作点を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "t", label: "t", min: -2, max: 5, step: 0.1, value: 0.8 },
        { key: "d", label: "d'", min: 0, max: 4, step: 0.1, value: 1.5 },
        { key: "pi", label: "\\pi", min: 0.01, max: 0.9, step: 0.01, value: 0.5 },
      ],
      () => draw()
    );

    // 左：スコア分布（x ∈ [-4, 8]）
    const A0 = 40, A1 = 310, BASE = 235, HT = 160;
    const ax = (x) => A0 + ((x + 4) / 12) * (A1 - A0);
    const ay = (v) => BASE - (v / 0.42) * HT;
    // 右：ROC 曲線
    const R0 = 380, R1 = 600, T0 = 30, T1 = 250;
    const rx = (v) => R0 + v * (R1 - R0), ry = (v) => T1 - v * (T1 - T0);

    function draw() {
      const { t, d, pi } = sl.get();
      const f = E.fmt;
      const m = metrics(d, t, pi);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // スコア分布（事前確率で重みを付けた密度）
      svg.append(E.s("line", { x1: A0, y1: BASE, x2: A1, y2: BASE, class: "ew-edge" }));
      [-4, 0, 4, 8].forEach((v) => svg.append(E.s("text", { x: ax(v), y: BASE + 16, class: "ew-s ew-c" }, String(v))));
      svg.append(E.s("text", { x: (A0 + A1) / 2, y: BASE + 38, class: "ew-s ew-c" }, "スコア"));
      const curve = (g) => {
        const out = [];
        for (let i = 0; i <= 120; i++) { const v = -4 + i * 0.1; out.push(ax(v).toFixed(1) + "," + ay(g(v)).toFixed(1)); }
        return out.join(" ");
      };
      svg.append(E.s("polyline", { "data-k": "neg", points: curve((v) => (1 - pi) * pdf(v)), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "pos", points: curve((v) => pi * pdf(v - d)), class: "ew-curve-d" }));
      svg.append(E.s("line", { "data-k": "t", x1: ax(t), y1: 28, x2: ax(t), y2: BASE, class: "ew-edge", style: "stroke:var(--main);stroke-width:2" }));
      svg.append(E.s("text", { x: ax(t), y: 14, class: "ew-s ew-c" }, "t"));
      svg.append(E.s("line", { x1: A0 + 4, y1: 44, x2: A0 + 28, y2: 44, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: A0 + 34, y: 44, class: "ew-s ew-start" }, "ネガティブ"));
      svg.append(E.s("line", { x1: A0 + 4, y1: 62, x2: A0 + 28, y2: 62, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: A0 + 34, y: 62, class: "ew-s ew-start" }, "ポジティブ"));

      // ROC 曲線
      svg.append(E.s("rect", { x: R0, y: T0, width: R1 - R0, height: T1 - T0, class: "ew-node", style: "fill:none" }));
      svg.append(E.s("line", { x1: rx(0), y1: ry(0), x2: rx(1), y2: ry(1), class: "ew-edge", "stroke-dasharray": "4 3" }));
      const pts = [];
      for (let i = 0; i <= 130; i++) {
        const tt = -4 + i * 0.1;
        pts.push(rx(1 - Phi(tt)).toFixed(1) + "," + ry(1 - Phi(tt - d)).toFixed(1));
      }
      svg.append(E.s("polyline", { "data-k": "roc", points: pts.join(" "), class: "ew-curve-f" }));
      svg.append(E.s("circle", { "data-k": "pt", cx: rx(m.fpr), cy: ry(m.tpr), r: 5.5, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: rx(0.5), y: T1 + 18, class: "ew-s ew-c" }, "FPR（偽陽性率）"));
      svg.append(E.s("text", { x: R0 - 8, y: ry(1), class: "ew-s ew-end" }, "1"));
      svg.append(E.s("text", { x: R0 - 8, y: ry(0), class: "ew-s ew-end" }, "0"));
      svg.append(E.s("text", { x: R0 + 4, y: 16, class: "ew-s ew-start" }, "TPR（再現率）"));
      svg.append(E.s("text", { x: R1 - 8, y: T1 - 14, class: "ew-t ew-end", "data-k": "roc" }, "AUC = " + f(m.auc)));

      E.formulaRows(rows, svg, [
        { tex: "\\mathrm{TPR}=1-\\Phi(t-d'),\\ \\ \\mathrm{FPR}=1-\\Phi(t)", num: "=" + f(m.tpr) + ",\\ " + f(m.fpr), hl: ["t", "pt"] },
        {
          tex: "TP,\\ FP,\\ FN,\\ TN\\ \\ （1000件あたり）",
          num: "=" + [m.TP, m.FP, m.FN, m.TN].map((v) => f(v, 0)).join(",\\ "),
          hl: ["pos", "neg"],
        },
        { tex: "\\text{適合率}=\\dfrac{TP}{TP+FP}", num: "=" + f(m.precision), hl: ["pt"] },
        { tex: "F_1=\\dfrac{2TP}{2TP+FP+FN}", num: "=" + f(m.f1), hl: ["pt"] },
        { tex: "\\text{正解率}=\\dfrac{TP+TN}{TP+TN+FP+FN}", num: "=" + f(m.acc), hl: ["pt"] },
        { tex: "\\mathrm{AUC}=\\Phi\\bigl(d'/\\sqrt{2}\\bigr)", num: "=" + f(m.auc), hl: ["roc"] },
      ]);
    }
    draw();
  });
})();
