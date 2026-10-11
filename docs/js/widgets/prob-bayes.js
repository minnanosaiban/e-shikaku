// ベイズの定理：1000 人の検査を点で数える（自然頻度）。陽性のうち本当に罹患している割合が事後確率
(function () {
  const E = window.ENote;
  E.math = E.math || {};

  // 事前確率 pr、感度 se=P(B|A)、偽陽性率 fp=P(B|¬A) から、事後確率などを求める
  E.math.bayes = function (pr, se, fp) {
    const pB = se * pr + fp * (1 - pr);
    const post = pB > 0 ? (se * pr) / pB : NaN;
    const pnB = 1 - pB;
    const postNeg = pnB > 0 ? ((1 - se) * pr) / pnB : NaN;
    return { pB, post, pnB, postNeg };
  };

  const COLS = 50, ROWS = 20, TOTAL = COLS * ROWS;
  const DX = 12, DY = 11, OX = 26, OY = 18;

  E.register("prob-bayes", (el) => {
    const svg = E.svg(el, 640, 292, "1000 人に検査をしたときの内訳を点で示す図。陽性の点のうち、本当に罹患している点の割合が事後確率");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "pr", label: "P(A)", min: 0.001, max: 0.5, step: 0.001, value: 0.05, digits: 3 },
        { key: "se", label: "P(B\\mid A)", min: 0.5, max: 0.999, step: 0.001, value: 0.9, digits: 3 },
        { key: "fp", label: "P(B\\mid \\bar A)", min: 0.001, max: 0.5, step: 0.001, value: 0.1, digits: 3 },
      ],
      () => draw()
    );

    function draw() {
      const { pr, se, fp } = sl.get();
      const f = E.fmt;
      const r = E.math.bayes(pr, se, fp);
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 1000 人の内訳（点の数）。整数に丸めるので、式の値とはわずかにずれる
      const nA = Math.round(TOTAL * pr);
      const tp = Math.round(nA * se), fn = nA - tp;
      const fpn = Math.round((TOTAL - nA) * fp), tn = TOTAL - nA - fpn;
      const order = [["tp", tp], ["fp", fpn], ["fn", fn], ["tn", tn]];
      let idx = 0;
      order.forEach(([k, cnt]) => {
        for (let c = 0; c < cnt; c++, idx++) {
          const cx = OX + (idx % COLS) * DX, cy = OY + Math.floor(idx / COLS) * DY;
          const a = { cx, cy, "data-k": k };
          if (k === "tp") Object.assign(a, { r: 4.2, class: "ew-dot-d" });
          else if (k === "fp") Object.assign(a, { r: 4.2, style: "fill:var(--muted);stroke:none" });
          else if (k === "fn") Object.assign(a, { r: 3.6, style: "fill:none;stroke:var(--main);stroke-width:1.4" });
          else Object.assign(a, { r: 2.6, style: "fill:var(--line-strong);stroke:none" });
          svg.append(E.s("circle", a));
        }
      });

      // 集計の一文
      const pos = tp + fpn;
      svg.append(E.s("text", { x: 16, y: 252, class: "ew-t ew-start" },
        "陽性 " + pos + " 人のうち、罹患している人は " + tp + " 人（" + (pos > 0 ? f((100 * tp) / pos, 1) : "—") + " %）"));
      // 凡例
      [["tp", 16, "罹患・陽性"], ["fn", 170, "罹患・陰性"], ["fp", 330, "健康・陽性"], ["tn", 484, "健康・陰性"]].forEach(([k, x, lab]) => {
        const a = { cx: x + 6, cy: 276, "data-k": k };
        if (k === "tp") Object.assign(a, { r: 4.2, class: "ew-dot-d" });
        else if (k === "fp") Object.assign(a, { r: 4.2, style: "fill:var(--muted);stroke:none" });
        else if (k === "fn") Object.assign(a, { r: 3.6, style: "fill:none;stroke:var(--main);stroke-width:1.4" });
        else Object.assign(a, { r: 2.6, style: "fill:var(--line-strong);stroke:none" });
        svg.append(E.s("circle", a));
        svg.append(E.s("text", { x: x + 18, y: 276, class: "ew-s ew-start" }, lab));
      });

      E.formulaRows(rows, svg, [
        { tex: "P(B)=P(B\\mid A)P(A)+P(B\\mid \\bar A)P(\\bar A)", num: "=" + f(se, 3) + "\\cdot" + f(pr, 3) + "+" + f(fp, 3) + "\\cdot" + f(1 - pr, 3) + "=" + f(r.pB, 4), hl: ["tp", "fp"] },
        { tex: "P(A\\mid B)=\\dfrac{P(B\\mid A)P(A)}{P(B)}", num: "=\\dfrac{" + f(se * pr, 4) + "}{" + f(r.pB, 4) + "}=" + f(r.post, 4), hl: ["tp", "fp"] },
        { tex: "P(A\\mid \\bar B)=\\dfrac{P(\\bar B\\mid A)P(A)}{P(\\bar B)}", num: "=\\dfrac{" + f((1 - se) * pr, 4) + "}{" + f(r.pnB, 4) + "}=" + f(r.postNeg, 4), hl: ["fn", "tn"] },
      ]);
    }
    draw();
  });
})();
