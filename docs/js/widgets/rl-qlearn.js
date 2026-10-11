// 強化学習：TD 更新 1 回ぶん（Q 学習と SARSA の違い）
//   Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a*) − Q(s,a) ]
//   Q 学習：a* = argmax_a' Q(s',a')　／　SARSA：a* = 実際に選んだ行動 a'
(function () {
  const E = window.ENote;

  // 1 回の更新（純関数）
  const update = (q, r, alpha, gamma, qNext, aNext, algo) => {
    const mx = Math.max(...qNext);
    const bootstrap = algo === "q" ? mx : qNext[aNext];
    const target = r + gamma * bootstrap;
    const delta = target - q;
    return { max: mx, bootstrap, target, delta, qNew: q + alpha * delta };
  };
  E.rlTdUpdate = update;

  // 横棒グラフの座標（値の範囲 −2〜2）
  const X0 = 200, X1 = 610, V0 = -2, V1 = 2;
  const sx = (v) => X0 + ((Math.min(V1, Math.max(V0, v)) - V0) / (V1 - V0)) * (X1 - X0);

  E.register("rl-qlearn", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const algo = E.toggle(
      top,
      [{ value: "q", label: "Q学習（オフ方策）" }, { value: "sarsa", label: "SARSA（オン方策）" }],
      "q",
      () => draw()
    );
    const act = E.toggle(
      top,
      [{ value: 0, label: "次に選ぶ a'=a₁" }, { value: 1, label: "a₂" }, { value: 2, label: "a₃" }],
      2,
      () => draw()
    );
    const actEl = top.lastElementChild; // 次の行動の選択（SARSA のときだけ表示する）
    const svg = E.svg(el, 640, 290, "状態 s の行動価値 Q(s,a) が、次の状態 s' の行動価値を使った TD 目標に向けて更新される様子を示す棒グラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "q", label: "Q(s,a)", min: -1, max: 1, step: 0.05, value: 0.2, digits: 2 },
        { key: "r", label: "r", min: -1, max: 1, step: 0.05, value: 0.5, digits: 2 },
        { key: "alpha", label: "\\alpha", min: 0, max: 1, step: 0.05, value: 0.5, digits: 2 },
        { key: "gamma", label: "\\gamma", min: 0, max: 0.99, step: 0.01, value: 0.9, digits: 2 },
        { key: "n1", label: "Q(s',a_1)", min: -1, max: 1, step: 0.05, value: 0.1, digits: 2 },
        { key: "n2", label: "Q(s',a_2)", min: -1, max: 1, step: 0.05, value: 0.8, digits: 2 },
        { key: "n3", label: "Q(s',a_3)", min: -1, max: 1, step: 0.05, value: 0.3, digits: 2 },
      ],
      () => draw()
    );

    function bar(k, y, v, cls, label, navy) {
      svg.append(E.s("text", { x: X0 - 12, y: y + 11, class: "ew-t ew-end" }, label));
      const x0 = sx(0), x1 = sx(v);
      svg.append(E.s("rect", { "data-k": k, x: Math.min(x0, x1), y, width: Math.max(Math.abs(x1 - x0), 1), height: 22, rx: 3, class: cls, ...(navy ? { style: "fill:var(--main)" } : {}) }));
    }

    function draw() {
      const { q, r, alpha, gamma, n1, n2, n3 } = sl.get();
      const f = E.fmt;
      const qn = [n1, n2, n3];
      const a = act.get();
      const isQ = algo.get() === "q";
      const u = update(q, r, alpha, gamma, qn, a, isQ ? "q" : "sarsa");
      const star = isQ ? qn.indexOf(u.max) : a; // 目標に使う行動（最大が複数なら先頭）
      actEl.style.display = isQ ? "none" : "";

      while (svg.childNodes.length > 2) svg.lastChild.remove();
      [-2, -1, 0, 1, 2].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: 14, x2: sx(v), y2: 246, class: v === 0 ? "ew-edge" : "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: 266, class: "ew-s ew-c ew-muted" }, String(v)));
      });

      // s の更新前後
      bar("old", 20, q, "ew-bar-0", "更新前 Q(s,a)");
      bar("tgt", 52, u.target, "ew-bar-1", "目標 r+γQ(s',a*)");
      bar("new", 84, u.qNew, "ew-bar-2", "更新後 Q(s,a)", true);
      // s' の 3 つの行動
      svg.append(E.s("line", { x1: 20, y1: 122, x2: X1, y2: 122, class: "ew-grid" }));
      qn.forEach((v, i) => {
        bar("n" + i, 136 + i * 34, v, "ew-bar-" + (i % 3), "Q(s',a" + "₁₂₃"[i] + ")" + (i === star ? " ◀" : ""));
      });

      const bootTex = isQ ? "\\max_{a'}Q(s',a')" : "Q(s',a_" + (a + 1) + ")";
      const bootNum = isQ ? "\\max(" + qn.map((v) => f(v, 2)).join(",") + ")" : "Q(s',a_" + (a + 1) + ")";
      E.formulaRows(rows, svg, [
        { tex: "\\text{目標}\\ y=r+\\gamma\\," + bootTex, num: "=" + f(r, 2) + "+" + f(gamma, 2) + "\\times" + f(u.bootstrap, 2) + "=" + f(u.target), hl: ["tgt", "n" + star] },
        { tex: "\\text{TD 誤差}\\ \\delta=y-Q(s,a)", num: "=" + f(u.target) + "-" + f(q, 2) + "=" + f(u.delta), hl: ["old", "tgt"] },
        { tex: "Q(s,a)\\leftarrow Q(s,a)+\\alpha\\,\\delta", num: "=" + f(q, 2) + "+" + f(alpha, 2) + "\\times" + f(u.delta) + "=" + f(u.qNew), hl: ["old", "new"] },
      ]);
    }

    draw();
  });
})();
