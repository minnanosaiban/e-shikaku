// 早期終了：検証損失が patience 回続けて改善しなければ止め、最良のエポックのモデルを返す
(function () {
  const E = window.ENote;

  const N = 100; // エポック数
  const train = (e) => 0.1 + 1.1 * Math.exp(-e / 16);
  // 検証損失：いったん下がり、過学習で上がる。小さな揺れ（ノイズ）が重なる
  const val = (e, a) => 0.45 + 0.9 * Math.exp(-e / 10) + 0.004 * e + a * Math.sin(0.9 * e);

  function run(patience, a) {
    let best = Infinity, bestE = 0, cnt = 0, stopE = N;
    for (let e = 0; e <= N; e++) {
      const v = val(e, a);
      if (v < best) { best = v; bestE = e; cnt = 0; } else cnt++;
      if (cnt >= patience) { stopE = e; break; }
    }
    return { best, bestE, stopE };
  }
  E._earlyRun = run; // 検算用

  const X0 = 70, X1 = 590, Y0 = 20, Y1 = 220, YMAX = 1.4;
  const sx = (e) => X0 + (e / N) * (X1 - X0);
  const sy = (v) => Y1 - (Math.min(v, YMAX) / YMAX) * (Y1 - Y0);

  E.register("reg-early", (el) => {
    const svg = E.svg(el, 640, 290, "訓練損失と検証損失の曲線と、早期終了で止めるエポックを示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "p", label: "\\text{patience}", min: 1, max: 20, step: 1, value: 5, digits: 0 },
        { key: "a", label: "\\text{ノイズ}", min: 0, max: 0.05, step: 0.005, value: 0.02, digits: 3 },
      ],
      () => draw()
    );

    function draw() {
      const { p, a } = sl.get();
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();
      const r = run(p, a);

      // 目盛
      [0, 20, 40, 60, 80, 100].forEach((v) => {
        svg.append(E.s("line", { x1: sx(v), y1: Y0, x2: sx(v), y2: Y1, class: "ew-grid" }));
        svg.append(E.s("text", { x: sx(v), y: Y1 + 16, class: "ew-s ew-c ew-muted" }, String(v)));
      });
      [0, 0.5, 1].forEach((v) => {
        svg.append(E.s("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: "ew-grid" }));
        svg.append(E.s("text", { x: X0 - 10, y: sy(v), class: "ew-s ew-end ew-muted" }, f(v, 1)));
      });
      svg.append(E.s("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "ew-edge" }));

      // 曲線
      const pt = (g) => {
        const out = [];
        for (let e = 0; e <= N; e++) out.push(sx(e).toFixed(1) + "," + sy(g(e)).toFixed(1));
        return out.join(" ");
      };
      svg.append(E.s("polyline", { "data-k": "tr", points: pt(train), class: "ew-curve-f" }));
      svg.append(E.s("polyline", { "data-k": "va", points: pt((e) => val(e, a)), class: "ew-curve-d" }));

      // 最良のエポックと停止するエポック（最良のラベルは線の左、停止のラベルは線の右）
      const mark = (e, key, text, y, cls, style, onLeft) => {
        const x = sx(e);
        const at = { "data-k": key, x1: x, y1: Y0, x2: x, y2: Y1, class: cls };
        if (style) at.style = style;
        svg.append(E.s("line", at));
        const left = onLeft ? x > X0 + 70 : x > X1 - 70;
        svg.append(E.s("text", { x: left ? x - 6 : x + 6, y, class: "ew-s " + (left ? "ew-end" : "ew-start") }, text));
      };
      mark(r.bestE, "best", "最良 " + r.bestE, Y0 + 12, "ew-edge", "stroke:var(--main);stroke-width:1.6", true);
      mark(r.stopE, "stop", "停止 " + r.stopE, Y0 + 32, "ew-edge", "stroke-dasharray:4 3");

      // 凡例（下）
      const ly = Y1 + 44;
      svg.append(E.s("line", { x1: X0, y1: ly, x2: X0 + 30, y2: ly, class: "ew-curve-f" }));
      svg.append(E.s("text", { x: X0 + 38, y: ly, class: "ew-s ew-start" }, "訓練損失"));
      svg.append(E.s("line", { x1: X0 + 140, y1: ly, x2: X0 + 170, y2: ly, class: "ew-curve-d" }));
      svg.append(E.s("text", { x: X0 + 178, y: ly, class: "ew-s ew-start" }, "検証損失"));
      svg.append(E.s("text", { x: X1, y: ly, class: "ew-s ew-end" }, "横軸：エポック"));

      E.formulaRows(rows, svg, [
        {
          tex: "e^{*}=\\arg\\min_e L_{\\mathrm{val}}(e)",
          num: "e^{*}=" + r.bestE + ",\\ L_{\\mathrm{val}}=" + f(r.best),
          hl: ["best", "va"],
        },
        {
          tex: "e_{\\mathrm{stop}}\\ \\bigl(\\text{改善なしが patience 回続く}\\bigr)",
          num: "e_{\\mathrm{stop}}=" + r.stopE + (r.stopE === N ? "\\ \\text{（最後まで止まらず）}" : ""),
          hl: ["stop"],
        },
        {
          tex: "\\text{返すモデル}=e^{*}\\text{ の重み}",
          num: "e_{\\mathrm{stop}}-e^{*}=" + (r.stopE - r.bestE) + "\\ \\text{エポックぶん戻る}",
          hl: ["best"],
        },
      ]);
    }

    draw();
  });
})();
