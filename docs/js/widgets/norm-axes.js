// 正規化の軸：どのセルが同じ平均・分散を共有するかを、テンソル (N, C, H×W) の上で見る
(function () {
  const E = window.ENote;
  const NN = 4, CC = 6, HW = 16; // サンプル数 N、チャネル数 C、各セルの空間 H×W = 4×4
  const X0 = 112, Y0 = 50, CW = 70, CH = 36;

  const METHODS = [
    { value: "bn", label: "BN" },
    { value: "ln", label: "LN" },
    { value: "in", label: "IN" },
    { value: "gn2", label: "GN（G=2）" },
    { value: "gn3", label: "GN（G=3）" },
  ];

  E.register("norm-axes", (el) => {
    const mode = E.toggle(el, METHODS, "bn", () => draw());
    const svg = E.svg(el, 640, 250, "テンソルの N 方向と C 方向のセルのうち、同じ平均と分散を共有するセルを示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "n", label: "n", min: 1, max: NN, step: 1, value: 2, digits: 0 },
        { key: "c", label: "c", min: 1, max: CC, step: 1, value: 3, digits: 0 },
      ],
      () => draw()
    );

    function draw() {
      const m = mode.get();
      const n0 = sl.get().n - 1, c0 = sl.get().c - 1;
      const G = m === "gn2" ? 2 : m === "gn3" ? 3 : 1;
      const per = CC / G; // 1 グループあたりのチャネル数
      const same = (n, c) => {
        if (m === "bn") return c === c0;
        if (m === "ln") return n === n0;
        if (m === "in") return n === n0 && c === c0;
        return n === n0 && Math.floor(c / per) === Math.floor(c0 / per);
      };
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      svg.append(E.s("text", { x: X0 + (CC * CW) / 2, y: 20, class: "ew-s ew-c ew-muted" }, "チャネル C（各セル＝ H×W = 4×4 の空間 16 要素）"));
      for (let c = 0; c < CC; c++) {
        svg.append(E.s("text", { x: X0 + c * CW + (CW - 4) / 2, y: 42, class: "ew-s ew-c ew-muted" }, "c=" + (c + 1)));
      }
      let cnt = 0;
      for (let n = 0; n < NN; n++) {
        svg.append(E.s("text", { x: X0 - 12, y: Y0 + n * CH + 22, class: "ew-s ew-end ew-muted" }, "サンプル" + (n + 1)));
        for (let c = 0; c < CC; c++) {
          const x = X0 + c * CW, y = Y0 + n * CH;
          const on = same(n, c);
          const isSel = n === n0 && c === c0;
          if (on) cnt++;
          const attrs = { x, y, width: CW - 4, height: CH - 4, rx: 3, class: "ew-node" };
          if (on) attrs["data-k"] = "grp";
          if (isSel) attrs["data-k"] = "sel";
          svg.append(E.s("rect", attrs));
          if (on) svg.append(E.s("rect", { x, y, width: CW - 4, height: CH - 4, rx: 3, class: "ew-back", "fill-opacity": 0.4 }));
          if (isSel) svg.append(E.s("circle", { cx: x + (CW - 4) / 2, cy: y + (CH - 4) / 2, r: 5, class: "ew-back" }));
        }
      }
      svg.append(E.s("text", { x: 320, y: 220, class: "ew-s ew-c ew-muted" }, "紺のセルが、● のセルと同じ平均・分散で正規化される範囲"));

      const size = cnt * HW; // 1 組の統計量を作る要素数 |S|
      const sets = m === "bn" ? CC : m === "ln" ? NN : m === "in" ? NN * CC : NN * G;
      const range = { bn: "(N,H,W)", ln: "(C,H,W)", in: "(H,W)", gn2: "(C/G,H,W)", gn3: "(C/G,H,W)" }[m];
      const cell = m === "bn" ? NN : m === "ln" ? CC : m === "in" ? 1 : per;
      E.formulaRows(rows, svg, [
        {
          tex: "\\mu=\\dfrac{1}{|S|}\\sum_{i\\in S}x_i,\\quad \\sigma^2=\\dfrac{1}{|S|}\\sum_{i\\in S}(x_i-\\mu)^2",
          num: "S:\\ " + range + ",\\quad |S|=" + cell + "\\times " + HW + "=" + size,
          hl: ["grp", "sel"],
        },
        {
          tex: "\\hat{x}_i=\\dfrac{x_i-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}",
          num: "\\text{統計量の組の数}=" + sets,
          hl: ["sel"],
        },
      ]);
    }
    draw();
  });
})();
