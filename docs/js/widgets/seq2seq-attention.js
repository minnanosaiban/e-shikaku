// Attention：クエリ s とエンコーダ状態 h_j のスコア → softmax → 重み付き和（2次元で可視化）
(function () {
  const E = window.ENote;

  // エンコーダの隠れ状態（固定の例。2次元）
  const H = [[1.5, 0.3], [0.2, 1.4], [-1.3, 0.5], [-0.4, -1.3]];
  // 加法アテンションの固定パラメータ（例）：score = v^T tanh(Wh h + Ws s)
  const WH = [[1, -0.5], [0.5, 1]];
  const WS = [[1, 0.5], [-0.5, 1]];
  const V = [1, 1];

  const scoreDot = (h, s) => h[0] * s[0] + h[1] * s[1];
  const scoreAdd = (h, s) => {
    let e = 0;
    for (let i = 0; i < 2; i++) {
      e += V[i] * Math.tanh(WH[i][0] * h[0] + WH[i][1] * h[1] + WS[i][0] * s[0] + WS[i][1] * s[1]);
    }
    return e;
  };
  function softmax(e) {
    const m = Math.max(...e);
    const ex = e.map((v) => Math.exp(v - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / sum);
  }

  const CX = 165, CY = 150, SC = 62; // 平面の原点と 1 あたりの画素数
  const BX = 400, BW = 160;

  E.register("seq2seq-attention", (el) => {
    const bar = E.h("div", { class: "ew-bar" });
    el.append(bar);
    const mode = E.toggle(bar, [{ value: "dot", label: "内積（Luong）" }, { value: "add", label: "加法（Bahdanau）" }], "dot", () => draw());
    const svg = E.svg(el, 640, 300, "クエリとエンコーダ状態の関係から注意の重みを求め、重み付き和でコンテキストベクトルを作る図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "s1", label: "s_1", min: -2, max: 2, step: 0.1, value: 1.2 },
        { key: "s2", label: "s_2", min: -2, max: 2, step: 0.1, value: 0.5 },
      ],
      () => draw()
    );

    const px = (v) => CX + v * SC;
    const py = (v) => CY - v * SC;

    function draw() {
      const f = E.fmt;
      const { s1, s2 } = sl.get();
      const s = [s1, s2];
      const score = mode.get() === "dot" ? scoreDot : scoreAdd;
      const e = H.map((h) => score(h, s));
      const a = softmax(e);
      const c = [0, 1].map((d) => a.reduce((acc, aj, j) => acc + aj * H[j][d], 0));

      while (svg.childNodes.length > 2) svg.lastChild.remove();

      // 平面
      svg.append(E.s("text", { x: 20, y: 14, class: "ew-s ew-start ew-muted" }, "平面：◯ h、矢印 s、● c"));
      svg.append(E.s("line", { x1: px(-2), y1: py(0), x2: px(2), y2: py(0), class: "ew-grid" }));
      svg.append(E.s("line", { x1: px(0), y1: py(-2), x2: px(0), y2: py(2), class: "ew-grid" }));
      // クエリ（原点から）
      if (Math.hypot(s1, s2) > 0.05) {
        svg.append(E.s("line", { x1: px(0), y1: py(0), x2: px(s1), y2: py(s2), class: "ew-curve-d", "marker-end": svg.arrow }));
      }
      svg.append(E.s("text", { x: px(s1) + 8, y: py(s2) - 10, class: "ew-s ew-start ew-back" }, "s"));
      // h_j
      H.forEach((h, j) => {
        svg.append(E.s("circle", { "data-k": "a" + j, cx: px(h[0]), cy: py(h[1]), r: 6 + 10 * a[j], class: "ew-node" }));
        svg.append(E.s("text", { x: px(h[0]) + 12 + 10 * a[j], y: py(h[1]) - 6, class: "ew-t ew-start" }, "h" + (j + 1)));
      });
      // コンテキストベクトル
      svg.append(E.s("circle", { "data-k": "c", cx: px(c[0]), cy: py(c[1]), r: 5, class: "ew-dot-d" }));
      svg.append(E.s("text", { x: px(c[0]) + 9, y: py(c[1]) + 12, class: "ew-s ew-start ew-back" }, "c"));

      // 重みの棒
      svg.append(E.s("text", { x: BX - 56, y: 14, class: "ew-s ew-start ew-muted" }, "注意の重み α"));
      a.forEach((aj, j) => {
        const y = 40 + j * 40;
        svg.append(E.s("text", { x: BX - 12, y: y + 12, class: "ew-t ew-end" }, "h" + (j + 1)));
        svg.append(E.s("rect", { "data-k": "a" + j, x: BX, y, width: Math.max(BW * aj, 1), height: 24, rx: 3, class: "ew-bar-0" }));
        svg.append(E.s("text", { x: BX + Math.max(BW * aj, 1) + 8, y: y + 12, class: "ew-s ew-start" }, f(aj)));
      });
      svg.append(E.s("text", { x: BX - 56, y: 215, class: "ew-s ew-start ew-muted" }, "円の大きさも α に比例"));

      const vec = (v) => "(" + v.map((x) => f(x, 2)).join(",\\ ") + ")";
      E.formulaRows(rows, svg, [
        {
          tex: mode.get() === "dot" ? "e_j=\\mathbf{h}_j^{\\top}\\mathbf{s}" : "e_j=\\mathbf{v}^{\\top}\\tanh(\\mathbf{W}_h\\mathbf{h}_j+\\mathbf{W}_s\\mathbf{s})",
          num: "(e_1,\\dots,e_4)=" + vec(e),
          hl: ["a0", "a1", "a2", "a3"],
        },
        {
          tex: "\\alpha_j=\\dfrac{\\exp(e_j)}{\\sum_{k}\\exp(e_k)}",
          num: "(\\alpha_1,\\dots,\\alpha_4)=" + vec(a) + ",\\ \\textstyle\\sum=" + f(a.reduce((x, y) => x + y, 0), 2),
          hl: ["a0", "a1", "a2", "a3"],
        },
        {
          tex: "\\mathbf{c}=\\sum_{j}\\alpha_j\\mathbf{h}_j",
          num: "=" + vec(c),
          hl: ["c"],
        },
      ]);
    }

    draw();
  });
})();
