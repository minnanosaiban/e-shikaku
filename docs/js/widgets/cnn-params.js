// 特別な畳み込みのパラメータ数の比較（バイアスは除く）
(function () {
  const E = window.ENote;
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const comma = (v) => E.fmt(v, 0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  const X0 = 150, W = 290;

  E.register("cnn-params", (el) => {
    const svg = E.svg(el, 640, 205, "標準・グループ化・分離可能などの畳み込みのパラメータ数を、標準を1として比べる棒グラフ");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "K", label: "K", min: 1, max: 7, step: 1, value: 3, digits: 0 },
        { key: "ci", label: "C_{in}", min: 1, max: 256, step: 1, value: 64, digits: 0 },
        { key: "co", label: "C_{out}", min: 1, max: 256, step: 1, value: 64, digits: 0 },
        { key: "G", label: "G", min: 1, max: 32, step: 1, value: 4, digits: 0 },
        { key: "r", label: "r", min: 1, max: 8, step: 1, value: 2, digits: 0 },
      ],
      () => draw()
    );

    const NAMES = ["標準", "グループ化", "DW＋PW", "PW のみ", "DW のみ"];
    const bars = NAMES.map((name, i) => {
      const y = 12 + i * 34;
      svg.append(E.s("text", { x: X0 - 10, y: y + 12, class: "ew-t ew-end" }, name));
      const rect = E.s("rect", { "data-k": "b" + i, x: X0, y, width: 0, height: 24, rx: 3, class: "ew-bar-0" });
      const label = E.s("text", { x: X0, y: y + 12, class: "ew-s ew-start" });
      svg.append(rect, label);
      return { rect, label };
    });
    svg.append(E.s("text", { x: 320, y: 190, class: "ew-s ew-c ew-muted" }, "棒の長さは、標準の畳み込みを 1 としたときの比"));

    function draw() {
      const { K, ci, co, G, r } = sl.get();
      // G は C_in と C_out の公約数でなければならないので、G 以下で最大の公約数に直す
      const g0 = gcd(ci, co);
      let g = 1;
      for (let d = 1; d <= Math.min(G, g0); d++) if (g0 % d === 0) g = d;

      const std = K * K * ci * co;
      const vals = [std, std / g, K * K * ci + ci * co, ci * co, K * K * ci];
      vals.forEach((v, i) => {
        const w = Math.max(W * (v / std), 1);
        bars[i].rect.setAttribute("width", w);
        bars[i].label.setAttribute("x", X0 + w + 8);
        bars[i].label.textContent = comma(v) + "（" + E.fmt((100 * v) / std, 1) + "%）";
      });

      const feff = r * (K - 1) + 1;
      const f = E.fmt;
      E.formulaRows(rows, svg, [
        { tex: "K^2C_{in}C_{out}", num: "=" + comma(std), hl: ["b0"] },
        {
          tex: "\\dfrac{K^2C_{in}C_{out}}{G}",
          num: "G=" + g + (g !== G ? "\\ (\\text{公約数に調整})" : "") + "\\ \\Rightarrow\\ " + comma(std / g),
          hl: ["b1"],
        },
        {
          tex: "K^2C_{in}+C_{in}C_{out}",
          num: "=" + comma(vals[2]) + "\\ \\ \\left(\\tfrac{1}{C_{out}}+\\tfrac{1}{K^2}=" + f(1 / co + 1 / (K * K), 4) + "\\right)",
          hl: ["b2", "b3", "b4"],
        },
        { tex: "C_{in}C_{out}\\ (1{\\times}1)", num: "=" + comma(vals[3]), hl: ["b3"] },
        { tex: "K^2C_{in}\\ (\\text{DW})", num: "=" + comma(vals[4]), hl: ["b4"] },
        {
          tex: "F_{\\mathrm{eff}}=r(K-1)+1",
          num: "=" + r + "\\cdot" + (K - 1) + "+1=" + feff + "\\ \\ (\\text{パラメータ数は標準と同じ})",
          hl: [],
        },
      ]);
    }

    draw();
  });
})();
