// 畳み込みの出力サイズ・窓の動き・パラメータ数
(function () {
  const E = window.ENote;
  const CELL = 22;
  const comma = (v) => E.fmt(v, 0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

  E.register("cnn-conv", (el) => {
    const svg = E.svg(el, 640, 330, "入力にパディングを足し、カーネルの窓をずらして出力の1マスを作る様子を示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(
      el,
      [
        { key: "H", label: "H", min: 3, max: 9, step: 1, value: 7, digits: 0 },
        { key: "F", label: "F", min: 1, max: 5, step: 1, value: 3, digits: 0 },
        { key: "P", label: "P", min: 0, max: 2, step: 1, value: 1, digits: 0 },
        { key: "S", label: "S", min: 1, max: 3, step: 1, value: 1, digits: 0 },
        { key: "k", label: "出力の位置", min: 0, max: 80, step: 1, value: 10, digits: 0 },
        { key: "ci", label: "C_{in}", min: 1, max: 64, step: 1, value: 3, digits: 0 },
        { key: "co", label: "C_{out}", min: 1, max: 64, step: 1, value: 16, digits: 0 },
      ],
      () => draw()
    );

    function draw() {
      const { H, F, P, S, k, ci, co } = sl.get();
      const n = H + 2 * P;
      const span = n - F;
      const o = span < 0 ? 0 : Math.floor(span / S) + 1;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      const ix = 16, iy = 30, ox = 346;
      svg.append(E.s("text", { x: ix, y: 14, class: "ew-s ew-start" }, "入力（外側の灰色はパディング）"));
      svg.append(E.s("text", { x: ox, y: 14, class: "ew-s ew-start" }, o > 0 ? "出力 " + o + "×" + o : "出力なし（窓が入らない）"));

      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const pad = r < P || r >= P + H || c < P || c >= P + H;
          svg.append(E.s("rect", {
            "data-k": "in", x: ix + c * CELL, y: iy + r * CELL, width: CELL, height: CELL,
            class: pad ? "ew-node-loss" : "ew-node",
          }));
        }
      }
      for (let r = 0; r < o; r++) {
        for (let c = 0; c < o; c++) {
          svg.append(E.s("rect", {
            "data-k": "out", x: ox + c * CELL, y: iy + r * CELL, width: CELL, height: CELL, class: "ew-node",
          }));
        }
      }
      if (o > 0) {
        // 選んだ出力の1マスと、それを作る入力の窓
        const idx = Math.min(k, o * o - 1);
        const rr = Math.floor(idx / o), cc = idx % o;
        svg.append(E.s("rect", {
          x: ox + cc * CELL, y: iy + rr * CELL, width: CELL, height: CELL, class: "ew-bar-0 ew-hl",
        }));
        svg.append(E.s("rect", {
          x: ix + cc * S * CELL, y: iy + rr * S * CELL, width: F * CELL, height: F * CELL,
          fill: "none", class: "ew-hl",
        }));
      }

      const params = F * F * ci * co + co;
      const mac = F * F * ci * co * o * o;
      const fc = ci * H * H * co * o * o;
      const r = [
        {
          tex: "O=\\left\\lfloor\\frac{H+2P-F}{S}\\right\\rfloor+1",
          num: "=\\left\\lfloor\\frac{" + H + "+2\\cdot" + P + "-" + F + "}{" + S + "}\\right\\rfloor+1=" + o,
          hl: ["out"],
        },
        {
          tex: "F^2C_{in}C_{out}+C_{out}",
          num: "=" + F + "^2\\cdot" + ci + "\\cdot" + co + "+" + co + "=" + comma(params),
          hl: [],
        },
        {
          tex: "F^2C_{in}C_{out}O^2",
          num: "=" + comma(mac) + "\\ \\text{回の積和}",
          hl: ["out"],
        },
        {
          tex: "\\text{全結合なら}\\ (C_{in}H^2)(C_{out}O^2)",
          num: "=" + comma(fc),
          hl: ["in"],
        },
      ];
      E.formulaRows(rows, svg, r);
    }

    draw();
  });
})();
