// Scaled Dot-Product Attention：√d_k での割り算とマスクの効果を、重みの棒で見る
(function () {
  const E = window.ENote;
  const NT = 4, MAXD = 64;

  // 乱数は固定（開くたびに同じ図になる）
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(r) {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const r = rng(7);
  const mat = () => Array.from({ length: NT }, () => Array.from({ length: MAXD }, () => normal(r)));
  const Q = mat(), K = mat();

  E.register("transformer-attn", (el) => {
    const scale = E.toggle(el, [{ value: 1, label: "√d_k で割る" }, { value: 0, label: "割らない" }], 1, () => draw());
    const mask = E.toggle(el, [{ value: 0, label: "マスクなし" }, { value: 1, label: "マスクあり（因果）" }], 0, () => draw());
    const svg = E.svg(el, 640, 232, "4つのクエリと4つのキーの間の注意の重みを棒で示す図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sl = E.sliders(el, [{ key: "dk", label: "d_k", min: 1, max: MAXD, step: 1, value: 16, digits: 0 }], () => draw());

    const X0 = 86, CW = 134, BW = 70, Y0 = 44, RH = 36;
    function draw() {
      const { dk } = sl.get();
      const sc = scale.get() ? Math.sqrt(dk) : 1;
      const mk = mask.get() === 1;
      const f = E.fmt;
      while (svg.childNodes.length > 2) svg.lastChild.remove();

      for (let j = 0; j < NT; j++) {
        svg.append(E.s("text", { x: X0 + j * CW, y: 26, class: "ew-t ew-start" }, "k" + "₁₂₃₄"[j]));
      }
      const S = [], A = [];
      for (let i = 0; i < NT; i++) {
        S.push([]);
        for (let j = 0; j < NT; j++) {
          let d = 0;
          for (let t = 0; t < dk; t++) d += Q[i][t] * K[j][t];
          S[i].push(d / sc);
        }
        const vis = S[i].map((v, j) => (mk && j > i ? -Infinity : v));
        const m = Math.max(...vis);
        const ex = vis.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
        const sum = ex.reduce((a, b) => a + b, 0);
        A.push(ex.map((v) => v / sum));
      }
      for (let i = 0; i < NT; i++) {
        const y = Y0 + i * RH;
        svg.append(E.s("text", { x: X0 - 14, y: y + 12, class: "ew-t ew-end" }, "q" + "₁₂₃₄"[i]));
        for (let j = 0; j < NT; j++) {
          const x = X0 + j * CW;
          if (mk && j > i) {
            svg.append(E.s("text", { x, y: y + 12, class: "ew-s ew-start ew-muted" }, "−∞ → 0"));
            continue;
          }
          svg.append(E.s("rect", { "data-k": "r" + i, x, y, width: Math.max(BW * A[i][j], 1), height: 22, rx: 3, class: "ew-bar-0" }));
          svg.append(E.s("text", { x: x + BW * A[i][j] + 6, y: y + 12, class: "ew-s ew-start" }, f(A[i][j], 2)));
        }
      }
      svg.append(E.s("text", { x: 320, y: 216, class: "ew-s ew-c ew-muted" }, "棒の長さ＝注意の重み（各行の合計は 1）。行がクエリ、列がキー"));

      const L = 3; // 式の行には最後のクエリ q₄ を使う
      const list = (v) => "(" + v.map((a) => f(a, 2)).join(",\\ ") + ")";
      const flat = S.flat();
      const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
      const sd = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
      const mx = A.map((row) => Math.max(...row));
      const avgMax = mx.reduce((a, b) => a + b, 0) / NT;
      E.formulaRows(rows, svg, [
        {
          tex: scale.get()
            ? "s_{4j}=\\dfrac{\\mathbf{q}_4\\cdot\\mathbf{k}_j}{\\sqrt{d_k}}"
            : "s_{4j}=\\mathbf{q}_4\\cdot\\mathbf{k}_j",
          num: "s_{4\\cdot}=" + list(S[L]),
          hl: ["r3"],
        },
        {
          tex: "a_{4j}=\\dfrac{e^{s_{4j}+M_{4j}}}{\\sum_l e^{s_{4l}+M_{4l}}}",
          num: "a_{4\\cdot}=" + list(A[L]),
          hl: ["r3"],
        },
        {
          tex: "\\overline{\\max_j a_{ij}}",
          num: "=" + f(avgMax, 2) + "\\quad(1\\text{ に近いほど one-hot})",
          hl: ["r0", "r1", "r2", "r3"],
        },
        {
          tex: "\\mathrm{std}(s_{ij})",
          num: "=" + f(sd, 2) + "\\quad(\\sqrt{d_k}=" + f(Math.sqrt(dk), 2) + ")",
          hl: ["r0", "r1", "r2", "r3"],
        },
      ]);
    }
    draw();
  });
})();
