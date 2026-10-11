// 物体検出：IoU（2つの矩形の重なり）と NMS（重複した検出の整理）
(function () {
  const E = window.ENote;

  // 矩形は { x, y, w, h }（左上と幅・高さ）
  const area = (b) => Math.max(0, b.w) * Math.max(0, b.h);
  const inter = (a, b) => {
    const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
    const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
    return { w: Math.max(0, w), h: Math.max(0, h), x: Math.max(a.x, b.x), y: Math.max(a.y, b.y) };
  };
  const iou = (a, b) => {
    const i = inter(a, b);
    const ia = i.w * i.h;
    const u = area(a) + area(b) - ia;
    return u > 0 ? ia / u : 0;
  };

  // 貪欲な NMS：スコア降順に並べ、すでに残した箱と IoU が θ を超えるものを抑制する
  // boxes: [{x,y,w,h,s}] → 各箱について { idx, keep, by, v }（by は抑制した箱、v はその IoU）をスコア降順で返す
  const nms = (boxes, theta) => {
    const order = boxes.map((_, i) => i).sort((p, q) => boxes[q].s - boxes[p].s);
    const kept = [];
    return order.map((i) => {
      let by = -1, v = 0;
      for (const k of kept) {
        const u = iou(boxes[i], boxes[k]);
        if (u > theta && u > v) { by = k; v = u; }
      }
      if (by < 0) kept.push(i);
      return { idx: i, keep: by < 0, by, v };
    });
  };

  E.visionIoU = iou;
  E.visionNms = nms;

  // ---- 図 ----
  const W = 640, H = 320;

  E.register("vision-iou-nms", (el) => {
    const top = E.h("div", { class: "ew-bar" });
    el.append(top);
    const mode = E.toggle(
      top,
      [{ value: "iou", label: "IoU" }, { value: "nms", label: "NMS" }],
      "iou",
      () => build()
    );
    const svg = E.svg(el, W, H, "矩形の重なりから IoU を求める図、および重複した検出を NMS で整理する図");
    const rows = E.h("div", { class: "ew-rows" });
    el.append(rows);
    const sliderBox = E.h("div");
    el.append(sliderBox);
    let sl = null;

    function clear() {
      while (svg.childNodes.length > 2) svg.lastChild.remove(); // title と defs は残す
    }

    function build() {
      sliderBox.innerHTML = "";
      if (mode.get() === "iou") {
        sl = E.sliders(
          sliderBox,
          [
            { key: "dx", label: "d_x", min: -6, max: 10, step: 0.5, value: 3, digits: 1 },
            { key: "dy", label: "d_y", min: -5, max: 8, step: 0.5, value: 2, digits: 1 },
            { key: "s", label: "s", min: 0.5, max: 1.5, step: 0.05, value: 1, digits: 2 },
          ],
          () => draw()
        );
      } else {
        sl = E.sliders(
          sliderBox,
          [
            { key: "th", label: "\\theta", min: 0.05, max: 0.95, step: 0.05, value: 0.5, digits: 2 },
            { key: "d", label: "\\text{物体の間隔}", min: 20, max: 200, step: 5, value: 140, digits: 0 },
          ],
          () => draw()
        );
      }
      draw();
    }

    // ---------- IoU ----------
    const U = 13, OX = 190, OY = 70; // 1 単位 = 13 px、A の左上 = (OX, OY)
    const px = (u) => OX + u * U, py = (u) => OY + u * U;

    function drawIoU() {
      const { dx, dy, s } = sl.get();
      const f = E.fmt;
      const A = { x: 0, y: 0, w: 9, h: 8 };
      const B = { x: dx, y: dy, w: 8 * s, h: 7 * s };
      const I = inter(A, B);
      const aA = area(A), aB = area(B), aI = I.w * I.h, aU = aA + aB - aI;
      const v = aU > 0 ? aI / aU : 0;

      // 共通部分（先に描いて枠線を上に重ねる）
      svg.append(E.s("rect", { "data-k": "inter", x: px(I.x), y: py(I.y), width: I.w * U, height: I.h * U, class: "ew-bar-0" }));
      svg.append(E.s("rect", { "data-k": "A", x: px(A.x), y: py(A.y), width: A.w * U, height: A.h * U, class: "ew-curve-f" }));
      svg.append(E.s("rect", { "data-k": "B", x: px(B.x), y: py(B.y), width: B.w * U, height: B.h * U, class: "ew-curve-d" }));

      // 右側の凡例と IoU のメーター（図形は x < 480 に収まる）
      svg.append(E.s("text", { x: 500, y: 40, class: "ew-s ew-start" }, "実線：A（正解）"));
      svg.append(E.s("text", { x: 500, y: 62, class: "ew-s ew-start" }, "破線：B（予測）"));
      svg.append(E.s("text", { x: 500, y: 84, class: "ew-s ew-start" }, "灰の面：A∩B"));
      svg.append(E.s("text", { x: 500, y: 150, class: "ew-t ew-start" }, "IoU = " + f(v)));
      svg.append(E.s("rect", { x: 500, y: 170, width: 120, height: 10, class: "ew-node" }));
      svg.append(E.s("rect", { "data-k": "iou", x: 500, y: 170, width: Math.max(0, Math.min(1, v)) * 120, height: 10, class: "ew-bar-1" }));
      svg.append(E.s("text", { x: 500, y: 196, class: "ew-s ew-start ew-muted" }, "0"));
      svg.append(E.s("text", { x: 620, y: 196, class: "ew-s ew-end ew-muted" }, "1"));

      E.formulaRows(rows, svg, [
        { tex: "|A|=w_Ah_A", num: "=9\\times8=" + f(aA, 1), hl: ["A"] },
        { tex: "|B|=w_Bh_B", num: "=" + f(B.w, 2) + "\\times" + f(B.h, 2) + "=" + f(aB, 2), hl: ["B"] },
        { tex: "|A\\cap B|=\\max(0,\\Delta x)\\cdot\\max(0,\\Delta y)", num: "=" + f(I.w, 2) + "\\times" + f(I.h, 2) + "=" + f(aI, 2), hl: ["inter"] },
        { tex: "|A\\cup B|=|A|+|B|-|A\\cap B|", num: "=" + f(aA, 1) + "+" + f(aB, 2) + "-" + f(aI, 2) + "=" + f(aU, 2), hl: ["A", "B", "inter"] },
        { tex: "\\mathrm{IoU}=\\dfrac{|A\\cap B|}{|A\\cup B|}", num: "=\\dfrac{" + f(aI, 2) + "}{" + f(aU, 2) + "}=" + f(v), hl: ["inter", "iou"] },
      ]);
    }

    // ---------- NMS ----------
    // 2 つの物体のまわりの 5 つの検出（相対位置とスコアは固定。物体の間隔だけ動かす）
    const BASE = [
      { ox: 0, oy: 0, w: 110, h: 130, s: 0.92, obj: 0 },
      { ox: 12, oy: 9, w: 112, h: 124, s: 0.81, obj: 0 },
      { ox: -14, oy: 14, w: 104, h: 118, s: 0.64, obj: 0 },
      { ox: 0, oy: 0, w: 100, h: 130, s: 0.88, obj: 1 },
      { ox: -10, oy: -10, w: 96, h: 120, s: 0.7, obj: 1 },
    ];
    const C0 = { x: 150, y: 185 }; // 物体 1 の中心

    function drawNMS() {
      const { th, d } = sl.get();
      const f = E.fmt;
      const boxes = BASE.map((b) => {
        const cx = C0.x + (b.obj ? d : 0) + b.ox, cy = C0.y + b.oy;
        return { x: cx - b.w / 2, y: cy - b.h / 2, w: b.w, h: b.h, s: b.s };
      });
      const res = nms(boxes, th);
      const rank = {};
      res.forEach((r, n) => (rank[r.idx] = n + 1));

      // 抑制された箱は薄く、残った箱は濃く描く
      res.slice().reverse().forEach((r) => {
        const b = boxes[r.idx];
        svg.append(E.s("rect", { "data-k": "b" + r.idx, x: b.x, y: b.y, width: b.w, height: b.h, class: r.keep ? "ew-curve-f" : "ew-edge" }));
      });
      // 箱の番号（スコアの順位）は上辺の内側に置く（箱ごとに横にずらして、番号どうしを重ねない）
      res.forEach((r) => {
        const b = boxes[r.idx];
        svg.append(E.s("text", { x: b.x + 14 + 20 * r.idx, y: b.y + 13, class: "ew-s ew-c" }, "①②③④⑤"[rank[r.idx] - 1]));
      });

      svg.append(E.s("text", { x: 500, y: 40, class: "ew-s ew-start" }, "濃い実線：残す"));
      svg.append(E.s("text", { x: 500, y: 62, class: "ew-s ew-start" }, "薄い線：抑制"));
      svg.append(E.s("text", { x: 500, y: 84, class: "ew-s ew-start" }, "番号：スコア順"));
      const nKeep = res.filter((r) => r.keep).length;
      svg.append(E.s("text", { x: 500, y: 150, class: "ew-t ew-start" }, "残った数 " + nKeep));

      const circ = "①②③④⑤";
      const r = res.map((q, n) => {
        const name = "\\text{" + circ[n] + "}\\ (s=" + f(boxes[q.idx].s, 2) + ")";
        if (q.keep && n === 0) return { tex: name, num: "\\text{最高スコアなので残す}", hl: ["b" + q.idx] };
        if (q.keep) return { tex: name, num: "\\text{残した箱との IoU がすべて }\\le" + f(th, 2) + "\\Rightarrow\\text{残す}", hl: ["b" + q.idx] };
        return {
          tex: name,
          num: "\\mathrm{IoU}(\\text{" + circ[res.findIndex((z) => z.idx === q.by)] + "})=" + f(q.v) + ">" + f(th, 2) + "\\Rightarrow\\text{抑制}",
          hl: ["b" + q.idx, "b" + q.by],
        };
      });
      E.formulaRows(rows, svg, r);
    }

    function draw() {
      clear();
      if (mode.get() === "iou") drawIoU();
      else drawNMS();
    }

    build();
  });
})();
