// 動く図の共通部品
//
// ページ側は <div class="ew" data-widget="名前"></div> を置くだけ。
// 各 widget ファイルは ENote.register("名前", (el) => {...}) で登録する。
//
// 共通の約束事：「数式の1行 ⇄ 図の要素 ⇄ 数値」を連動させる。
// 図（SVG）の要素には data-k="キー" を付け、数式の行は hl: ["キー", ...] で
// 光らせる要素を指定する。
(function () {
  const E = (window.ENote = window.ENote || { widgets: {} });

  E.register = function (name, fn) {
    E.widgets[name] = fn;
    E.mountAll(document);
  };

  E.mountAll = function (root) {
    root.querySelectorAll(".ew[data-widget]").forEach((el) => {
      if (el.dataset.mounted) return;
      const fn = E.widgets[el.dataset.widget];
      if (!fn) return;
      el.dataset.mounted = "1";
      fn(el);
    });
  };

  // 数値の表示（-0.000 を 0.000 にそろえる）
  E.fmt = function (v, n = 3) {
    if (!isFinite(v)) return "—";
    const s = (+v).toFixed(n);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  };

  // 指数表記も使う表示（勾配消失のように桁が大きく変わる値）
  E.fmtSci = function (v) {
    if (v === 0) return "0";
    if (Math.abs(v) >= 0.001) return E.fmt(v, 3);
    const [m, e] = v.toExponential(2).split("e");
    return m + "\\times 10^{" + Number(e) + "}";
  };

  E.tex = function (src, el) {
    try {
      katex.render(src, el, { throwOnError: false });
    } catch (e) {
      el.textContent = src;
    }
  };

  // 要素を作る小さなヘルパー
  E.h = function (tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    [].concat(children).forEach((c) => el.append(c));
    return el;
  };

  // スライダー群。defs: [{ key, label, min, max, step, value }]
  // 戻り値の get() で { key: 数値 } を返す
  E.sliders = function (parent, defs, onInput) {
    const box = E.h("div", { class: "ew-sliders" });
    const inputs = {};
    defs.forEach((d) => {
      const out = E.h("span", { class: "ew-out" });
      const input = E.h("input", {
        type: "range", min: d.min, max: d.max, step: d.step, value: d.value,
        "aria-label": d.label,
      });
      const lab = E.h("span", { class: "ew-lab" });
      E.tex(d.label, lab);
      const sync = () => (out.textContent = E.fmt(input.value, d.digits ?? 2));
      input.addEventListener("input", () => { sync(); onInput(); });
      sync();
      inputs[d.key] = input;
      box.append(E.h("label", { class: "ew-slider" }, [lab, input, out]));
    });
    parent.append(box);
    return {
      get: () => Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, +i.value])),
    };
  };

  // 切り替えボタン。options: [{ value, label }]
  E.toggle = function (parent, options, value, onChange) {
    const box = E.h("div", { class: "ew-toggle", role: "group" });
    let current = value;
    const btns = options.map((o) =>
      E.h("button", { type: "button", class: "ew-btn", onclick: () => set(o.value) }, o.label)
    );
    function set(v) {
      current = v;
      btns.forEach((b, i) => b.setAttribute("aria-pressed", options[i].value === v));
      onChange(v);
    }
    box.append(...btns);
    parent.append(box);
    btns.forEach((b, i) => b.setAttribute("aria-pressed", options[i].value === value));
    return { get: () => current, set };
  };

  // 数式の行。rows: [{ tex, num, hl: [キー] }]
  // 行にマウスを乗せる（またはフォーカスする）と、svg 内の data-k が一致する要素を光らせる
  E.formulaRows = function (box, svg, rows) {
    box.innerHTML = "";
    rows.forEach((r) => {
      const left = E.h("div", { class: "ew-tex" });
      const right = E.h("div", { class: "ew-num" });
      E.tex(r.tex, left);
      if (r.num) E.tex(r.num, right);
      const row = E.h("div", { class: "ew-row", tabindex: "0" }, [left, right]);
      const targets = () =>
        (r.hl || []).flatMap((k) => [...svg.querySelectorAll(`[data-k="${k}"]`)]);
      const on = () => targets().forEach((t) => t.classList.add("ew-hl"));
      const off = () => targets().forEach((t) => t.classList.remove("ew-hl"));
      row.addEventListener("mouseenter", on);
      row.addEventListener("mouseleave", off);
      row.addEventListener("focus", on);
      row.addEventListener("blur", off);
      box.append(row);
    });
  };

  // SVG 要素を作る
  const NS = "http://www.w3.org/2000/svg";
  E.s = function (tag, attrs = {}, text) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text != null) el.textContent = text;
    return el;
  };

  E.svg = function (parent, w, h, title) {
    const svg = E.s("svg", { viewBox: `0 0 ${w} ${h}`, class: "ew-svg", role: "img" });
    svg.append(E.s("title", {}, title));
    const defs = E.s("defs");
    const m = E.s("marker", {
      id: "ew-arrow-" + Math.random().toString(36).slice(2, 8),
      viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6,
      orient: "auto-start-reverse",
    });
    m.append(E.s("path", { d: "M2 1L8 5L2 9", fill: "none", stroke: "context-stroke", "stroke-width": 1.5 }));
    defs.append(m);
    svg.append(defs);
    svg.arrow = `url(#${m.id})`;
    parent.append(svg);
    return svg;
  };

  if (window.document$) document$.subscribe(() => E.mountAll(document));
  else document.addEventListener("DOMContentLoaded", () => E.mountAll(document));
})();
