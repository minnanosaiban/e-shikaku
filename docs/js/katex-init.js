// arithmatex（generic）が出力した数式を KaTeX で描画する
function renderMath(root) {
  renderMathInElement(root, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
}

if (window.document$) {
  document$.subscribe(({ body }) => renderMath(body));
} else {
  document.addEventListener("DOMContentLoaded", () => renderMath(document.body));
}
