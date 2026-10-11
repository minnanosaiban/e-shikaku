// タブのドロップダウン（「裁判文書公開」等）。開閉そのものはネイティブの
// <details>/<summary>（overrides/partials/tabs-item.html）がブラウザの機能として行うので、
// このJSはあくまで補助（他のドロップダウンが開いていたら閉じる／外側クリックで閉じる）。
// 独自クリック処理で開閉させていた前の版は、開発機では動いたがユーザーの実機Chromeでは
// 反応しなかった（原因不明）ため、確実に動くネイティブな仕組みに切り替えた（2026-09-22）。
// この補助JSが万一効かなくても、開閉・ページ遷移という核心の機能には影響しない。
document.addEventListener("DOMContentLoaded", function () {
  var details = document.querySelectorAll(".md-tabs__dropdown-details");
  if (!details.length) return;

  details.forEach(function (d) {
    d.addEventListener("toggle", function () {
      if (!d.open) return;
      details.forEach(function (other) {
        if (other !== d) other.open = false;
      });
    });
  });

  document.addEventListener("click", function (e) {
    details.forEach(function (d) {
      if (d.open && !d.contains(e.target)) d.open = false;
    });
  });
});
