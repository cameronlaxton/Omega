// Minimal progressive enhancement for the read-only console.
// No data mutation, no network calls — just local UI niceties.
(function () {
  "use strict";

  // Submit filter forms automatically when a <select> changes.
  document.querySelectorAll("form.filters select").forEach(function (sel) {
    sel.addEventListener("change", function () {
      var form = sel.closest("form");
      if (form) form.submit();
    });
  });

  // Ctrl/Cmd-clicking a <pre class="json"> selects its contents for easy copy.
  document.querySelectorAll("pre.json").forEach(function (pre) {
    pre.addEventListener("click", function (ev) {
      if (!(ev.ctrlKey || ev.metaKey)) return;
      var range = document.createRange();
      range.selectNodeContents(pre);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
  });
})();
