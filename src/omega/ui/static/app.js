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

  // Keyboard navigation for review cards / rails. Pure DOM focus movement —
  // ArrowUp/Down move between focusable items inside a [data-keynav] container,
  // Enter follows the item's first link. No network calls, no computation.
  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp" && ev.key !== "Enter") return;
    var active = document.activeElement;
    if (!active || typeof active.matches !== "function") return;
    if (!active.matches("[data-keynav] [tabindex]")) return;
    var container = active.closest("[data-keynav]");
    if (!container) return;
    var items = Array.prototype.slice.call(container.querySelectorAll("[tabindex]"));
    var idx = items.indexOf(active);
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      (items[idx + 1] || items[0]).focus();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      (items[idx - 1] || items[items.length - 1]).focus();
    } else if (ev.key === "Enter") {
      var link = active.querySelector("a[href]");
      if (link) {
        ev.preventDefault();
        window.location.href = link.href;
      }
    }
  });

  // Calibration chart hover tooltip. Display-only: it reads data-* attributes
  // the server already computed and never derives a betting value.
  var tip = null;
  function ensureTip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tooltip";
      document.body.appendChild(tip);
    }
    return tip;
  }
  document.querySelectorAll(".chart-dot").forEach(function (dot) {
    dot.addEventListener("mouseenter", function () {
      var t = ensureTip();
      var label = dot.getAttribute("data-label") || "";
      var m = dot.getAttribute("data-model");
      var k = dot.getAttribute("data-market");
      var html = "<strong>" + label + "</strong>";
      if (m) html += "<br>Omega: " + m + "%";
      if (k) html += "<br>Market: " + k + "%";
      t.innerHTML = html;
      t.style.opacity = "1";
    });
    dot.addEventListener("mousemove", function (ev) {
      var t = ensureTip();
      t.style.left = ev.clientX + 12 + "px";
      t.style.top = ev.clientY + 12 + "px";
    });
    dot.addEventListener("mouseleave", function () {
      if (tip) tip.style.opacity = "0";
    });
  });

  // Generic display-only tooltip binder for the V2 visuals. Each handler only
  // reads data-* attributes the server already computed — no derivation here.
  function bindTooltip(el, htmlFn) {
    el.addEventListener("mouseenter", function () {
      var t = ensureTip();
      t.innerHTML = htmlFn(el);
      t.style.opacity = "1";
    });
    el.addEventListener("mousemove", function (ev) {
      var t = ensureTip();
      t.style.left = ev.clientX + 12 + "px";
      t.style.top = ev.clientY + 12 + "px";
    });
    el.addEventListener("mouseleave", function () {
      if (tip) tip.style.opacity = "0";
    });
  }

  // Comparison strip dots (dumbbell / ribbon).
  document.querySelectorAll(".strip-dot").forEach(function (d) {
    bindTooltip(d, function (el) {
      var label = el.getAttribute("data-strip-label") || "";
      var val = el.getAttribute("data-strip-value") || "";
      var unit = el.getAttribute("data-strip-unit") || "";
      return "<strong>" + label + "</strong><br>" + val + (unit ? "<br><span style=\"opacity:.7\">" + unit + "</span>" : "");
    });
  });

  // Reliability diagram dots (model bucket vs realized hit rate).
  document.querySelectorAll(".reliability-dot").forEach(function (d) {
    bindTooltip(d, function (el) {
      return "<strong>" + (el.getAttribute("data-rel-label") || "") + "</strong>" +
        "<br>model " + el.getAttribute("data-rel-model") + "%" +
        "<br>realized " + el.getAttribute("data-rel-hit") + "%" +
        "<br>n=" + el.getAttribute("data-rel-n");
    });
  });

  // CLV scatter dots (closing-line value vs net result).
  document.querySelectorAll(".scatter-dot").forEach(function (d) {
    bindTooltip(d, function (el) {
      var st = el.getAttribute("data-sc-status");
      return "<strong>" + (el.getAttribute("data-sc-label") || "") + "</strong>" +
        "<br>CLV " + el.getAttribute("data-sc-clv") +
        "<br>net " + el.getAttribute("data-sc-pnl") +
        (st ? "<br>" + st : "");
    });
  });
})();
