// Minimal progressive enhancement for the read-only console.
// The console itself never mutates data. The only network calls are the optional
// Deep Dive panel, which talks to the SEPARATE, opt-in enrichment service mounted
// at /enrich (omega-enrich) — generation POSTs land there, never on console data.
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

  // ---- Deep Dive (optional /enrich enrichment service) --------------------
  (function () {
    var panel = document.querySelector(".deepdive[data-trace]");
    if (!panel) return;
    var traceId = panel.getAttribute("data-trace");
    var body = panel.querySelector("#deepdive-body");
    var status = panel.querySelector("#deepdive-status");
    var genBtn = panel.querySelector("#deepdive-generate");
    function setStatus(t) { if (status) status.textContent = t; }

    function mdToHtml(md) {
      var esc = (md || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      var out = [], inList = false;
      esc.split("\n").forEach(function (line) {
        if (/^### /.test(line)) { if (inList){out.push("</ul>");inList=false;} out.push("<h5>" + line.slice(4) + "</h5>"); }
        else if (/^## /.test(line)) { if (inList){out.push("</ul>");inList=false;} out.push("<h4>" + line.slice(3) + "</h4>"); }
        else if (/^- /.test(line)) { if (!inList){out.push("<ul>");inList=true;} out.push("<li>" + line.slice(2) + "</li>"); }
        else if (line.trim()) { if (inList){out.push("</ul>");inList=false;} out.push("<p>" + line + "</p>"); }
      });
      if (inList) out.push("</ul>");
      return out.join("");
    }
    function feedbackHtml(id) {
      return '<div class="deepdive-feedback" data-eid="' + id + '">' +
        '<span class="muted small">Was this useful?</span> ' +
        '<button type="button" data-rate="1">\u{1F44D}</button> ' +
        '<button type="button" data-rate="-1">\u{1F44E}</button> ' +
        '<span class="deepdive-fbmsg muted small"></span></div>';
    }
    function wireFeedback(id) {
      var fb = body.querySelector(".deepdive-feedback");
      if (!fb) return;
      fb.querySelectorAll("button[data-rate]").forEach(function (b) {
        b.addEventListener("click", function () {
          fetch("/enrich/enrichments/" + id + "/feedback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_rating: parseInt(b.getAttribute("data-rate"), 10) })
          }).then(function () {
            var m = fb.querySelector(".deepdive-fbmsg"); if (m) m.textContent = "thanks";
          }).catch(function () {});
        });
      });
    }
    function renderRecord(rec) {
      if (!rec) { setStatus("not generated"); return; }
      if (rec.status === "completed" && rec.narrative_md) {
        body.innerHTML = mdToHtml(rec.narrative_md) + feedbackHtml(rec.id);
        wireFeedback(rec.id);
        setStatus("completed" + (rec.provider ? " · " + rec.provider : ""));
      } else if (rec.status === "failed") {
        setStatus("failed: " + (rec.error || "unknown"));
      } else { setStatus(rec.status || ""); }
    }
    function poll(eid, tries) {
      if (tries <= 0) { setStatus("timed out"); if (genBtn) genBtn.disabled = false; return; }
      fetch("/enrich/enrichments/" + eid).then(function (r) { return r.json(); }).then(function (rec) {
        if (rec.status === "completed" || rec.status === "failed") {
          renderRecord(rec); if (genBtn) genBtn.disabled = false;
        } else { setStatus(rec.status || "running"); setTimeout(function () { poll(eid, tries - 1); }, 1200); }
      }).catch(function () { setStatus("error polling"); if (genBtn) genBtn.disabled = false; });
    }
    if (genBtn) {
      genBtn.addEventListener("click", function () {
        genBtn.disabled = true; setStatus("queued"); body.innerHTML = "";
        fetch("/enrich/traces/" + encodeURIComponent(traceId), { method: "POST" })
          .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
          .then(function (j) { poll(j.enrichment_id, 30); })
          .catch(function () { setStatus("enrichment service not running (start omega-enrich)"); genBtn.disabled = false; });
      });
    }
    fetch("/enrich/traces/" + encodeURIComponent(traceId) + "/latest")
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (j) { if (j.latest) renderRecord(j.latest); })
      .catch(function () { setStatus("not generated"); });
  })();
})();
