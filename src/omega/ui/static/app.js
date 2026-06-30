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
  function appendLine(el, text, opts) {
    if (el.childNodes.length) el.appendChild(document.createElement("br"));
    var node = opts && opts.strong ? document.createElement("strong") : document.createElement("span");
    if (opts && opts.muted) node.style.opacity = ".7";
    node.textContent = text == null ? "" : String(text);
    el.appendChild(node);
  }
  function setTipLines(el, lines) {
    el.replaceChildren();
    lines.forEach(function (line) {
      if (!line || line.text == null || line.text === "") return;
      appendLine(el, line.text, line);
    });
  }
  document.querySelectorAll(".chart-dot").forEach(function (dot) {
    dot.addEventListener("mouseenter", function () {
      var t = ensureTip();
      var label = dot.getAttribute("data-label") || "";
      var m = dot.getAttribute("data-model");
      var k = dot.getAttribute("data-market");
      setTipLines(t, [
        { text: label, strong: true },
        m ? { text: "Omega: " + m + "%" } : null,
        k ? { text: "Market: " + k + "%" } : null,
      ]);
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
    var intentHeaders = { "X-Omega-Enrich-Intent": "operator-console" };
    var jsonIntentHeaders = {
      "Content-Type": "application/json",
      "X-Omega-Enrich-Intent": "operator-console"
    };
    function setStatus(t) { if (status) status.textContent = t; }
    function clearBody() { if (body) body.textContent = ""; }

    function addText(target, tag, text) {
      var el = document.createElement(tag);
      el.textContent = text;
      target.appendChild(el);
      return el;
    }
    function appendMarkdown(target, md) {
      var list = null;
      (md || "").split("\n").forEach(function (line) {
        if (/^### /.test(line)) { list = null; addText(target, "h5", line.slice(4)); }
        else if (/^## /.test(line)) { list = null; addText(target, "h4", line.slice(3)); }
        else if (/^- /.test(line)) {
          if (!list) { list = document.createElement("ul"); target.appendChild(list); }
          addText(list, "li", line.slice(2));
        } else if (line.trim()) { list = null; addText(target, "p", line); }
      });
    }
    function appendFeedback(id) {
      var fb = document.createElement("div");
      fb.className = "deepdive-feedback";
      fb.setAttribute("data-eid", id);
      var prompt = document.createElement("span");
      prompt.className = "muted small";
      prompt.textContent = "Was this useful?";
      fb.appendChild(prompt);
      ["1", "-1"].forEach(function (rate) {
        var b = document.createElement("button");
        b.type = "button";
        b.setAttribute("data-rate", rate);
        b.textContent = rate === "1" ? "Yes" : "No";
        fb.appendChild(document.createTextNode(" "));
        fb.appendChild(b);
      });
      fb.appendChild(document.createTextNode(" "));
      var msg = document.createElement("span");
      msg.className = "deepdive-fbmsg muted small";
      fb.appendChild(msg);
      body.appendChild(fb);
      wireFeedback(id);
    }
    function wireFeedback(id) {
      var fb = body.querySelector(".deepdive-feedback");
      if (!fb) return;
      fb.querySelectorAll("button[data-rate]").forEach(function (b) {
        b.addEventListener("click", function () {
          fetch("/enrich/enrichments/" + id + "/feedback", {
            method: "POST", headers: jsonIntentHeaders,
            body: JSON.stringify({ user_rating: parseInt(b.getAttribute("data-rate"), 10) })
          }).then(function (r) {
            if (!r.ok) throw new Error();
            var m = fb.querySelector(".deepdive-fbmsg"); if (m) m.textContent = "thanks";
          }).catch(function () {
            var m = fb.querySelector(".deepdive-fbmsg"); if (m) m.textContent = "feedback failed";
          });
        });
      });
    }
    function renderRecord(rec) {
      if (!rec) { setStatus("not generated"); return; }
      if (rec.status === "completed" && rec.narrative_md) {
        clearBody();
        appendMarkdown(body, rec.narrative_md);
        appendFeedback(rec.id);
        setStatus("completed" + (rec.provider ? " · " + rec.provider : ""));
      } else if (rec.status === "failed") {
        setStatus("failed: " + (rec.error || "unknown"));
      } else { setStatus(rec.status || ""); }
    }
    function poll(eid, tries) {
      if (tries <= 0) { setStatus("timed out"); if (genBtn) genBtn.disabled = false; return; }
      fetch("/enrich/enrichments/" + eid).then(function (r) {
        if (!r.ok) throw new Error();
        return r.json();
      }).then(function (rec) {
        if (rec.status === "completed" || rec.status === "failed") {
          renderRecord(rec); if (genBtn) genBtn.disabled = false;
        } else { setStatus(rec.status || "running"); setTimeout(function () { poll(eid, tries - 1); }, 1200); }
      }).catch(function () { setStatus("error polling"); if (genBtn) genBtn.disabled = false; });
    }
    if (genBtn) {
      genBtn.addEventListener("click", function () {
        genBtn.disabled = true; setStatus("queued"); clearBody();
        fetch("/enrich/traces/" + encodeURIComponent(traceId), {
          method: "POST",
          headers: intentHeaders
        })
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
  // Generic display-only tooltip binder for the V2 visuals. Each handler only
  // reads data-* attributes the server already computed — no derivation here.
  function bindTooltip(el, linesFn) {
    el.addEventListener("mouseenter", function () {
      var t = ensureTip();
      setTipLines(t, linesFn(el));
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
      return [
        { text: label, strong: true },
        { text: val },
        unit ? { text: unit, muted: true } : null,
      ];
    });
  });

  // Reliability diagram dots (model bucket vs realized hit rate).
  document.querySelectorAll(".reliability-dot").forEach(function (d) {
    bindTooltip(d, function (el) {
      return [
        { text: el.getAttribute("data-rel-label") || "", strong: true },
        { text: "model " + el.getAttribute("data-rel-model") + "%" },
        { text: "realized " + el.getAttribute("data-rel-hit") + "%" },
        { text: "n=" + el.getAttribute("data-rel-n") },
      ];
    });
  });

  // CLV scatter dots (closing-line value vs net result).
  document.querySelectorAll(".scatter-dot").forEach(function (d) {
    bindTooltip(d, function (el) {
      var st = el.getAttribute("data-sc-status");
      return [
        { text: el.getAttribute("data-sc-label") || "", strong: true },
        { text: "CLV " + el.getAttribute("data-sc-clv") },
        { text: "net " + el.getAttribute("data-sc-pnl") },
        st ? { text: st } : null,
      ];
    });
  });
})();
