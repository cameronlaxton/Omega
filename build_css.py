import re


def main():
    # Read the mockup HTML
    with open(r"c:\repos\Omega\docs\design\operator_console_mockup.html", encoding="utf-8") as f:
        html = f.read()

    # Extract the <style> block
    style_match = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    if not style_match:
        print("Could not find <style> in mockup.")
        return

    mockup_css = style_match.group(1).strip()

    # We want to remove the specific mockup labels CSS as it's not needed in production
    mockup_css = re.sub(
        r"/\* ---- Label for mockup sections ---- \*/.*", "", mockup_css, flags=re.DOTALL
    )

    # Now read the original styles.css to extract specific missing styles
    with open(r"c:\repos\Omega\src\omega\ui\static\styles.css", encoding="utf-8") as f:
        f.read()

    # Sections to port over:
    # 1. KV table styles (with new colors)
    # 2. Block, Json, Details
    # 3. Split layout
    # 4. Banner
    # 5. Legend
    # 6. Pager
    # 7. QA and health
    # 8. Diagnostics tiles / pstatus
    # 9. Review Queue buckets
    # 10. Print

    additional_css = """
/* ==========================================================================
   LEGACY / PAGE-SPECIFIC COMPONENTS PORTED TO NEW DESIGN
   ========================================================================== */

/* Extracted-field value + provenance + computed badge (for B.1) */
.fv { cursor: help; }
.fv.computed { color: var(--accent); }
.cbadge {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  background: #16263a; color: var(--accent); border: 1px solid #1d3a59;
  padding: 0 4px; border-radius: 3px; vertical-align: middle;
}
.small { font-size: 0.8rem; }
.rawdetails { margin-top: 0.6rem; }

/* KV Tables */
table.kv { border-collapse: collapse; background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); width: 100%; }
table.kv th, table.kv td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border-subtle); font-size: 13px; }
table.kv th { color: var(--ink-3); font-weight: 600; white-space: nowrap; width: 1%; background: var(--surface-2); }

/* Block & Pre (JSON/Prose) */
.block { background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 1rem 1.25rem; margin: 1rem 0; }
.count { color: var(--ink-3); font-weight: 400; font-size: 12px; }
pre.json, pre.prose {
  background: #0b0d11; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  padding: 0.8rem 1rem; overflow: auto; max-height: 32rem;
  font-family: var(--mono); font-size: 12px;
}
pre.prose { white-space: pre-wrap; font-family: var(--sans); }
details summary { cursor: pointer; color: var(--accent); outline: none; margin-bottom: 0.5rem; }
.linklist { margin: 0.3rem 0; padding-left: 1.1rem; }

/* Split panel for sessions */
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; }
@media (max-width: 900px) { .split { grid-template-columns: 1fr; } }
.sidecar { border-left: 3px solid var(--warn); padding-left: 1rem; }
.db { border-left: 3px solid var(--pos); padding-left: 1rem; }

/* Banners & Notes */
.banner { padding: 0.8rem 1.2rem; border-radius: var(--radius-sm); margin: 0.8rem 0; }
.banner.warn { background: var(--warn-bg); border: 1px solid rgba(251, 191, 36, 0.3); color: var(--warn); }
.srcnote { color: var(--ink-3); font-size: 12px; margin: 0.4rem 0 0.8rem; }

/* Legends */
.legend { margin: 1.2rem 0; padding: 0.8rem 1.2rem; background: var(--surface-1); border: 1px dashed var(--border-subtle); border-radius: var(--radius-sm); font-size: 12px; }
.legend strong { display: block; margin-bottom: 0.4rem; color: var(--ink-3); }
.legend-row { display: inline-flex; gap: 6px; align-items: center; margin: 0 1rem 0.4rem 0; }

/* Pager */
.pager { display: flex; justify-content: space-between; align-items: center; margin: 1rem 0; color: var(--ink-3); font-size: 13px; }
.pagelinks a { padding: 5px 12px; background: var(--surface-2); border-radius: var(--radius-xs); margin-left: 0.4rem; color: var(--ink); text-decoration: none; transition: background 0.15s ease; }
.pagelinks a:hover { background: var(--border-subtle); }
.pagelinks .disabled { padding: 5px 12px; color: var(--ink-3); margin-left: 0.4rem; cursor: not-allowed; opacity: 0.5; }

/* QA / Health specifics */
.qa-pass { color: var(--pos); }
.qa-fail { color: var(--neg); font-weight: 700; }
.qa-unknown { color: var(--warn); }
.health { border-left: 3px solid var(--accent); padding-left: 1rem; margin-top: 1rem;}
.hstats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin: 0.4rem 0; }
.hstats.sub { color: var(--ink-3); font-size: 12px; }
.hstat em { font-style: normal; font-weight: 700; color: var(--ink); }
.hstat.neg { color: var(--neg); }
.nowrap { white-space: nowrap; }

/* Warning chips */
.warnchips { display: flex; flex-direction: column; gap: 0.4rem; margin: 0.6rem 0 0.3rem; }
.wchip {
  font-size: 12px; padding: 4px 10px; border-radius: var(--radius-xs);
  border: 1px solid var(--border-subtle); background: var(--surface-1);
}
.wchip strong { text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; margin-right: 6px; }
.wchip.sev-info { border-color: rgba(77, 163, 255, 0.3); color: var(--ink-2); }
.wchip.sev-warn { border-color: rgba(251, 191, 36, 0.3); background: var(--warn-bg); color: var(--warn); }
.wchip.sev-fail { border-color: rgba(248, 113, 113, 0.3); background: var(--neg-bg); color: var(--neg); }
.wchip.sev-unknown { border-color: var(--border-subtle); color: var(--ink-3); }

/* Diagnostics tiles */
.tiles { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0 1rem; }
.tile {
  display: flex; flex-direction: column; gap: 2px; min-width: 120px;
  background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  padding: 0.6rem 1rem; transition: transform 0.15s ease, border-color 0.15s ease;
}
.tile:hover { border-color: var(--border); transform: translateY(-1px); }
.tlabel { color: var(--ink-3); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.tval { font-size: 20px; font-weight: 500; font-family: var(--mono); }
.tval.pos { color: var(--pos); }
.tval.muted { color: var(--ink-3); }

/* Calibration profiles */
.pstatus {
  font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 4px; display: inline-block;
}
.pstatus-production { background: var(--pos-bg); color: var(--pos); }
.pstatus-candidate { background: var(--accent-dim); color: var(--accent); }
.pstatus-archived { background: var(--surface-2); color: var(--ink-3); }
.pstatus-rejected { background: var(--neg-bg); color: var(--neg); }

table.grid tr.profile.active { background: rgba(52, 211, 153, 0.05); }
table.grid tr.profile.active:hover { background: rgba(52, 211, 153, 0.08); }
.badge-prod {
  font-size: 9px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  background: var(--pos-bg); color: var(--pos); border: 1px solid rgba(52, 211, 153, 0.2);
  padding: 1px 6px; border-radius: 4px; vertical-align: middle;
}

/* Review Queue buckets */
.bucket { background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 1rem 1.25rem; margin-bottom: 1rem; }
.bucket.bucket-info { border-left: 3px solid var(--accent); }
.bucket.bucket-warn { border-left: 3px solid var(--warn); }
.bucket.bucket-fail { border-left: 3px solid var(--neg); }
.bucket-count {
  display: inline-block; min-width: 1.5rem; text-align: center;
  font-weight: 700; font-size: 12px; padding: 2px 8px; border-radius: 999px;
  background: var(--surface-2); color: var(--ink);
}
.bucket-count.sev-warn { color: var(--warn); background: var(--warn-bg); }
.bucket-count.sev-fail { color: var(--neg); background: var(--neg-bg); }

/* EvCov */
.evcov { display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem; margin: 0.2rem 0; }
.evstat { color: var(--ink-3); }
.evstat strong { color: var(--ink); }
.evcol { text-align: right; white-space: nowrap; }

/* Helper classes */
.empty { color: var(--ink-3); text-align: center; font-style: italic; padding: 2rem; }
.future { margin: 2rem 0; }
.ph-list { list-style: none; padding: 0; display: flex; gap: 0.75rem; flex-wrap: wrap; }
.ph-list li { background: var(--surface-1); border: 1px dashed var(--border-subtle); border-radius: var(--radius-sm); padding: 0.5rem 0.8rem; color: var(--ink-3); font-size: 13px; }
.ph-list em { color: var(--warn); font-style: normal; font-size: 11px; margin-left: 6px; }

/* Print */
@media print {
  :root { --base: #fff; --ink: #000; --surface-1: #f8f8f8; --surface-2: #f0f0f0; --ink-3: #555; --border-subtle: #ccc; --border: #999; --accent: #000; --pos: #000; --warn: #000; --neg: #000;}
  body { background: #fff; color: #000; font-size: 11pt; }
  .bg-glow { display: none; }
  a { color: #000; text-decoration: underline; }
  .topbar { position: static; border-bottom: 2px solid #000; background: #fff; }
  .nav, .navph, .ro-badge, .status-dot, .status-text { display: none; }
  .filter-bar, .pager, .pagelinks, .chips, .filter-clear { display: none; }
  .content { padding: 0.5rem; max-width: 100%; }
  table.grid { font-size: 9pt; }
  table.grid thead th { position: static; background: #eee; border-bottom: 2px solid #000; }
  pre.json, pre.prose { max-height: none; overflow: visible; border: 1px solid #ccc; background: #fff; }
  .footer { border-top: 1px solid #ccc; }
  .src-badge { background: #eee; color: #333; border: 1px solid #ccc; }
  .rec-card, .wchip { break-inside: avoid; border: 1px solid #ccc; }
  .cbadge { background: #eee; color: #333; border: 1px solid #ccc; }
}
"""

    final_css = mockup_css + "\n" + additional_css

    with open(r"c:\repos\Omega\src\omega\ui\static\styles.css", "w", encoding="utf-8") as f:
        f.write(final_css)

    print("styles.css updated successfully.")


if __name__ == "__main__":
    main()
