# Omega Operator Console — Redesign Brief

**For:** Claude Design
**From:** Cam (cameron@laxtonit.com)
**Date:** 2026-06-22
**Status:** Design kickoff

---

## 1. What This Is

The Omega Operator Console is a **read-only web dashboard** for a sports analytics engine. It surfaces analysis traces, bet ledger entries, session QA, calibration status, signal performance, and system diagnostics. Think of it as a mission-control panel for a quantitative betting operation; the operator never writes through the console, only reads and audits.

Current stack: Python FastAPI + Jinja2 server-rendered HTML + vanilla CSS. No JS framework, no charts, no real-time updates. The existing UI is functional but visually sparse; we want to take it to a **sleek, modern, smooth** operator console aesthetic.

---

## 2. Current State (reference files)

All source files live in the repo under `src/omega/ui/`:

| File | What it does |
|---|---|
| `templates/base.html` | Layout shell: sticky top nav, footer, Jinja macros for pagination, provenance badges, filter chips, field rendering, warning chips |
| `templates/index.html` | Landing: 8 card links + runtime health summary |
| `templates/traces.html` | Trace Ledger Explorer (filterable table) |
| `templates/trace_detail.html` | Single trace drill-down: normalized recommendation cards, evidence coverage, predictions, sim distributions, outcomes, bets, QA, closing lines, raw JSON |
| `templates/bets.html` | Bet Ledger Explorer (filterable table) |
| `templates/bet_detail.html` | Single bet drill-down |
| `templates/sessions.html` | Session Review list |
| `templates/session_detail.html` | Session drill-down: split-panel sidecar narrative vs DB-backed traces |
| `templates/diagnostics.html` | System health tiles + calibration/signal summaries |
| `templates/calibration.html` | Calibration profiles with status badges |
| `templates/signals.html` | Signal performance table (direction accuracy, hit rate, brier, cal gap) |
| `templates/review.html` | Review Queue: operator work buckets (ungraded traces, pending bets, problem sessions) |
| `templates/clv.html` | Closing-line value / market movement |
| `static/styles.css` | 280-line dark-theme CSS with full design system (custom properties, all components) |
| `schemas.py` | ~50 Pydantic response models (defines every data shape) |
| `normalizers.py` | Read-only normalization layer for trace payloads |
| `service.py` | `ConsoleService` — all data reading, pagination, filtering |
| `api.py` | GET-only JSON API router |
| `../ops/console_server.py` | FastAPI app, route handlers, static mount, auth middleware |

### Current design tokens (from `styles.css`)

```css
:root {
  --bg: #0f1115;        /* deep navy-black */
  --panel: #171a21;     /* card/panel bg */
  --panel2: #1f242d;    /* elevated panel */
  --ink: #e7e9ee;       /* primary text */
  --muted: #9aa3b2;     /* secondary text */
  --line: #2a313c;      /* borders */
  --accent: #5db0ff;    /* links, active states, accent blue */
  --pos: #4cc38a;       /* positive/green */
  --neg: #f0786f;       /* negative/red */
  --warn: #e7b84b;      /* warning/amber */
}
```

---

## 3. Design Direction

### Vibe
**"Bloomberg Terminal meets Linear meets Vercel Dashboard"** — information-dense but not cluttered, smooth transitions, glass-panel depth, tight typography. The operator should feel like they're in a cockpit, not reading a spreadsheet.

### Key principles

1. **Information density without clutter.** This is a pro tool. Every pixel should earn its place. Don't hide data behind extra clicks; use visual hierarchy to let the eye scan.

2. **Depth and layering.** Use subtle elevation (glass/frosted panels, faint borders, shadow layers) to create a sense of depth. Cards float above the background. Modals and detail panels feel like pulling something forward.

3. **Smooth motion.** Page transitions, filter applications, hover states, expanding sections should all have micro-animations (150-300ms easing). Nothing instant-pops; everything slides or fades.

4. **Status at a glance.** Color-coding for pos/neg/warn should be immediately readable. Use subtle background tints (not just text color) so status registers in peripheral vision.

5. **Monospace where data, proportional where prose.** Numbers, IDs, code, probabilities in a good mono font (JetBrains Mono, Berkeley Mono, or SF Mono). Prose and labels in a clean sans (Inter, SF Pro, Geist).

6. **Provenance is a feature.** Every value's source is visible (hover or badge). This is a trust mechanism; make it elegant, not noisy.

---

## 4. Page-by-Page Notes

### 4.1 Landing / Index (`index.html`)

Current: 8 plain link-cards in a grid + a basic health KV table.

**Redesign goals:**
- Hero area with the Omega wordmark/logo and a one-line system status (green dot + "All systems operational" or amber + issue count)
- Card grid should feel like a Bloomberg launchpad; each card gets an icon, a live stat preview (e.g. "142 traces", "3 pending bets"), and a subtle hover-lift animation
- Runtime health section becomes a compact "system vitals" bar below the cards (horizontal tiles, not a table)
- Consider a subtle animated background (very slow gradient shift or low-opacity particle field) for the hero, keeping it performant

### 4.2 Trace Ledger Explorer (`traces.html`)

Current: filter form + data table.

**Redesign goals:**
- Filters should be a collapsible top bar with inline pills (think Notion/Linear filter bar). Active filters show as removable chips below.
- Table needs horizontal scroll containment, sticky first column (trace_id), alternating row tints, and row hover highlight with a subtle left-border accent
- Confidence tiers and market columns could use color-coded pills instead of plain text
- Evidence column: replace the raw number with a mini horizontal bar or dot-scale (0-10 range)
- Pagination: infinite scroll with a "load more" sentinel, or at minimum a cleaner pager with page-size selector

### 4.3 Trace Detail (`trace_detail.html`)

Current: KV table, recommendation cards, raw JSON blocks.

**Redesign goals:**
- This is the most data-rich page. Use a **two-column layout**: left column for the recommendation cards (the "story"), right column for metadata/linked entities
- Recommendation cards are the hero element. Each card should have:
  - A clear visual hierarchy: selection + market as the headline, probability/edge/odds in a compact grid below
  - A color-coded left stripe (green = positive edge, amber = marginal, red = negative)
  - Provenance tooltips as subtle info icons, not inline badges cluttering the numbers
  - The "computed" badge stays but gets refined (maybe a small ◆ diamond icon)
- Evidence coverage: radial or arc chart showing applied vs shadow vs total
- Warning chips: keep the severity color-coding but refine the shape (rounded pill, icon prefix)
- Raw JSON sections: dark code block with syntax highlighting (Prism-style), collapsible with smooth accordion animation
- Linked bets/outcomes: card-style links with status badges, not a bulleted list

### 4.4 Bet Ledger (`bets.html` / `bet_detail.html`)

**Redesign goals:**
- Same table treatment as traces (sticky columns, pills, hover)
- PnL column: green/red background tint on the cell, not just text color
- Settlement status as a badge (won/lost/pending/void) with distinct icon per status
- Bet detail page: prominent PnL display (large number, colored), with the linked trace recommendation shown as an inline preview card

### 4.5 Sessions (`sessions.html` / `session_detail.html`)

**Redesign goals:**
- Session list: compact cards instead of table rows. Each card shows session_id, date, event count, trace count, QA gate (pass/fail badge)
- Session detail: the split-panel (sidecar narrative vs DB traces) is a good pattern. Refine it:
  - Left panel (sidecar): styled like a terminal/log output with timestamps and indentation
  - Right panel (DB): structured cards/tables
  - Add a subtle vertical divider with a draggable resize handle

### 4.6 Diagnostics (`diagnostics.html`)

Current: tiles + KV tables.

**Redesign goals:**
- This page should feel like a **status board**. Large, bold stat tiles with subtle pulse animation on the "ok" status indicator
- Calibration registry summary: horizontal bar chart showing profile status distribution (production/candidate/archived/rejected)
- Signal scoring: sparkline or mini chart of last N runs if data is available; otherwise a timestamp + row count in a clean card

### 4.7 Calibration (`calibration.html`)

**Redesign goals:**
- Profile rows as expandable cards, not table rows
- Active production profile gets a prominent highlight (glow border or accent background)
- Status badges: refined color-coded pills (production=green, candidate=blue, archived=grey, rejected=red)
- Held-out metrics: small inline charts or gauge indicators (brier score, calibration gap)

### 4.8 Signal Performance (`signals.html`)

**Redesign goals:**
- Data table with sortable columns
- Direction accuracy and hit rate: small inline progress bars in the table cells
- Brier score / cal gap: color gradient (green-to-red) based on value quality
- Per-league grouping with collapsible sections

### 4.9 Review Queue (`review.html`)

**Redesign goals:**
- Kanban-style columns or severity-stacked cards (like Linear's triage view)
- Each bucket (ungraded traces, pending bets, problem sessions) as a distinct column
- Items within each bucket as compact cards with one-line summaries
- Count badges on each column header

### 4.10 CLV / Market Movement (`clv.html`)

**Redesign goals:**
- Per-bet CLV should be visualized as a deviation chart (horizontal bars showing beat/miss relative to closing line)
- Aggregate beat-the-close percentage as a large stat tile
- Scatter plot: opening line taken vs closing line (each dot is a bet) with a 45-degree reference line

---

## 5. Global Component Library

These components appear across multiple pages and should be designed as a reusable system:

| Component | Used on | Notes |
|---|---|---|
| **Top nav bar** | All pages | Sticky, dark panel, brand + nav links + READ-ONLY badge |
| **Filter bar** | Traces, Bets, CLV | Collapsible, inline pill inputs, submit + clear |
| **Filter chips** | Traces, Bets, CLV | Active filter pills with × remove button |
| **Data table** | Traces, Bets, Signals, CLV | Sticky header, sortable, row hover, alternating tints |
| **KV table** | Detail pages, Diagnostics | Two-column key-value display |
| **Stat tiles** | Index, Diagnostics | Large number + small label, optional status color |
| **Cards** | Index, Sessions, Review | Clickable containers with hover-lift |
| **Recommendation card** | Trace detail | Selection/market headline, probability grid, provenance |
| **Status badges** | Calibration, Bets, Sessions | Colored pills (production/candidate/won/lost/pass/fail) |
| **Provenance badge** | Detail pages | Source label (db_trace_payload, bet_ledger, sidecar_process) |
| **Warning chips** | Trace detail, Diagnostics | Severity-colored (info/warn/fail) with icon prefix |
| **Pagination** | List pages | Page N/M, prev/next, optional page-size selector |
| **Breadcrumbs** | Detail pages | ‹ Back to List |
| **Code block** | Detail pages | Dark bg, mono font, syntax highlighting, collapsible |
| **Split panel** | Session detail | Side-by-side with draggable divider |

---

## 6. Typography

| Role | Font | Weight | Size |
|---|---|---|---|
| Body / labels | Inter (or Geist Sans) | 400/500 | 14px |
| Headings | Inter | 600/700 | 18-24px |
| Nav items | Inter | 500 | 13px |
| Data / numbers | JetBrains Mono (or Berkeley Mono) | 400/500 | 13px |
| Badges / pills | Inter | 700 | 10-11px, uppercase, tracked |
| Code blocks | JetBrains Mono | 400 | 12px |

---

## 7. Color System (starting point)

Keep the current dark palette but refine for more depth:

```
Background layers:
  base:       #0a0c10   (deepest)
  surface-1:  #12151b   (main panels)
  surface-2:  #1a1e27   (elevated cards)
  surface-3:  #222733   (popovers, modals)

Text:
  primary:    #eaedf2
  secondary:  #8891a0
  tertiary:   #5a6270

Borders:
  subtle:     #1e2330
  default:    #2a3040
  strong:     #3a4255

Accent:
  blue:       #4da3ff   (links, active)
  blue-muted: #1a3050   (blue bg tint)

Status:
  positive:   #34d399   (green; wins, pass, production)
  pos-bg:     #0d2818   (green background tint)
  negative:   #f87171   (red; losses, fail, rejected)
  neg-bg:     #2a1215   (red background tint)
  warning:    #fbbf24   (amber; pending, caution)
  warn-bg:    #2a2210   (amber background tint)
```

---

## 8. Interaction & Motion

- **Page loads:** content fades in with a 200ms ease-out, staggered by section (hero first, then cards, then tables)
- **Hover states:** cards lift 2px with a subtle shadow expansion (150ms ease). Table rows get a left-border accent slide-in.
- **Filters:** applying a filter adds a chip with a scale-in animation; removing slides it out. Table content cross-fades.
- **Expanding sections:** accordion expand with height animation + opacity fade (250ms ease-in-out)
- **Navigation:** page transitions use a quick fade or slide (if SPA) or the browser default (if staying server-rendered)
- **Loading states:** skeleton screens or subtle shimmer placeholders, not spinners

---

## 9. Responsive Behavior

- **Primary target:** desktop, 1280px+ (this is a pro operator tool, not a mobile app)
- **Breakpoints:** 1440px (full), 1024px (compact sidebar or stacked), 768px (emergency tablet, stack everything)
- Tables should horizontally scroll with sticky first columns below 1024px
- The split-panel on session detail should stack vertically below 900px (already does in current CSS)

---

## 10. Constraints

- **Read-only.** The console never mutates data. No forms that submit writes. The READ-ONLY badge is a trust signal; keep it visible.
- **Server-rendered.** We're currently Jinja2/HTML. The redesign can propose moving to a React SPA if justified, but the backend API already exists (`/api/*` routes) so either approach works.
- **No external auth flow in the UI.** Auth is bearer-token via header, handled outside the console.
- **Provenance must remain visible.** Every data value traces back to a source. This is non-negotiable for trust.

---

## 11. Deliverables Requested

1. **Component library** in Figma: all components from Section 5, with variants for each state
2. **Page designs** for all 12 pages (landing, 4 list views, 4 detail views, diagnostics, calibration, signals)
3. **Interactive prototype** showing navigation flow, filter interactions, and accordion/expand patterns
4. **Design tokens** exported as CSS custom properties (or a JSON token file)
5. **Motion spec** for the key animations (hover, expand, page transition, loading)

---

## 12. Inspiration / Reference

- **Linear** (linear.app): clean data tables, subtle depth, smooth transitions
- **Vercel Dashboard** (vercel.com/dashboard): minimalist dark theme, stat tiles, deployment status
- **Bloomberg Terminal**: information density, color-coded status, operator-tool feel
- **Raycast**: glass-panel depth, smooth micro-interactions
- **Grafana**: dashboard tiles, time-series display, dark theme data viz
- **Stripe Dashboard**: clean typography, status badges, card layouts

---

## Attached Mockup

See the companion HTML mockup file (`operator_console_mockup.html`) for a visual target of the landing page and key components. This is not pixel-perfect; it demonstrates the aesthetic direction (glass panels, depth, refined typography, smooth hover states, stat previews in cards).
