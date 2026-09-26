---
name: DataForge
description: A local AI data scientist, set like a Basel science poster.
colors:
  pine-field: "#0f6b4f"
  pine-field-dark: "#14805e"
  pine-text-dark: "#5cc79d"
  field-ink: "#ffffff"
  field-ink-soft: "#cfe3da"
  field-ink-soft-dark: "#eef7f3"
  paper: "#f4f4f1"
  paper-recess: "#eaeae5"
  ink: "#181818"
  ink-secondary: "#4f4f4a"
  ink-tertiary: "#6a6a64"
  paper-dark: "#161618"
  paper-recess-dark: "#1f1f22"
  ink-dark: "#ecebe6"
  ink-secondary-dark: "#b6b5ae"
  ink-tertiary-dark: "#93928c"
  signal-red: "#b4261a"
  signal-red-dark: "#ff7b6b"
typography:
  wordmark:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "34px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.035em"
    fontVariation: "'wdth' 112"
  display:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "clamp(34px, 4.1vw, 54px)"
    fontWeight: 700
    lineHeight: 1.04
    letterSpacing: "-0.035em"
  figure:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "46px"
    fontWeight: 300
    lineHeight: 1
    letterSpacing: "-0.03em"
    fontVariation: "'wdth' 80"
    fontFeature: "'tnum'"
  title:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 500
    lineHeight: 1.45
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.62
  label:
    fontFamily: "Archivo, Segoe UI, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.4
    fontVariation: "'wdth' 88"
  code:
    fontFamily: "Cascadia Mono, Consolas, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  none: "0px"
spacing:
  xs: "6px"
  sm: "12px"
  md: "20px"
  lg: "28px"
  xl: "36px"
  gutter: "clamp(24px, 5vw, 72px)"
  label-col: "104px"
  measure: "880px"
components:
  button-primary:
    backgroundColor: "{colors.pine-field}"
    textColor: "{colors.field-ink}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "38px"
  button-primary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "38px"
  chat-item-active:
    backgroundColor: "{colors.field-ink}"
    textColor: "{colors.pine-field}"
    rounded: "{rounded.none}"
    padding: "7px 12px"
  mode-tab-selected:
    backgroundColor: "{colors.pine-field}"
    textColor: "{colors.field-ink}"
    rounded: "{rounded.none}"
    height: "30px"
  composer:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
---

# Design System: DataForge

## Overview

**Creative North Star: "The Swiss Science Poster"**

DataForge borrows the discipline of 1950s-60s Basel pharmaceutical design (Geigy's International Typographic Style): one grotesk, a strict asymmetric grid, and a single saturated colour field. The app reads like a science poster where the data is the image. The pine green field owns the sidebar and nothing else competes with it; the reading column stays achromatic so matplotlib charts carry the colour.

Density is that of a working instrument: calm over hours, legible in seconds for someone judging a demo. Hierarchy comes from weight, width (Archivo's `wdth` axis) and placement on the grid, never from tinted boxes or shadows. Light and dark themes follow the operating system and share every rule.

It deliberately rejects the dark-navy chat clone with a purple accent, identical rounded cards, tracked all-caps labels, bouncing-dot loaders and avatar circles.

**Key Characteristics:**
- One saturated field (pine green), used for the sidebar, primary actions and selected states only.
- One typeface (Archivo, self-hosted), ranked by weight and width.
- Square corners and 1px rules everywhere.
- A hanging label column: role labels sit to the left of a reading column that everything else aligns to.
- Data figures set large, light and condensed, always with context beneath.

## Colors

A near-neutral paper and ink with one committed pine green field (chosen by the user over the original ultramarine).

- **Pine Field** (`#0f6b4f`, dark theme `#14805e`): the full-height sidebar, primary buttons and the selected mode tab. Where green is used as text or a mark on the page (assistant label, focus rings, hover titles), dark mode switches to the lighter **Pine Text** `#5cc79d` so it stays readable.
- **Paper / Ink** (`#f4f4f1` / `#181818`; dark `#161618` / `#ecebe6`): ground and text. **Paper Recess** (`#eaeae5` / `#1f1f22`) holds code blocks and hover fills.
- **Secondary and tertiary ink**: descriptions, captions, context lines; both meet 4.5:1 on paper.
- **Signal Red** (`#b4261a` / `#ff7b6b`): errors and the agent's failed-step mark only.

**The One Field Rule.** Pine green is the only chromatic colour the interface draws. Charts may be any colour; chrome may not.

**The Inversion Rule.** Selection inverts rather than tints: the active chat turns white on the field, the selected mode fills with the field, and text selection uses the field.

## Typography

**Archivo** (SIL OFL, bundled in `frontend/fonts/`, variable weight 100-900, width 62-125%) carries everything; **Cascadia Mono** (system) is used only for code, output and formulas.

- Wordmark 34/800 at 112% width; display 34-54px/700; profile figures 46px/300 at 80% width with tabular numerals; user questions 18/500; body 15/400 at 1.62; labels 12/600 at 88% width in sentence case.

**The Width Rule.** Rank with width and weight before size. Labels go narrower, the wordmark goes wider, figures go condensed and light.

**The Sentence Case Rule.** No tracked all-caps labels anywhere.

## Layout

A sidebar field (272px, 232px at 1040px and below) beside a main column. Main content uses a two-column grid: a label column (104px, 76px at 1040px and below) and a reading column capped at 880px, separated by a 28px gap. The empty state, every turn, and the composer align to the reading column's left edge. Gutters are `clamp(24px, 5vw, 72px)`. At windows 720px tall or shorter, the empty state compacts; at 760px wide and below the label column stacks above the content. The Electron window's minimum size is 900x600.

## Elevation & Depth

Flat. Depth comes from 1px rules (ink at 14% for dividers, full ink for structural edges such as the composer, section tops and table headers). The only overlay, the new-project dialog, sits on a scrim with a 1px ink edge and no shadow.

## Shapes

Every corner is square (radius 0): buttons, inputs, chips, code blocks, charts and the composer. Charts sit in a 1px rule frame on white.

## Components

- **Primary button**: field fill, white text, 38px, turns ink-on-paper on hover, nudges 1px down on press.
- **Outline button**: 1px ink edge, inverts to ink on hover.
- **Composer**: a 1px ink-ruled bar; the focus state swaps the edge to the field. Holds tool buttons (Attach, Workspace), a segmented mode control and a square send button.
- **Mode tabs**: a segmented radio group with a 1px rule; the selected tab fills with the field.
- **Profile readings**: four figures (rows, columns, missing values, duplicate rows), each with a context line such as "2.1% of all cells". Shown as a 4-up row, or 2x2 at 1040px and below.
- **Agent steps**: ruled rows with a mark and state (spinning ring for running, check for done, warning for a failed step being revised), with collapsible code and error.
- **Thinking sign**: a 3x3 grid of cells lighting in a diagonal wave, the current activity ("Planning", "Revising the code", "Thinking about the next step"), and a live "for 1 min 12s" counter. After 20s, 60s and 3 min a line underneath explains that local models are slower. It also shows between agent steps, so the app never looks idle while the model works.
- **Sidebar navigation**: a Projects link (inverts when active) above a "Recent" list of chats from every project, newest first, each with its project name underneath.
- **Projects pages**: a ruled table of projects (name, description, chat count, last activity) and a page per project with Edit details, Delete (confirmed inline, never with a dialog), its chats, and a composer that starts a new chat inside the project. Chats show a "Project / chat title" breadcrumb.
- **Tables**: an ink rule under a sticky header, faint row rules, numeric cells right-aligned.

## Do's and Don'ts

- Do keep the reading column achromatic and let charts carry colour.
- Do give every number context (share of rows, share of cells, type split).
- Do use Phosphor icons from the inline sprite, one stroke family.
- Don't add rounded corners, drop shadows, gradients or glass.
- Don't introduce a second accent colour, or use the pine field as decoration.
- Don't use all-caps tracked labels, eyebrows, section numbers or em dashes in UI copy.
- Don't use emoji or Unicode glyphs as icons.
