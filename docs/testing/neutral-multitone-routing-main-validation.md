# Neutral multi-tone routing — main-based validation

## Scope

This change is based directly on `main` commit `249872c3a590cd09aa6c92b69c3dd3c7dce9119e`.

The existing analyzer implementation is preserved byte-for-byte by reusing its exact blob (`70fb527996dc5e910d3dda48ad763ce9d469cc35`) at `engine/app/analyzer_base.py`. The compatibility wrapper changes only automatic routing when a substantial neutral mid-tone survives the existing anti-alias film removal and the original recommendation is `single_color` or `lineart`.

## Required proof

The PR is not eligible for merge unless all of the following are true:

- neutral multi-tone contract tests pass;
- true binary silhouette keeps binary routing;
- anti-alias-only gray film does not trigger the guard;
- output-quality contract completes;
- 7 diagnostic cases run twice through the production pipeline;
- all 7 selected SVG artifacts are byte-deterministic;
- structural failures remain zero;
- unrelated release contracts stay green.

## Safety boundaries

- no quality threshold is lowered;
- no evaluator verdict is waived;
- no user-selected mode is changed;
- no RFV corpus or release decision is changed;
- merge remains blocked while any required CI check is failing or incomplete.
