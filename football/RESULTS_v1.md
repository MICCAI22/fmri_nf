# v1 results (run 2026-11-20)

Data: football-data.co.uk, E0+E1, seasons 1718-2627 (2627 up to 2026-11-08).
Test: walk-forward, 2223-2627, 4,008 matches.

| model | logloss | brier | acc |
|---|---|---|---|
| A (our features) | 1.0312 | 0.6193 | 48.1% |
| B (features + opening odds) | 1.0110 | 0.6055 | 50.0% |
| B (log-odds features) | 1.0102 | - | - |
| Market opening | 1.0083 | 0.6040 | 49.9% |
| Market closing | 1.0054 | 0.6019 | 50.1% |

Gap to closing (bootstrap 95% CI): A +0.026 [0.020, 0.032], B +0.0057 [0.0026, 0.0086].
Both are significantly worse than the closing line. B is also worse than the opening odds it receives as input.

Caveats:
- PSCH (Pinnacle closing) is missing in ~48% of 2526 and 100% of 2627, so the benchmark falls back to AvgC.
  The benchmark is inconsistent across seasons.
- Original ROI is against B365 (6% overround), so every model loses (-5% to -13%).
- At Max odds, B(log-odds) shows +2% to +3% ROI, but it is unstable by season (-1% to +12%),
  and Max odds are not realistically available. Not evidence of an edge.
- Model A calibration is reasonable. Its "edges" come from being less sharp than the market, not from overconfidence.
