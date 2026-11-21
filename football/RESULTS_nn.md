# Neural network (nn_model.py) - walk-forward round by round

Test: 22/23 to 26/27 (through 08/11/2026), E0+E1, 4,008 matches, 260 rounds.
Each round: predict -> reveal results -> update the network (warm start, replay) -> next round.
Full refit at the start of each season. Hyper-parameters chosen on 20/21-21/22 only.
Ensemble of 3 seeds. NN-A = statistics only, NN-B = + opening odds (residual on market).

## 1X2 (log loss)
| model | logloss |
|---|---|
| market close | 1.0054 |
| market open | 1.0083 |
| v1 logistic B | 1.0110 |
| **NN-B** | 1.0123 |
| v1 logistic A | 1.0312 |
| **NN-A** | 1.0322 |
| naive Poisson | 1.0579 |

## Other markets (NN-B vs naive rolling-average Poisson, bootstrap 95% CI)
| market | NN-B minus reference | CI |
|---|---|---|
| over 2.5 vs market close | +0.0076 (worse) | [+0.0045, +0.0108] |
| over 2.5 vs naive | -0.0014 | [-0.0032, +0.0003] n.s. |
| BTTS vs naive | +0.0032 (worse) | [+0.0006, +0.0060] |
| corners over 9.5 vs naive | -0.0018 | [-0.0038, +0.0002] n.s. |
| cards over 3.5 vs naive | -0.0025 (better) | [-0.0040, -0.0008] |

Correlation of predicted vs actual totals: goals 0.16, corners 0.10, cards 0.15.
Value check at opening average odds (edge > 5%): 1X2 -11% to -15%, over 2.5 -7% to +1% (147 bets, noise).

Conclusion: the network does not beat the logistic regression on 1X2, does not beat
the market on any market that has odds, and adds little over simple averages on
corners/cards. Corners and cards are mostly noise at the match level with these features.
