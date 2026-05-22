# Stockfish distillation via PySR

Design doc: `docs/benchmark_stockfish_distillation.md`

**Mode:** STOCKFISH depth=15

**Corpus:** 500 positions, eval range [-1000, 1000] cp; mean=-23.4, std=523.0

**Split:** random 80/19

## Baselines (TEST)

| Model | R2 | MSE | Kendall-τ | Expression |
|---|---|---|---|---|
| constant | -0.0016 | 262558.5 | 0.0000 | `-19.26` |
| material_only_LR | 0.3791 | 162754.5 | 0.4695 | `69.1132 * material_net + -18.05` |
| full_LR | 0.2876 | 186751.5 | 0.3750 | `+0.797*WP_count +11.088*WN_count +13.627*WB_count -47.369*WR_count -28.596*WQ_count +75.731*BP_count -17.779*BN_count -9` |

## PySR Pareto front (TRAIN/TEST)

| Complexity | TRAIN loss | TEST R2 | TEST τ | Equation |
|---|---|---|---|---|
| 1 | 2.724e+05 | 0.0075 | 0.4695 | `material_net` |
| 3 | 1.535e+05 | 0.3791 | 0.4695 | `material_net * 69.15066` |
| 5 | 1.526e+05 | 0.3775 | 0.4669 | `(material_net - B_king_zone_enemy) / 0.0146158505` |
| 7 | 1.518e+05 | 0.3744 | 0.4652 | `(material_net / 0.014378776) - (central_BP * W_mobility)` |
| 9 | 1.515e+05 | 0.3665 | 0.4572 | `(material_net / 0.014378776) - ((WN_count * central_BP) * W_mobility)` |
| 10 | 1.510e+05 | 0.3863 | 0.4728 | `(material_net / 0.014500216) - (central_BP / tanh(1.0133632 - B_king_zone_own_pawns))` |
| 11 | 1.489e+05 | 0.3842 | 0.4729 | `(material_net / 0.014341535) - ((central_BP * W_mobility) / (1.3503796 - B_king_zone_own_pawns))` |
| 12 | 1.487e+05 | 0.3824 | 0.4684 | `(material_net / 0.014331786) - (W_mobility * (central_BP / tanh(1.3503796 - B_king_zone_own_pawns)))` |
| 13 | 1.482e+05 | 0.3910 | 0.4741 | `(material_net / 0.01460058) - ((W_mobility * central_BP) / ((1.3168311 - B_king_zone_own_pawns) - central_BB))` |
| 14 | 1.481e+05 | 0.3896 | 0.4741 | `(material_net / 0.01460058) - (W_mobility * (central_BP / tanh((1.3168311 - B_king_zone_own_pawns) - central_BB)))` |
| 15 | 1.475e+05 | 0.3600 | 0.4232 | `(material_net / 0.014558159) - ((BB_count * W_mobility) - (WP_count * ((WP_count * B_king_zone_own_pawns) * central_WP)))` |
| 17 | 1.475e+05 | 0.3600 | 0.4387 | `(material_net / 0.014558159) - (WN_count * (W_mobility - ((WP_count * ((BR_count * B_king_zone_own_pawns) * B_king_zone_own_pawns)) * central_WP)))` |
| 18 | 1.468e+05 | 0.3450 | 0.4149 | `(material_net / 0.0152835995) - ((W_mobility - ((WP_count * central_WP) * abs((B_king_zone_own_pawns * 4.5386086) + material_net))) * BR_count)` |
| 20 | 1.450e+05 | 0.3330 | 0.4266 | `(material_net / 0.014558157) - ((W_mobility - abs(central_WP * ((material_net + (BR_count * (B_king_zone_own_pawns * B_king_zone_own_pawns))) * WP_cou` |
| 22 | 1.448e+05 | 0.3351 | 0.4262 | `(material_net / 0.014558157) - ((W_mobility - (central_WP * abs(((0.84703195 * material_net) + ((BR_count * B_king_zone_own_pawns) * B_king_zone_own_p` |
| 23 | 1.442e+05 | 0.3345 | 0.4313 | `(material_net / 0.014057474) - ((W_mobility - abs(central_WP * ((material_net + (BR_count * B_king_zone_own_pawns)) * ((B_king_zone_own_pawns * WP_cou` |
| 24 | 1.439e+05 | 0.3574 | 0.4222 | `(material_net / 0.014974028) - ((W_mobility * BB_count) - abs((central_WP * (3.997501 - (WP_count * B_king_zone_own_pawns))) * (BR_count * ((WN_count ` |
| 26 | 1.431e+05 | 0.3587 | 0.4254 | `(material_net / 0.014714847) - ((BB_count * W_mobility) - abs((central_WP * (BR_count * (material_net + (WN_count * 3.3158555)))) * ((WR_count * 2.552` |
| 28 | 1.423e+05 | 0.3530 | 0.4246 | `(material_net / 0.014804782) - ((W_mobility * BB_count) - abs((WN_count * central_WP) * ((((B_king_zone_own_pawns + BB_count) * WR_count) - (WP_count ` |
| 30 | 1.423e+05 | 0.3553 | 0.4234 | `(material_net / 0.014972368) - ((BB_count * W_mobility) - abs(WN_count * (central_WP * (((WP_count * B_king_zone_own_pawns) - (WR_count * (B_king_zone` |
| 32 | 1.413e+05 | 0.3423 | 0.4201 | `(material_net / 0.015097968) - ((BB_count * W_mobility) - abs(((((phase - -1.249843) + (((BB_count * WR_count) - WP_count) * B_king_zone_own_pawns)) *` |

**Best PySR by TEST R2:** cx=13, R2=0.3910

## Acceptance (per design doc)

- (1) compact (cx<30) eq with TEST R2>0.85 beating full-LR by >0.03: FAIL
- (2) very compact (cx≤15) eq with TEST R2>0.75: FAIL

_Elapsed: 172.8s_
