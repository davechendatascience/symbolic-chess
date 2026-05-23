# Self-play iter-1 distillation

Design: Architecture A from `docs/math_engine_cpp_v0.md`. math-engine-cpp self-play games + Stockfish ground-truth labels, joint regression target z(decay_outcome) + z(sf_cp).

**Self-play corpus:** data/chess/cpp_self_play_iter1.parquet (500 games at depth 5)

**Stockfish labels:** depth=15 on subsample of 5000 unique non-opening positions

**Split:** random 80/19

## Baselines on joint target (TEST)

| Model | R2 (joint) | Kendall-τ | Expression |
|---|---|---|---|
| constant | -0.0000 | 0.0000 | `-0.000` |
| material_only_LR | 0.3593 | 0.4988 | `0.1843 * material_net + 0.098` |
| full_LR | 0.4702 | 0.5204 | `+0.060*WP_count -0.013*WN_count +0.164*WB_count +0.022*WR_count -0.104*WQ_count -0.018*BP_count -0.024*BN_count -0.061*BB_count +0.030*BR_co` |

## PySR Pareto (joint target trained; evaluated on each component)

| Cx | TRAIN loss | R2 joint | R2 outcome | R2 sf_cp | τ outcome | τ sf_cp | Equation |
|---|---|---|---|---|---|---|---|
| 1 | 7.505e-01 | -0.0000 | -0.0001 | -0.0000 | 0.0000 | 0.0000 | `2.475345e-6` |
| 2 | 6.710e-01 | 0.0983 | -0.2066 | 0.3499 | 0.1906 | 0.5477 | `tanh(material_net)` |
| 3 | 4.417e-01 | 0.3505 | 0.0937 | 0.4353 | 0.1906 | 0.5477 | `material_net * 0.1788507` |
| 5 | 4.323e-01 | 0.3593 | 0.0944 | 0.4478 | 0.1906 | 0.5477 | `(material_net - -0.53306013) * 0.18427286` |
| 6 | 4.271e-01 | 0.3582 | 0.0931 | 0.4474 | 0.1778 | 0.5189 | `(material_net + tanh(WB_count)) * 0.17921849` |
| 7 | 4.115e-01 | 0.3773 | 0.1017 | 0.4677 | 0.2072 | 0.5229 | `((WB_count + material_net) - central_BP) * 0.17531382` |
| 9 | 4.079e-01 | 0.3908 | 0.1139 | 0.4761 | 0.2111 | 0.5295 | `(material_net + ((WB_count - central_BP) - central_BN)) * 0.17522278` |
| 11 | 4.037e-01 | 0.3742 | 0.1014 | 0.4634 | 0.2148 | 0.5200 | `((WB_count + (material_net - central_BP)) - (B_passed_pawns * WQ_count)) * 0.17544349` |
| 13 | 4.006e-01 | 0.3871 | 0.1133 | 0.4711 | 0.2190 | 0.5260 | `((material_net + ((WB_count - central_BP) - (WQ_count * B_passed_pawns))) - central_BN) * 0.17509298` |
| 15 | 3.977e-01 | 0.3797 | 0.1046 | 0.4685 | 0.2124 | 0.5082 | `((WB_count + (material_net - central_BP)) - min(B_passed_pawns, WP_count * (WQ_count - 0.13210723))) * 0.17799103` |
| 16 | 3.971e-01 | 0.3766 | 0.1049 | 0.4635 | 0.2232 | 0.5208 | `((material_net + (WB_count - central_BP)) - min((WQ_count * WP_count) - sign(WR_count), B_passed_pawns)) * 0.17923877` |
| 17 | 3.937e-01 | 0.3932 | 0.1145 | 0.4791 | 0.2185 | 0.5191 | `((((material_net + WB_count) - central_BP) - min((WQ_count * WP_count) - 0.64000183, B_passed_pawns)) - central_BN) * 0.18130049` |
| 18 | 3.925e-01 | 0.3906 | 0.1154 | 0.4744 | 0.2278 | 0.5261 | `(material_net + (((WB_count - central_BP) - min((WQ_count * WP_count) - sign(WR_count), B_passed_pawns)) - central_BN)) * 0.18130049` |
| 19 | 3.892e-01 | 0.4027 | 0.1221 | 0.4860 | 0.2135 | 0.5227 | `(2.1272836 - max(1.9303244, WB_count)) * (((WB_count - min(B_passed_pawns, (WQ_count * WP_count) - 0.7443978)) + material_net) - central_BP)` |
| 20 | 3.882e-01 | 0.4005 | 0.1253 | 0.4795 | 0.2170 | 0.5220 | `((WB_count - min(B_passed_pawns, (WQ_count * WP_count) - sign(WR_count))) + (material_net - central_BP)) * (2.1256528 - max(WB_count, 1.9312488))` |
| 21 | 3.880e-01 | 0.4017 | 0.1247 | 0.4819 | 0.2119 | 0.5207 | `(2.1274502 - max(WB_count, 1.9337238)) * (((WB_count - min((WQ_count * WP_count) - tanh(sign(WR_count)), B_passed_pawns)) + material_net) - central_BP)` |
| 22 | 3.879e-01 | 0.4015 | 0.1248 | 0.4815 | 0.2170 | 0.5220 | `max(((WB_count - min(B_passed_pawns, (WQ_count * WP_count) - sign(WR_count))) + (material_net - central_BP)) * (2.1256528 - max(WB_count, 1.9312488)), -3.362444` |
| 23 | 3.846e-01 | 0.4134 | 0.1358 | 0.4886 | 0.2167 | 0.5243 | `((((material_net - central_BP) + WB_count) - min((WP_count * WQ_count) - 0.86094135, B_passed_pawns)) - central_BN) * min(max(1.1815635 - WB_count, 0.12860867),` |
| 25 | 3.805e-01 | 0.4205 | 0.1475 | 0.4878 | 0.2255 | 0.5169 | `max(min(0.20079236, 1.1775645 - WB_count), 0.11920246) * ((WB_count + ((material_net - central_BP) - min(((WP_count * WQ_count) * B_king_zone_own_pawns) - 1.000` |
| 27 | 3.781e-01 | 0.4195 | 0.1480 | 0.4859 | 0.2223 | 0.5166 | `min(0.20079236, max(1.1775645 - WB_count, 0.11920246)) * ((((WB_count + material_net) - central_BP) - min(((WP_count * WQ_count) * B_king_zone_own_pawns) - min(` |
| 29 | 3.776e-01 | 0.4189 | 0.1439 | 0.4890 | 0.2189 | 0.5278 | `min(0.20136636, max(1.1808852 - WB_count, 0.12665501)) * (((WB_count + (material_net - central_BP)) - min(((WP_count * WQ_count) * B_king_zone_own_pawns) - min(` |
| 31 | 3.768e-01 | 0.4193 | 0.1446 | 0.4889 | 0.2289 | 0.5184 | `max(min(0.19755289, 1.175862 - WB_count), 0.124054536) * (((material_net - central_BP) + (WB_count - min(B_passed_pawns, (((WP_count + -1.1260582) * WQ_count) *` |
| 33 | 3.746e-01 | 0.4248 | 0.1488 | 0.4931 | 0.2239 | 0.5305 | `min(0.20079598, max(1.177605 - WB_count, 0.11921919)) * (((WB_count - min(((max(B_passed_pawns, central_BP) * B_king_zone_own_pawns) * ((WP_count * WQ_count) - ` |

**Picked:** cx=9 (joint R2=0.3908, outcome R2=0.1139, sf_cp R2=0.4761)

`(material_net + ((WB_count - central_BP) - central_BN)) * 0.17522278`

_Elapsed: 888.3s_
