# A2 paired test: transformer vs CNN/hybrid drift

|    | descriptor        | bs_roformer_vs   |   n_bs_greater |   of |   median_ratio |   wilcoxon_p |
|---:|:------------------|:-----------------|---------------:|-----:|---------------:|-------------:|
|  0 | dyn_attack_slope  | scnet            |             13 |   20 |           1.62 |       0.0441 |
|  1 | dyn_attack_slope  | htdemucs         |              9 |   20 |           0.88 |       0.5459 |
|  2 | dyn_impulsiveness | scnet            |             11 |   20 |           1.44 |       0.6742 |
|  3 | dyn_impulsiveness | htdemucs         |             12 |   20 |           1.27 |       0.5706 |
|  4 | dyn_variation     | scnet            |             10 |   20 |           0.93 |       0.8408 |
|  5 | dyn_variation     | htdemucs         |              8 |   20 |           0.52 |       0.2455 |
|  6 | dyn_crest         | scnet            |             10 |   20 |           0.97 |       0.8408 |
|  7 | dyn_crest         | htdemucs         |              8 |   20 |           0.69 |       0.2774 |
|  8 | onset_f_measure   | scnet            |              8 |   20 |           0.8  |       0.2598 |
|  9 | onset_f_measure   | htdemucs         |             12 |   20 |           1.01 |       0.6477 |
