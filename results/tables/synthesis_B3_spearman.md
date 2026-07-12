# B3: pooled Spearman SI-SDR vs analyst metrics (drums)

|    | analyst_metric   |   spearman_vs_SI_SDR |   p_value |   n | expected_if_SDR_tracked_it          |
|---:|:-----------------|---------------------:|----------:|----:|:------------------------------------|
|  0 | onset_f_measure  |                0.618 |    0      | 200 | higher SDR -> higher onset-F        |
|  1 | |d_attack_slope| |               -0.169 |    0.0165 | 200 | higher SDR -> smaller |Δ attack|    |
|  2 | |d_variation|    |               -0.156 |    0.0277 | 200 | higher SDR -> smaller |Δ variation| |
|  3 | |d_crest|        |               -0.292 |    0      | 200 | higher SDR -> smaller |Δ crest|     |
