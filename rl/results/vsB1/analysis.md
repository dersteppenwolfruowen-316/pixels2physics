
## all (n=108)

| arm | greedy_sim_success | bo4_sim_success | pass@1 | pass@4 | greedy_iou_heldout | greedy_wmae_heldout | greedy_iou_heldout_ok | greedy_wmae_heldout_ok | greedy_reward |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.269 | 0.565 | 0.227 | 0.565 | 0.084 | 0.911 | 0.314 | 0.668 | 0.221 |
| B1+DPO | 0.120 | 0.315 | 0.104 | 0.315 | 0.026 | 0.980 | 0.218 | 0.834 | 0.131 |
| B1+DPO-B1 | -0.148 [-0.265, -0.023] | -0.250 [-0.393, -0.108] | -0.123 [-0.204, -0.046] | -0.250 [-0.393, -0.108] | -0.058 [-0.107, -0.007] | +0.069 [+0.027, +0.105] | -0.656 [-0.656, -0.656] | +0.614 [+0.614, +0.614] | -0.090 [-0.144, -0.030] |

McNemar greedy runs B1+DPO-B1: arm-only 12, base-only 28, p=0.0166

## old16 (n=16)

| arm | greedy_sim_success | bo4_sim_success | pass@1 | pass@4 | greedy_iou_heldout | greedy_wmae_heldout | greedy_iou_heldout_ok | greedy_wmae_heldout_ok | greedy_reward |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.250 | 0.625 | 0.234 | 0.625 | 0.116 | 0.952 | 0.464 | 0.809 | 0.215 |
| B1+DPO | 0.188 | 0.375 | 0.125 | 0.375 | 0.030 | 0.970 | 0.160 | 0.840 | 0.150 |
| B1+DPO-B1 | -0.062 [-0.375, +0.250] | -0.250 [-0.562, +0.062] | -0.109 [-0.234, +0.016] | -0.250 [-0.562, +0.062] | -0.086 [-0.248, +0.047] | +0.018 [-0.080, +0.127] | – | – | -0.065 [-0.216, +0.075] |

McNemar greedy runs B1+DPO-B1: arm-only 3, base-only 4, p=1
