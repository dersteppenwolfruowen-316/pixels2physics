# MPMWorlds 数据分析报告

## 自动发现的问题

1. 完整性问题 {'missing_video': 6}（yaml 解析失败 / 缺程序 / 缺视频的场景需要先清理）
2. schema 不统一：66 个字段只在部分场景出现（见第 2 节），训练目标需要按材料/物体类型条件化
3. [elastic/rho] 取值离散：n_unique=17，最常见值占 71%（更像分类而非回归，损失与评估要按离散值设计）
4. [elastic/E] 取值离散：n_unique=12，最常见值占 64%（更像分类而非回归，损失与评估要按离散值设计）
5. [elastic/nu] 取值离散：n_unique=3，最常见值占 96%（更像分类而非回归，损失与评估要按离散值设计）
6. [elastoplastic/rho] 取值离散：n_unique=2，最常见值占 98%（更像分类而非回归，损失与评估要按离散值设计）
7. [elastoplastic/E] 取值离散：n_unique=3，最常见值占 90%（更像分类而非回归，损失与评估要按离散值设计）
8. [elastoplastic/nu] 取值离散：n_unique=1，最常见值占 100%（更像分类而非回归，损失与评估要按离散值设计）
9. [fem_elastic/rho] 取值离散：n_unique=1，最常见值占 100%（更像分类而非回归，损失与评估要按离散值设计）
10. [fem_elastic/E] 取值离散：n_unique=10，最常见值占 58%（更像分类而非回归，损失与评估要按离散值设计）
11. [fem_elastic/nu] 取值离散：n_unique=3，最常见值占 82%（更像分类而非回归，损失与评估要按离散值设计）
12. [liquid_eos/rho] 取值离散：n_unique=2，最常见值占 59%（更像分类而非回归，损失与评估要按离散值设计）
13. [liquid_shear_free/rho] 取值离散：n_unique=14，最常见值占 94%（更像分类而非回归，损失与评估要按离散值设计）
14. [liquid_shear_free/E] 取值离散：n_unique=14，最常见值占 81%（更像分类而非回归，损失与评估要按离散值设计）
15. [liquid_shear_free/nu] 取值离散：n_unique=3，最常见值占 99%（更像分类而非回归，损失与评估要按离散值设计）
16. [plastic_drucker_prager/rho] 取值离散：n_unique=2，最常见值占 56%（更像分类而非回归，损失与评估要按离散值设计）
17. [plastic_drucker_prager/E] 取值离散：n_unique=5，最常见值占 51%（更像分类而非回归，损失与评估要按离散值设计）
18. [plastic_drucker_prager/nu] 取值离散：n_unique=2，最常见值占 56%（更像分类而非回归，损失与评估要按离散值设计）
19. [plastic_snow/rho] 取值离散：n_unique=5，最常见值占 85%（更像分类而非回归，损失与评估要按离散值设计）
20. [plastic_snow/E] 取值离散：n_unique=11，最常见值占 59%（更像分类而非回归，损失与评估要按离散值设计）
21. [plastic_snow/nu] 取值离散：n_unique=3，最常见值占 99%（更像分类而非回归，损失与评估要按离散值设计）
22. [viscoplastic/rho] 取值离散：n_unique=7，最常见值占 74%（更像分类而非回归，损失与评估要按离散值设计）
23. 部分物体 E 或 ρ 缺失/为 0：{'E_bad': {'elastic': 4, 'elastoplastic': 0, 'fem_elastic': 0, 'liquid_eos': 506, 'liquid_shear_free': 1, 'plastic_drucker_prager': 0, 'plastic_snow': 0, 'static': 1, 'viscoplastic': 26}, 'rho_bad': {'elastic': 1, 'elastoplastic': 0, 'fem_elastic': 0, 'liquid_eos': 0, 'liquid_shear_free': 0, 'plastic_drucker_prager': 0, 'plastic_snow': 1, 'static': 1, 'viscoplastic': 0}}（这些材料的参数评估要去掉对应字段）
24. 每帧仿真时间 substeps×dt 在场景间变化（14 种取值）：视频里读不出的潜变量，必须作为预测目标或已知条件
25. code 较长（P99≈4976 tokens），加上视频 token 要检查 max_len 与显存
26. 91% 的程序有近重复拷贝（相似度>0.95）：随机划分会严重泄漏，必须按结构/模板分组划分
27. grid_res 在 >50% 的程序里没能用正则抽到（变量名不规范），一致性结论不可靠
28. yaml 与代码不一致：dt 在 12 个场景不匹配（不一致率 0.1%），需决定以谁为准
29. substeps 在 >50% 的程序里没能用正则抽到（变量名不规范），一致性结论不可靠
30. frames 在 >50% 的程序里没能用正则抽到（变量名不规范），一致性结论不可靠
31. rho 在 >50% 的程序里没能用正则抽到（变量名不规范），一致性结论不可靠
32. yaml 与代码不一致：E 在 111 个场景不匹配（不一致率 1.5%），需决定以谁为准
33. yaml 与代码不一致：nu 在 105 个场景不匹配（不一致率 1.4%），需决定以谁为准
34. 视频与 yaml 的帧数/fps 不完全一致（帧数 85.7%, fps 95.0%）

## 1. 数据概览

- 场景数（yaml 解析成功）: **8457**
- 完整性问题: {'missing_video': 6}
- 物体总数 28551；带本构模型的物体 15316，分布在 8456 个场景

## 2. Schema 与结构

- key path 共 2901 个；**可选字段（出现在 5%~95% 的场景）66 个**
| path | share |
|---|---|
| output.video_framerate | 93.7% |
| objects[].visualization.color | 91.2% |
| simulation.duration_frames | 84.5% |
| objects[].shape.type | 84.1% |
| objects[].initial_state.particle_count | 80.7% |
| output.folder_name | 76.0% |
| objects[].visualization.particle_radius | 61.4% |
| output.background_color | 56.2% |
| objects[].initial_state.initial_velocity | 53.7% |
| objects[].physics_properties.hardening_rate | 44.9% |
| objects[].physics_properties.yield_stretch | 44.8% |
| objects[].physics_properties.yield_compression | 44.8% |
| objects[].physics_properties.hardening_max | 42.1% |
| objects[].physics_properties.hardening_min | 42.1% |
| simulation.quality_factor | 40.5% |
| output.gui_resolution | 38.2% |
| objects[].shape.radius | 35.7% |
| objects[].initial_state.shape.type | 33.5% |
| objects[].physics_properties.hardening_multiplier | 29.5% |
| objects[].initial_state.shape.lower_corner | 28.5% |
| objects[].initial_state.shape.upper_corner | 28.4% |
| objects[].shape.center | 21.9% |
| objects[].visualization | 20.8% |
| objects[].initial_state.initial_region.type | 19.7% |
| environment.boundary_conditions.type | 18.4% |
- otype: dynamic_body=14015, static_collider=6669, kinematic_collider=6473, emitter=1394
- cm: nan=13235, liquid_shear_free=4679, elastic=4285, plastic_snow=3794, plastic_drucker_prager=566, viscoplastic=544, liquid_eos=512, elastoplastic=497, fem_elastic=438, static=1
- em: nan=14197, corotated=12958, neo_hookean=1396
- shape: nan=11702, box=5030, circle=2987, line_segment=577, rectangle=519, gear=491, fan_blades=352, boundary_box=224, cross=214, box_container=207, box_boundary=188, rigid_circle=185
- 每场景物体数分布: {1: 205, 2: 1782, 3: 3075, 4: 2082, 5: 873, 6: 287, 7: 98, 8: 41, 9: 13, 11: 1}
- 结构签名: 唯一 1483 种 / 8457 场景；top1/top5/top10 占比 1.1% / 4.5% / 8.0%

## 3. 物理参数分布

| cm | param | n | n_unique | top_value | top_share | round_share | min | median | max | log10_span |
|---|---|---|---|---|---|---|---|---|---|---|
| elastic | rho | 4284 | 17 | 1 | 0.712 | 1 | 1 | 1 | 50 | 1.7 |
| elastic | E | 4285 | 12 | 5e+03 | 0.64 | 0.999 | 0 | 5e+03 | 1.5e+04 | nan |
| elastic | nu | 4285 | 3 | 0.2 | 0.959 | 1 | 0.2 | 0.2 | 0.3 | 0.176 |
| elastoplastic | rho | 497 | 2 | 1 | 0.976 | 1 | 1 | 1 | 2 | 0.301 |
| elastoplastic | E | 497 | 3 | 5e+03 | 0.899 | 1 | 1e+03 | 5e+03 | 5e+03 | 0.699 |
| elastoplastic | nu | 497 | 1 | 0.2 | 1 | 1 | 0.2 | 0.2 | 0.2 | 0 |
| fem_elastic | rho | 438 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| fem_elastic | E | 438 | 10 | 5e+03 | 0.584 | 1 | 5e+03 | 5e+03 | 8e+04 | 1.2 |
| fem_elastic | nu | 438 | 3 | 0.2 | 0.815 | 0.995 | 0.2 | 0.2 | 0.278 | 0.143 |
| liquid_eos | rho | 512 | 2 | 1 | 0.586 | 1 | 1 | 1 | 400 | 2.6 |
| liquid_eos | E | 6 | 1 | 5e+03 | 1 | 1 | 5e+03 | 5e+03 | 5e+03 | 0 |
| liquid_eos | nu | 6 | 1 | 0.2 | 1 | 1 | 0.2 | 0.2 | 0.2 | 0 |
| liquid_shear_free | rho | 4679 | 14 | 1 | 0.936 | 1 | 0.2 | 1 | 20 | 2 |
| liquid_shear_free | E | 4679 | 14 | 5e+03 | 0.809 | 1 | 0 | 5e+03 | 1e+04 | nan |
| liquid_shear_free | nu | 4653 | 3 | 0.2 | 0.985 | 1 | 0.2 | 0.2 | 0.35 | 0.243 |
| plastic_drucker_prager | rho | 566 | 2 | 400 | 0.56 | 1 | 1 | 400 | 400 | 2.6 |
| plastic_drucker_prager | E | 566 | 5 | 3.54e+05 | 0.509 | 0.491 | 1e+03 | 3.54e+05 | 3.54e+05 | 2.55 |
| plastic_drucker_prager | nu | 566 | 2 | 0.3 | 0.56 | 1 | 0.2 | 0.3 | 0.3 | 0.176 |
| plastic_snow | rho | 3793 | 5 | 1 | 0.853 | 1 | 1 | 1 | 3 | 0.477 |
| plastic_snow | E | 3794 | 11 | 5e+03 | 0.593 | 1 | 500 | 5e+03 | 1e+04 | 1.3 |
| plastic_snow | nu | 3794 | 3 | 0.2 | 0.99 | 1 | 0.2 | 0.2 | 0.3 | 0.176 |
| viscoplastic | rho | 544 | 7 | 1.4 | 0.739 | 1 | 1 | 1.4 | 2.5 | 0.398 |
| viscoplastic | E | 518 | 26 | 1.69e+03 | 0.239 | 0.114 | 788 | 1.47e+03 | 8e+03 | 1.01 |
| viscoplastic | nu | 518 | 17 | 0.227 | 0.295 | 0.122 | 0.18 | 0.219 | 0.312 | 0.24 |

**E/ρ（可辨识的组合量）的取值数：**

| cm | n | n_unique_E_over_rho | top_share |
|---|---|---|---|
| elastic | 4280 | 27 | 0.419 |
| elastoplastic | 497 | 4 | 0.875 |
| fem_elastic | 438 | 10 | 0.584 |
| liquid_eos | 6 | 1 | 1 |
| liquid_shear_free | 4678 | 27 | 0.778 |
| plastic_drucker_prager | 566 | 5 | 0.509 |
| plastic_snow | 3793 | 15 | 0.447 |
| viscoplastic | 518 | 33 | 0.239 |

**E<=0 或缺失、ρ<=0 或缺失的数量：**

| cm | E_bad | rho_bad |
|---|---|---|
| elastic | 4 | 1 |
| elastoplastic | 0 | 0 |
| fem_elastic | 0 | 0 |
| liquid_eos | 506 | 0 |
| liquid_shear_free | 1 | 0 |
| plastic_drucker_prager | 0 | 0 |
| plastic_snow | 0 | 1 |
| static | 1 | 1 |
| viscoplastic | 26 | 0 |
> 本节未生成（params）：TypeError: unsupported type: <class 'method'>

## 4. 仿真配置与时间尺度

| field | n_unique | top_values |
|---|---|---|
| dim | 1 | 2(8456) |
| grid_res | 4 | 128(8393), 64(26), 96(26), 256(11) |
| dt | 6 | 0.0001(5306), 5e-05(2375), 0.0002(375), 2e-05(319) |
| substeps | 17 | 200(2394), 20(2001), 50(1374), 100(1131) |
| frames | 2 | 300(7146), 0.0001(1) |
| fps | 1 | 30(7925) |
| sim_dt_per_frame | 14 | 0.002(2461), 0.01(2304), 0.02(1674), 0.005(1049) |
| sim_time_total | 15 | 0.6(2258), 3(1837), 6(1386), 1.5(962) |
| sim_speed | 14 | 0.06(2363), 0.3(2086), 0.6(1562), 0.15(1000) |

**CFL 型数 dt·√(E/ρ)/dx（粗略指标，E 基于杨氏模量；液体实际应看 λ）：**

| cm | n | median | max | share_gt1 |
|---|---|---|---|---|
| elastic | 4233 | 0.64 | 1.81 | 0.0555 |
| elastoplastic | 496 | 0.905 | 0.905 | 0 |
| fem_elastic | 437 | 0.181 | 7.24 | 0.365 |
| liquid_eos | 6 | 0.905 | 0.905 | 0 |
| liquid_shear_free | 4633 | 0.905 | 4.53 | 0.091 |
| plastic_drucker_prager | 564 | 0.761 | 1.28 | 0.0618 |
| plastic_snow | 3742 | 0.523 | 1.81 | 0.0271 |
| static | 0 | nan | nan | 0 |
| viscoplastic | 516 | 0.415 | 0.957 | 0 |

## 5. 程序统计

- code tokens: P50=2826, P90=3552, P99=4976, max=6208；yaml tokens P50=541
- 超过阈值的比例: > 2048: 97.3%, > 4096: 3.6%, > 8192: 0.0%
- 样板行占比（字符）: mean=0.17, median=0.17；非样板且含数字的行数 median=90
- 模板（数字/注释/空白归一化后完全相同）: 唯一 3325 / 8457；处于大小>1 模板中的场景占 67.0%；最大模板 15 个
- 最近邻余弦相似度: P10=0.955, P50=1.000, P90=1.000, P99=1.000；>0.95 的比例 90.9%

## 6. yaml 与 program.py 一致性

| field | match | mismatch | not_extracted | mismatch_rate |
|---|---|---|---|---|
| grid_res | 2058 | 0 | 6399 | 0 |
| dt | 8055 | 12 | 390 | 0.00149 |
| substeps | 804 | 0 | 7653 | 0 |
| frames | 997 | 0 | 7460 | 0 |
| rho | 1517 | 0 | 6940 | 0 |
| E | 7445 | 111 | 901 | 0.0147 |
| nu | 7268 | 105 | 1084 | 0.0142 |

不一致样例（前 10）：

| scene | field | code_value | yaml_value |
|---|---|---|---|
| mpm_worlds_data_part_01/instances/scene_000093_base | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000001 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000002 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000003 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000004 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000005 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000006 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000007 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000008 | dt | 5e-05 | 5e-06 |
| mpm_worlds_data_part_01/instances/scene_000093_perturbation_000009 | dt | 5e-05 | 5e-06 |

**'定义了但没使用'启发式（名字只出现一次）：**

| name | programs_defining | defined_but_unused | share_unused |
|---|---|---|---|
| rho | 1.61e+03 | 0 | 0 |
| E | 7.85e+03 | 0 | 0 |
| nu | 7.43e+03 | 0 | 0 |
| mu_0 | 7.29e+03 | 2.08e+03 | 0.285 |
| lam_0 | 513 | 12 | 0.0234 |
| mu | 6.31e+03 | 320 | 0.0507 |
| lam | 577 | 0 | 0 |
| friction | 129 | 12 | 0.093 |
| mu_f | 490 | 0 | 0 |
| yield_stress | 18 | 0 | 0 |
| hardening | 137 | 0 | 0 |

## 7. 视频

- 抽查 300 个视频；分辨率 {(512, 512): 274, (640, 640): 25, (800, 800): 1}；fps {30.0: 300}
- 帧数: {'count': 300.0, 'mean': 300.0, 'std': 0.0, 'min': 300.0, '25%': 300.0, '50%': 300.0, '75%': 300.0, 'max': 300.0}
- 与 yaml 一致: 帧数 85.7%, fps 95.0%