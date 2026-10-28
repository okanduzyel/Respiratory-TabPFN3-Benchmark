# Revision analysis results

Backend: {'backend': 'cloud', 'version': 'tabpfn_client ?'}  
Cycles: 6860, patients: 123

## Table A2: patients / cycles per class in each test fold (Stratified Group 5-fold, seed 42)

| Class | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| Bronchiectasis | 1 / 41 | 4 / 45 | 0 / 0 | 0 / 0 | 2 / 18 |
| Bronchiolitis | 1 / 13 | 2 / 50 | 3 / 97 | 0 / 0 | 0 / 0 |
| COPD | 11 / 869 | 13 / 1421 | 12 / 1228 | 14 / 1277 | 14 / 951 |
| Healthy | 7 / 90 | 3 / 33 | 6 / 91 | 6 / 56 | 4 / 52 |
| Pneumonia | 1 / 26 | 0 / 0 | 2 / 71 | 1 / 112 | 2 / 76 |
| URTI | 3 / 51 | 5 / 127 | 2 / 20 | 3 / 37 | 1 / 8 |

Every class present in every training and test fold (all schemes): False

## Table 6: patient-level 5-fold (mean ± std over folds)

| Model | Precision (macro) | Recall (macro) | F1 (macro) | Accuracy | MCC | Weighted F1 |
|---|---|---|---|---|---|---|
| TabPFN-3 | 0.4731±0.0915 | 0.4469±0.1257 | 0.4346±0.1024 | 0.8736±0.0224 | 0.5665±0.0458 | 0.8762±0.0199 |
| Naive Bayes | 0.3264±0.0427 | 0.3297±0.0522 | 0.3024±0.0305 | 0.7495±0.0434 | 0.3833±0.0674 | 0.7974±0.0242 |
| Logistic Regression | 0.3495±0.1019 | 0.2846±0.0579 | 0.2775±0.0543 | 0.8441±0.0049 | 0.3766±0.1101 | 0.8217±0.0021 |
| SVM | 0.4024±0.0992 | 0.3241±0.0765 | 0.3201±0.0611 | 0.8379±0.0225 | 0.4422±0.0628 | 0.8399±0.0107 |
| KNN | 0.3398±0.0329 | 0.2878±0.0462 | 0.2867±0.0267 | 0.8252±0.0192 | 0.3740±0.0385 | 0.8200±0.0218 |
| Decision Tree | 0.3192±0.0170 | 0.2811±0.0285 | 0.2791±0.0184 | 0.8153±0.0236 | 0.3667±0.0437 | 0.8156±0.0198 |
| MLP | 0.4279±0.0922 | 0.3397±0.0751 | 0.3125±0.0572 | 0.8384±0.0314 | 0.4204±0.0949 | 0.8324±0.0223 |
| Random Forest | 0.3383±0.0641 | 0.2545±0.0281 | 0.2501±0.0152 | 0.8447±0.0243 | 0.3680±0.0577 | 0.8152±0.0231 |
| XGBoost | 0.4423±0.0716 | 0.2964±0.0231 | 0.3082±0.0339 | 0.8473±0.0265 | 0.4311±0.0404 | 0.8360±0.0196 |

## Table 9: distillation, patient-level 5-fold

| Model | Precision (macro) | Recall (macro) | F1 (macro) | Accuracy | MCC | Weighted F1 |
|---|---|---|---|---|---|---|
| TabPFN-3 | 0.4731±0.0915 | 0.4469±0.1257 | 0.4346±0.1024 | 0.8736±0.0224 | 0.5665±0.0458 | 0.8762±0.0199 |
| Student XGBoost | 0.3828±0.0316 | 0.2900±0.0274 | 0.3019±0.0263 | 0.8441±0.0266 | 0.4062±0.0488 | 0.8290±0.0244 |
| Student Decision Tree | 0.2867±0.0723 | 0.2840±0.0441 | 0.2679±0.0508 | 0.7981±0.0255 | 0.3202±0.0398 | 0.7999±0.0202 |
| Student MLP | 0.2200±0.0894 | 0.2461±0.0732 | 0.2051±0.0572 | 0.8317±0.0227 | 0.2703±0.2011 | 0.7954±0.0470 |
| Student Transformer | 0.4172±0.0713 | 0.3517±0.0841 | 0.3440±0.0784 | 0.8477±0.0272 | 0.4837±0.0825 | 0.8508±0.0210 |
| Student Decision Tree (depth 4) | 0.2347±0.0509 | 0.2536±0.0216 | 0.2154±0.0313 | 0.8212±0.0325 | 0.3356±0.0409 | 0.7985±0.0200 |

## Repeated 3x5 (15 folds): overall (mean ± std)

| Model | Precision (macro) | Recall (macro) | F1 (macro) | Accuracy | MCC | Weighted F1 |
|---|---|---|---|---|---|---|
| TabPFN-3 | 0.5168±0.0641 | 0.5434±0.0691 | 0.5010±0.0625 | 0.8722±0.0393 | 0.5788±0.0755 | 0.8687±0.0452 |
| Naive Bayes | 0.3388±0.0547 | 0.4065±0.0830 | 0.3371±0.0578 | 0.7581±0.0453 | 0.3973±0.0649 | 0.7972±0.0378 |
| Logistic Regression | 0.3471±0.0621 | 0.3400±0.0668 | 0.3217±0.0608 | 0.8442±0.0430 | 0.4111±0.0799 | 0.8191±0.0566 |
| SVM | 0.4125±0.0705 | 0.4057±0.0864 | 0.3707±0.0588 | 0.8378±0.0475 | 0.4625±0.0775 | 0.8333±0.0562 |
| KNN | 0.3112±0.0337 | 0.3072±0.0487 | 0.2948±0.0354 | 0.8190±0.0472 | 0.3717±0.0692 | 0.8115±0.0532 |
| Decision Tree | 0.3294±0.0528 | 0.3416±0.0583 | 0.3176±0.0489 | 0.8172±0.0393 | 0.3810±0.0592 | 0.8151±0.0445 |
| MLP | 0.4063±0.0858 | 0.3912±0.0712 | 0.3571±0.0575 | 0.8370±0.0461 | 0.4521±0.0831 | 0.8320±0.0518 |
| Random Forest | 0.3660±0.1262 | 0.2725±0.0312 | 0.2743±0.0362 | 0.8472±0.0415 | 0.3830±0.0529 | 0.8141±0.0534 |
| XGBoost | 0.4488±0.0442 | 0.3666±0.0590 | 0.3669±0.0508 | 0.8525±0.0416 | 0.4598±0.0705 | 0.8352±0.0522 |
| Student XGBoost | 0.4271±0.0917 | 0.3741±0.0693 | 0.3670±0.0573 | 0.8490±0.0433 | 0.4315±0.0744 | 0.8280±0.0532 |
| Student Decision Tree | 0.3382±0.0587 | 0.3450±0.0635 | 0.3233±0.0517 | 0.8187±0.0426 | 0.3910±0.0608 | 0.8160±0.0493 |
| Student MLP | 0.2669±0.0697 | 0.2954±0.0738 | 0.2482±0.0459 | 0.8461±0.0407 | 0.3965±0.1236 | 0.8102±0.0586 |
| Student Transformer | 0.4180±0.0898 | 0.4166±0.0601 | 0.3883±0.0658 | 0.8585±0.0413 | 0.5140±0.0857 | 0.8507±0.0470 |
| Student Decision Tree (depth 4) | 0.2513±0.0640 | 0.2862±0.0338 | 0.2430±0.0296 | 0.8225±0.0501 | 0.3756±0.0546 | 0.8016±0.0569 |
| MLP hard labels only (alpha=0) | 0.2291±0.0427 | 0.2696±0.0540 | 0.2386±0.0416 | 0.8505±0.0401 | 0.3869±0.1367 | 0.8097±0.0572 |
| MLP soft labels only (alpha=1) | 0.2791±0.0766 | 0.3165±0.0626 | 0.2750±0.0674 | 0.8483±0.0412 | 0.4434±0.1060 | 0.8196±0.0594 |

## Table 7 / A3: TabPFN-3 per class, 5-fold

| Class | Precision | Recall (sensitivity) | Specificity | F1 | per-fold F1 |
|---|---|---|---|---|---|
| Bronchiectasis | 0.5609±0.4635 | 0.5449±0.4485 | 0.9978±0.0024 | 0.5524±0.4551 | 0.949, 0.813, 0.000, 0.000, 1.000 |
| Bronchiolitis | 0.0761±0.0801 | 0.1835±0.3318 | 0.9831±0.0135 | 0.0878±0.1280 | 0.338, 0.029, 0.071, 0.000, 0.000 |
| COPD | 0.9661±0.0253 | 0.9696±0.0190 | 0.7987±0.1785 | 0.9674±0.0096 | 0.964, 0.965, 0.986, 0.961, 0.961 |
| Healthy | 0.4600±0.1848 | 0.4716±0.1040 | 0.9691±0.0168 | 0.4401±0.1234 | 0.372, 0.304, 0.359, 0.527, 0.637 |
| Pneumonia | 0.4830±0.2649 | 0.2688±0.1994 | 0.9818±0.0157 | 0.3138±0.1906 | 0.500, 0.000, 0.385, 0.197, 0.486 |
| URTI | 0.2922±0.2590 | 0.2428±0.1542 | 0.9683±0.0190 | 0.2463±0.1993 | 0.321, 0.562, 0.068, 0.281, 0.000 |

## Student MLP per class, 5-fold

| Class | Precision | Recall (sensitivity) | Specificity | F1 | per-fold F1 |
|---|---|---|---|---|---|
| Bronchiectasis | 0.1000±0.2000 | 0.0111±0.0222 | 0.9983±0.0029 | 0.0200±0.0400 | 0.000, 0.000, 0.000, 0.000, 0.100 |
| Bronchiolitis | 0.0000±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.0000±0.0000 | 0.000, 0.000, 0.000, 0.000, 0.000 |
| COPD | 0.9037±0.0632 | 0.9732±0.0342 | 0.4509±0.3582 | 0.9351±0.0307 | 0.887, 0.971, 0.966, 0.926, 0.925 |
| Healthy | 0.0918±0.1332 | 0.3256±0.4096 | 0.9614±0.0473 | 0.1406±0.1977 | 0.000, 0.198, 0.000, 0.000, 0.505 |
| Pneumonia | 0.2171±0.2772 | 0.0868±0.1071 | 0.9905±0.0100 | 0.1214±0.1487 | 0.000, 0.000, 0.304, 0.000, 0.303 |
| URTI | 0.0072±0.0145 | 0.0800±0.1600 | 0.9710±0.0571 | 0.0133±0.0266 | 0.000, 0.000, 0.066, 0.000, 0.000 |

## Table A3: TabPFN-3 sensitivity / specificity (pooled 5-fold OOF), 95% patient-bootstrap CI

| Class | Sens. (cycle) | Spec. (cycle) | Sens. (patient) | Spec. (patient) |
|---|---|---|---|---|
| Bronchiectasis | 0.885 [0.734, 0.987] | 0.997 [0.995, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Bronchiolitis | 0.106 [0.005, 0.333] | 0.984 [0.971, 0.993] | 0.167 [0.000, 0.500] | 0.983 [0.957, 1.000] |
| COPD | 0.970 [0.949, 0.986] | 0.823 [0.696, 0.935] | 0.922 [0.859, 0.984] | 0.949 [0.898, 1.000] |
| Healthy | 0.438 [0.317, 0.579] | 0.967 [0.954, 0.978] | 0.538 [0.346, 0.731] | 0.897 [0.845, 0.948] |
| Pneumonia | 0.267 [0.146, 0.479] | 0.980 [0.965, 0.991] | 0.333 [0.000, 0.667] | 0.940 [0.897, 0.974] |
| URTI | 0.362 [0.162, 0.497] | 0.969 [0.953, 0.981] | 0.429 [0.143, 0.714] | 0.890 [0.835, 0.945] |

| Summary | value [95% CI] |
|---|---|
| cycle_macro_f1 | 0.501 [0.445, 0.559] |
| cycle_mcc | 0.560 [0.491, 0.632] |
| cycle_balanced_acc | 0.505 [0.452, 0.567] |
| patient_accuracy | 0.724 [0.659, 0.789] |
| patient_balanced_acc | 0.565 [0.473, 0.661] |
| patient_macro_f1 | 0.560 [0.467, 0.651] |
| patient_mcc | 0.590 [0.497, 0.683] |

Patient-level confusion matrix (rows true, cols pred; order ['bronchiectasis', 'bronchiolitis', 'copd', 'healthy', 'pneumonia', 'urti']):

```
[[ 7  0  0  0  0  0]
 [ 0  1  0  3  0  2]
 [ 0  0 59  0  4  1]
 [ 0  2  0 14  2  8]
 [ 0  0  2  1  2  1]
 [ 0  0  1  6  1  6]]
```

## Table 8 (mcc): TabPFN-3 vs baselines, Nadeau-Bengio, 15 folds, rho=0.2515

| Baseline | Mean delta | t | p | Holm p_adj |
|---|---|---|---|---|
| Naive Bayes | 0.1815 | 3.1653 | 0.00688 | 0.01330 |
| Logistic Regression | 0.1677 | 3.3862 | 0.00443 | 0.01330 |
| SVM | 0.1163 | 3.2679 | 0.00561 | 0.01330 |
| KNN | 0.2072 | 6.0777 | 0.00003 | 0.00020 |
| Decision Tree | 0.1978 | 5.0703 | 0.00017 | 0.00102 |
| MLP | 0.1267 | 3.5333 | 0.00331 | 0.01324 |
| Random Forest | 0.1958 | 6.3742 | 0.00002 | 0.00014 |
| XGBoost | 0.1190 | 3.8953 | 0.00162 | 0.00808 |

## Table 8 (macro_f1): TabPFN-3 vs baselines, Nadeau-Bengio, 15 folds, rho=0.2515

| Baseline | Mean delta | t | p | Holm p_adj |
|---|---|---|---|---|
| Naive Bayes | 0.1639 | 3.6574 | 0.00259 | 0.01492 |
| Logistic Regression | 0.1794 | 3.3658 | 0.00462 | 0.01846 |
| SVM | 0.1303 | 2.6322 | 0.01970 | 0.03941 |
| KNN | 0.2062 | 6.0500 | 0.00003 | 0.00024 |
| Decision Tree | 0.1834 | 3.6774 | 0.00249 | 0.01492 |
| MLP | 0.1439 | 3.1456 | 0.00715 | 0.02146 |
| Random Forest | 0.2267 | 4.9819 | 0.00020 | 0.00141 |
| XGBoost | 0.1341 | 2.5821 | 0.02172 | 0.03941 |

## Table 11 (mcc): teacher vs students, Nadeau-Bengio, 15 folds

| Student | Mean loss | t | p | Holm p_adj |
|---|---|---|---|---|
| Student XGBoost | 0.1473 | 4.6778 | 0.00036 | 0.00107 |
| Student Decision Tree | 0.1878 | 5.5120 | 0.00008 | 0.00038 |
| Student MLP | 0.1823 | 1.8874 | 0.08002 | 0.08002 |
| Student Transformer | 0.0648 | 2.3254 | 0.03559 | 0.07118 |
| Student Decision Tree (depth 4) | 0.2032 | 4.9557 | 0.00021 | 0.00084 |

## Table 11 (macro_f1): teacher vs students, Nadeau-Bengio, 15 folds

| Student | Mean loss | t | p | Holm p_adj |
|---|---|---|---|---|
| Student XGBoost | 0.1340 | 2.5431 | 0.02343 | 0.02470 |
| Student Decision Tree | 0.1777 | 3.9349 | 0.00150 | 0.00449 |
| Student MLP | 0.2528 | 5.2817 | 0.00012 | 0.00046 |
| Student Transformer | 0.1127 | 2.8700 | 0.01235 | 0.02470 |
| Student Decision Tree (depth 4) | 0.2580 | 6.4864 | 0.00001 | 0.00007 |

## Table 11 (macro_recall): teacher vs students, Nadeau-Bengio, 15 folds

| Student | Mean loss | t | p | Holm p_adj |
|---|---|---|---|---|
| Student XGBoost | 0.1693 | 2.8554 | 0.01271 | 0.02543 |
| Student Decision Tree | 0.1984 | 4.7730 | 0.00030 | 0.00119 |
| Student MLP | 0.2480 | 3.9441 | 0.00147 | 0.00441 |
| Student Transformer | 0.1268 | 2.8370 | 0.01318 | 0.02543 |
| Student Decision Tree (depth 4) | 0.2572 | 5.2772 | 0.00012 | 0.00058 |

## TOST on MCC (teacher - student), margin 0.05, NB-corrected 90% CI

| Student | Mean diff | 90% CI | p_TOST | Equivalent |
|---|---|---|---|---|
| Student MLP | 0.1823 | [0.0122, 0.3524] | 0.9038 | False |
| Student Transformer | 0.0648 | [0.0157, 0.1138] | 0.6980 | False |

## Table A4: hard vs soft vs combined (Student MLP, 15 folds)

| Target | Accuracy | MCC | Macro F1 | Macro Recall |
|---|---|---|---|---|
| Hard labels only (alpha = 0) | 0.8505±0.0401 | 0.3869±0.1367 | 0.2386±0.0416 | 0.2696±0.0540 |
| Soft labels only (alpha = 1) | 0.8483±0.0412 | 0.4434±0.1060 | 0.2750±0.0674 | 0.3165±0.0626 |
| Combined loss (alpha = 0.5) | 0.8461±0.0407 | 0.3965±0.1236 | 0.2482±0.0459 | 0.2954±0.0738 |

Combined vs MLP hard labels only (alpha=0): delta MCC = 0.0096, t = 0.090, p = 0.9295

Combined vs MLP soft labels only (alpha=1): delta MCC = -0.0469, t = -0.501, p = 0.6244

## Table 1: feature-group permutation importance, TabPFN-3, 15 folds

| Group | N_g | Total weight | Per-feature weight | Mean accuracy drop | Mean MCC drop |
|---|---|---|---|---|---|
| Spectral Domain Metrics | 6 | 0.4059±0.1216 | 0.0677±0.0203 | 0.0428±0.0148 | 0.1630±0.0554 |
| Adventitious Sounds | 2 | 0.0857±0.0889 | 0.0429±0.0444 | 0.0073±0.0064 | 0.0281±0.0189 |
| Spectral Cepstral (Mean) | 20 | 0.3003±0.1251 | 0.0150±0.0063 | 0.0360±0.0219 | 0.3126±0.0693 |
| Temporal Domain Metrics | 4 | 0.1077±0.0808 | 0.0269±0.0202 | 0.0142±0.0116 | 0.0994±0.0375 |
| Spectral Cepstral (Std) | 20 | 0.1004±0.0559 | 0.0050±0.0028 | 0.0122±0.0084 | 0.0688±0.0251 |

## SHAP (TabPFN-3, 150 explained test cycles over 5 folds)

Top 20 (mean |SHAP| over samples and classes):

- mfcc_mean_1: 0.0484
- spectral_bandwidth_mean: 0.0338
- rms_mean: 0.0214
- mfcc_mean_4: 0.0177
- zcr_mean: 0.0172
- rms_std: 0.0159
- crackle: 0.0121
- mfcc_mean_7: 0.0110
- mfcc_mean_6: 0.0108
- mfcc_mean_20: 0.0103
- mfcc_mean_2: 0.0085
- spectral_rolloff_mean: 0.0084
- spectral_centroid_mean: 0.0083
- mfcc_mean_3: 0.0082
- mfcc_std_8: 0.0079
- mfcc_mean_19: 0.0078
- wheeze: 0.0073
- mfcc_mean_5: 0.0072
- mfcc_mean_8: 0.0069
- mfcc_mean_15: 0.0068

Fold stability: mean pairwise Spearman = 0.856 (min 0.819); mean top-10 overlap = 0.75

Class-specific top 5:

- Bronchiectasis: mfcc_mean_1 (0.1823), rms_mean (0.1209), spectral_bandwidth_mean (0.0673), zcr_mean (0.0663), mfcc_mean_4 (0.0543)
- Bronchiolitis: mfcc_mean_1 (0.0249), rms_mean (0.0238), mfcc_mean_17 (0.0214), mfcc_mean_7 (0.0174), spectral_bandwidth_mean (0.0157)
- COPD: spectral_bandwidth_mean (0.0676), mfcc_mean_1 (0.0507), crackle (0.0165), rms_std (0.0146), zcr_mean (0.0142)
- Healthy: mfcc_mean_1 (0.0662), spectral_bandwidth_mean (0.0541), mfcc_mean_4 (0.0446), zcr_mean (0.0343), rms_mean (0.0298)
- Pneumonia: mfcc_mean_1 (0.0473), spectral_bandwidth_mean (0.0371), rms_mean (0.0337), mfcc_mean_5 (0.0221), mfcc_mean_6 (0.0215)
- URTI: mfcc_mean_1 (0.0542), spectral_bandwidth_mean (0.0385), rms_mean (0.0264), rms_std (0.0263), mfcc_mean_4 (0.0219)

## Table 10: rules from the distilled depth-4 decision tree (pooled data)

| Class | Rule | Confidence | Support | Lift |
|---|---|---|---|---|
| COPD | mfcc_mean_1 > -410.5 AND mfcc_std_4 ≤ 25.78 AND mfcc_mean_4 > -7.619 | 99.6% | 3994 | 1.19 |
| Healthy | mfcc_mean_20 > 2.83 AND mfcc_std_8 > 5.505 AND spectral_bandwidth_mean > 455.5 AND mfcc_mean_1 ≤ -441.4 | 28.5% | 772 | 6.07 |

Agreement of the depth-4 tree with the teacher (argmax, pooled): 0.861

(min support 20; all leaves in student_tree_all_leaves.csv; full tree in student_tree_full.txt)

Student parameters: {'Student MLP': 5670, 'Student Transformer': 70726}
