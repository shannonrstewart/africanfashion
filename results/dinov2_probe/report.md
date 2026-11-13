# DINOv2 linear probe

Frozen `facebook/dinov2-base`, features `cls+patchmean`, logistic regression C=0.1 (class-balanced), chosen on validation macro-F1.

## Validation grid

| features      |     C |   val_acc |   val_macro_f1 |
|:--------------|------:|----------:|---------------:|
| cls           | 0.001 |    0.8352 |         0.8067 |
| cls           | 0.01  |    0.8676 |         0.8418 |
| cls           | 0.1   |    0.8642 |         0.835  |
| cls           | 1     |    0.8514 |         0.8186 |
| cls+patchmean | 0.001 |    0.8446 |         0.819  |
| cls+patchmean | 0.01  |    0.8668 |         0.8408 |
| cls+patchmean | 0.1   |    0.8711 |         0.8424 |
| cls+patchmean | 1     |    0.8642 |         0.8415 |

## Test results

- accuracy **0.869** (95% CI 0.851-0.886), macro-F1 **0.846** (95% CI 0.821-0.869), n=1260

|                    |   precision |   recall |   f1-score |   support |
|:-------------------|------------:|---------:|-----------:|----------:|
| Adire              |       0.913 |    0.859 |      0.885 |       135 |
| Idoma              |       0.893 |    0.758 |      0.82  |        66 |
| Igala              |       0.82  |    0.82  |      0.82  |        50 |
| Igbo               |       0.9   |    0.926 |      0.913 |       457 |
| Tiv                |       0.772 |    0.782 |      0.777 |        78 |
| Tswana-Shweshwe    |       0.858 |    0.884 |      0.871 |       164 |
| Xhosa-South Africa |       0.843 |    0.884 |      0.863 |       121 |
| Zulu               |       0.835 |    0.804 |      0.819 |       189 |
| macro avg          |       0.854 |    0.84  |      0.846 |      1260 |

## By subset

| subset | n | accuracy | macro-F1 | ladder: specific % | wrong-specific % | collapse % |
|---|---:|---:|---:|---:|---:|---:|
| test (all) | 1260 | 0.869 | 0.846 | 87.5 | 4.8 | 0.0 |
| test: small (<128 px) | 64 | 0.891 | 0.727 | 89.1 | 6.2 | 0.0 |
| test: >=128 px | 1196 | 0.868 | 0.846 | 87.4 | 4.7 | 0.0 |
| test: expert-review images | 26 | 0.923 | 0.915 | 92.3 | 0.0 | 0.0 |

## Granularity ladder, all test images

```
class                    n     L1    L2    L3    L4     r0    r1    r2    r3   mean
-----------------------------------------------------------------------------------
Adire                  135  100.0  95.6  85.9  85.9    0.0   4.4   9.6  85.9   2.81
Idoma                   66  100.0  98.5  75.8  75.8    0.0   1.5  22.7  75.8   2.74
Igala                   50  100.0  96.0  82.0  82.0    0.0   4.0  14.0  82.0   2.78
Igbo                   457  100.0  96.1  92.6  92.6    0.0   3.9   3.5  92.6   2.89
Tiv                     78  100.0  93.6  78.2  78.2    0.0   6.4  15.4  78.2   2.72
Tswana-Shweshwe        164  100.0  96.3  92.7  88.4    0.0   3.7   3.7  92.7   2.89
Xhosa-South Africa     121  100.0  98.3  88.4  88.4    0.0   1.7   9.9  88.4   2.87
Zulu                   189  100.0  89.4  80.4  80.4    0.0  10.6   9.0  80.4   2.70
OVERALL               1260  100.0  95.2  87.5  86.9    0.0   4.8   7.8  87.5   2.83

Monolithic-collapse rate (said only 'African'):   0.0%
Wrong-specific rate (named a wrong place/people):   4.8%
Specific rate (people or technique correct):    87.5%
Generic-term rate (ankara, wax print, ...):      0.0%
```

_A closed-set classifier always answers with one of the 8 classes, so L1 is 100% by construction and its collapse rate is 0%; its rung-1 answers are all wrong-specific (e.g. a Zulu image called Igbo). These numbers are the reference point for the VLM comparison, which has to produce answers open-endedly._
_Gold for Tswana-Shweshwe, Xhosa and Zulu is still a draft awaiting expert verification._