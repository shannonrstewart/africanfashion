# Split v1

CLS cosine grouping threshold tau=0.91; seed 42; groups: 7688 (largest 31, mixed-label 15)

| clean_label        |   train |   val |   test |   total |
|:-------------------|--------:|------:|-------:|--------:|
| Adire              |     657 |   132 |    135 |     924 |
| Idoma              |     250 |    51 |     66 |     367 |
| Igala              |     236 |    47 |     50 |     333 |
| Igbo               |    2023 |   404 |    457 |    2884 |
| Tiv                |     362 |    73 |     78 |     513 |
| Tswana-Shweshwe    |     806 |   161 |    164 |    1131 |
| Xhosa-South Africa |     590 |   117 |    121 |     828 |
| Zulu               |     928 |   186 |    189 |    1303 |

Fractions (%): train 70.7, val 14.1, test 15.2

Images forced into test because they (or a duplicate) are in the expert-review sample: 49

Small (<128 px short side) images per split:

| small   |   train |   val |   test |
|:--------|--------:|------:|-------:|
| False   |    5626 |  1116 |   1196 |
| True    |     226 |    55 |     64 |