# Dedup report (pHash Hamming threshold 10/64 same orientation, 4/64 rotated/flipped, worst-cell colour diff <= 15)

| class | before | kept | removed |
|---|---:|---:|---:|
| Igbo | 3872 | 2884 | 988 |
| Zulu | 2187 | 1303 | 884 |
| Xhosa-South Africa | 1486 | 828 | 658 |
| Tswana-Shweshwe | 1333 | 1131 | 202 |
| Idoma | 1149 | 367 | 782 |
| Adire | 1039 | 924 | 115 |
| Tiv | 725 | 513 | 212 |
| Igala | 572 | 333 | 239 |
| **total** | 12363 | 8283 | 4080 |

Removal reasons: kept=8283, duplicate_near=3859, label_conflict=201, duplicate_exact=20

Clusters: 8349; singletons 5531; largest 15; clusters of 5+: 132

Clusters spanning more than one *source* label: Idgo+Igbo: 158, Idgo+Idoma: 1, Idgo+Xhosa-South Africa: 2, Idama+Idoma: 101, Idama+Idoma+Tiv: 1, Idoma+Tiv: 1, Idoma+Zulu: 1, Idoma+Igbo: 1, Igala+Tiv: 1, Tiv+Xhosa-South Africa: 3, Tiv+Xhosa-South Africa+Zulu: 1, Tswana-Shweshwe+Zulu: 24, Tswana-Shweshwe+Xhosa-South Africa: 12, Xhosa-South Africa+Zulu: 17, Igbo+Zulu: 1

Label-conflict clusters (different clean labels, excluded pending review): 66 clusters / 201 images

  Idoma vs Igbo: 2, Igbo vs Xhosa-South Africa: 2, Idoma vs Tiv: 2, Idoma vs Zulu: 1, Igala vs Tiv: 1, Tiv vs Xhosa-South Africa: 3, Tiv vs Xhosa-South Africa vs Zulu: 1, Tswana-Shweshwe vs Zulu: 24, Tswana-Shweshwe vs Xhosa-South Africa: 12, Xhosa-South Africa vs Zulu: 17, Igbo vs Zulu: 1

Flat/near-constant images (excluded from hash matching): 0

Small images (<128 px short side) among kept: 345
