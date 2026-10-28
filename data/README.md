# Data

| Folder | Contents |
|---|---|
| `raw/sonar_data.csv` | Original Gorman & Sejnowski sonar dataset (UCI). 208 rows, 60 numeric features, label in column 61. No header row. |
| `processed/train.csv` | 166 rows (80 %), stratified. Written by `python -m src.train`. |
| `processed/test.csv` | 42 rows (20 %), stratified. Never used for training or model selection. |

**Features.** Each of the 60 columns is the energy returned in one frequency band of a chirped sonar pulse, integrated over time. Values are normalised by the dataset authors to the range 0.0–1.0. Band 1 is the lowest frequency.

**Label.** `M` = mine (metal cylinder), `R` = rock. 111 mines and 97 rocks (53 % / 47 %).

**Source.** Gorman, R. P., & Sejnowski, T. J. (1988). *Analysis of hidden units in a layered network trained to classify sonar targets.* Neural Networks, 1(1), 75–89. Dataset: UCI Machine Learning Repository, "Connectionist Bench (Sonar, Mines vs. Rocks)".
