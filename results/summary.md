# DamageNet-VGG16 — results summary

Road-damage patch classification on RDD2022 (4 classes, class-balanced test set of 1200 patches).

## Main results

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Val acc (best epoch) | Epochs run (best) |
|---|---|---|---|---|---|---|
| Baseline CNN (scratch, 64px) | 0.6892 | 0.6853 | 0.6892 | 0.6825 | 0.6713 | 30 (24) |
| VGG16 frozen (feature extraction, A4) | 0.7000 | 0.7050 | 0.7000 | 0.6992 | 0.7388 | 9 (4) |
| VGG16 fine-tuned (block 5) | 0.7167 | 0.7162 | 0.7167 | 0.7152 | 0.7462 | 4 (1) |

## Regularisation ablation (stage 1 head, cumulative)

| Run | Added | Test accuracy | Macro F1 | Final train acc | Final val acc |
|---|---|---|---|---|---|
| A0 | none | 0.6883 | 0.6890 | 1.000 | 0.690 |
| A1 | + dropout 0.5 | 0.6883 | 0.6889 | 0.996 | 0.695 |
| A2 | + L2 1e-3 | 0.6925 | 0.6930 | 0.996 | 0.703 |
| A3 | + horizontal-flip copies | 0.7083 | 0.7080 | 0.993 | 0.716 |
| A4 | + early stopping | 0.7000 | 0.6992 | 0.918 | 0.730 |

## Per-class F1

| Class | Baseline CNN (scratch, 64px) | VGG16 frozen (feature extraction, A4) | VGG16 fine-tuned (block 5) |
|---|---|---|---|
| D00 longitudinal crack | 0.680 | 0.664 | 0.677 |
| D10 transverse crack | 0.677 | 0.668 | 0.674 |
| D20 alligator crack | 0.560 | 0.649 | 0.668 |
| D40 pothole | 0.812 | 0.816 | 0.842 |

## Error analysis (VGG16 fine-tuned)

- Errors: 340 / 1200; **71.2% of errors are confusions between crack types** (D00/D10/D20).
- Top confusions: D10 → D00 (53), D10 → D20 (46), D20 → D00 (41), D00 → D20 (36), D20 → D10 (35), D00 → D10 (31)
- Accuracy by box size (thirds of max side, px): small 31–104: 0.780, medium 105–218: 0.651, large 219–2145: 0.719
- Accuracy by country: China_Drone 0.816 (n=49), China_MotorBike 0.698 (n=86), Czech 0.759 (n=29), India 0.774 (n=239), Japan 0.699 (n=422), Norway 0.686 (n=172), United_States 0.690 (n=203)

## Data

- Annotated images parsed: 35322; boxes kept: D00 26015, D10 11830, D20 10614, D40 6544
- Patches per class: train/val/test = 900/200/300

## Runtime

Pipeline steps (image downloads were already cached; a first run adds the rate-limited download time, ~45 min anonymous):

- 1. download + parse annotations: 0.1 min
- 2. build patch dataset: 0.6 min
- 3. baseline CNN: 1.9 min
- 4. VGG16 feature caching: 1.1 min
- 5. VGG16 stage 1 + ablation: 0.8 min
- 6. VGG16 stage 2 fine-tuning: 0.9 min
- 7. analysis: 0.0 min
- 8. figures: 0.0 min
- total: 5.4 min
