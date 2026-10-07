# T09–T11 — Modelagem (só validação)

Mesmo split da T08 (`models/split.joblib`, seed 42). Mesmas 30 features. Única diferença: o classificador no `Pipeline` com o preprocess da T06. `GridSearchCV` (cv=5, scoring=`recall`) **só em X_train**. Métricas abaixo são de **validação**. Teste lacrado até T12.

Classe positiva = maligno (`y=1`). `class_weight='balanced'` na logística e na RF (desbalanceamento da T01).

## Comparação na validação

| Modelo | Best params | Accuracy val | Precision val | Recall val | F1 val |
|---|---|---|---|---|---|
| A logística | `C=1` | 0.965 | 0.939 | **0.969** | 0.954 |
| B Random Forest | `n_estimators=100`, `max_depth=None` | 0.965 | 0.968 | 0.938 | 0.952 |
| C KNN | `n_neighbors=3`, `weights=uniform` | 0.965 | 0.968 | 0.938 | 0.952 |

Números exatos: `reports/metrics_val_logreg.json`, `metrics_val_rf.json`, `metrics_val_knn.json`. Predições: `reports/pred_val_*.csv`.

## Grid da logística (C)

Scoring da grid = recall em CV no **treino**, não F1 no teste.

| C | Papel |
|---|---|
| 0.01 | mais regularizado |
| 0.1 | |
| **1** | escolhido (melhor CV-recall) |
| 10 | menos regularizado |

## KNN e dimensionalidade

30 features contínuas: a vizinhança do KNN dilui (maldição da dimensionalidade). Se perder na T12, o argumento é este — não refazer o split.

## O que não foi feito

Não há `classification_report(y_test)` aqui. Vencedor formal na T12, olhando a val e reportando o teste **uma vez**.
