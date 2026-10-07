# Epic 3 — Modelagem

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E3_modelagem`  
Tasks: T09, T10, T11  
Anterior: [E2 Pré-processamento](E2_preprocessamento.md) · Próximo: [E4 Avaliação, interpretação e ética](E4_avaliacao_interpretacao_etica.md)

Critério do PDF: **duas ou mais** técnicas (ex.: regressão logística, árvore, KNN).

Não treinar antes da T08.

Artefatos desta branch:

- T09: `models/logreg.joblib` · `reports/metrics_val_logreg.json`
- T10: `models/rf.joblib` · `reports/metrics_val_rf.json`
- T11: `models/knn.joblib` · `reports/metrics_val_knn.json`
- Protocolo: [modelagem.md](modelagem.md) · `notebooks/02_modelagem.ipynb`

---

## T09 — Modelo A — baseline linear (Regressão logística)

- **Tipo:** obrigatório · ML
- **Depende de:** T08
- **Esforço:** 3 h
- **Aceite:**
  - Treinado **somente** no treino, via pipeline de T06.
  - Hiperparâmetros ajustados na **validação** (ou `GridSearchCV` com CV no treino, sem o teste).
  - Predições da validação salvas para comparação.

## T10 — Modelo B — árvore / ensemble (Decision Tree ou Random Forest)

- **Tipo:** obrigatório · ML
- **Depende de:** T08
- **Esforço:** 3 h
- **Aceite:** iguais a T09, modelo não-linear, mesmo split e mesmas features.

## T11 — Modelo C — KNN ou SVM (recomendado, não estritamente obrigatório)

- **Tipo:** recomendado · ML
- **Depende de:** T08
- **Esforço:** 2 h
- **Por quê:** o PDF lista KNN explicitamente; terceiro modelo fortalece a discussão “por que escolhemos X”.
- **Aceite:** iguais a T09.
