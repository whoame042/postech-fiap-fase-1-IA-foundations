# T12 — Qual métrica manda

Classe positiva = **maligno**. Falso negativo = exame maligno predito benigno (atrasa conduta). Falso positivo = biópsia/ansiedade a mais. Neste hospital universitário, **recall (sensibilidade) manda**. F1 desempatá. Accuracy **não** desempata: a classe maligna é 37% (T01); “chutar benigno” já acerta ~63%.

## Vencedor (só validação — antes do teste)

| Modelo | Recall val | F1 val | Accuracy val |
|---|---|---|---|
| **logística** | **0.969** | 0.954 | 0.965 |
| Random Forest | 0.938 | 0.952 | 0.965 |
| KNN | 0.938 | 0.952 | 0.965 |

**vencedor = logística** porque o recall-val (0.969) supera RF e KNN (0.938), com F1-val ligeiramente maior. Grid: `C=1`, `class_weight='balanced'`. O teste abaixo é reportado **uma vez**; não há novo grid.

Fonte: `reports/metrics_val_*.json`.

## Teste (uma vez, sem retreinar)

`reports/metrics_test.csv`. Logística (vencedor da val): accuracy 0.988 · precision 1.00 · recall 0.969 · F1 0.984 · AUC 0.997 · FN=1 · FP=0. Não houve segundo grid.
