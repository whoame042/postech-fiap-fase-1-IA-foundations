# Epic 4 — Avaliação, interpretação e ética clínica

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E4_avaliacao_interpretacao_etica`  
Tasks: T12, T13, T14, T15  
Anterior: [E3 Modelagem](E3_modelagem.md) · Próximo: [E5 EXTRA CNN](E5_extra_cnn.md)

Não olhar o teste antes da T12.

Artefatos desta branch:

- T12: [metrica.md](metrica.md) · `reports/metrics_test.csv` · `reports/figures/03_*.png`
- T13: [importance.md](importance.md) · `reports/figures/04_importance.png`
- T14: [shap.md](shap.md) · `reports/figures/05_shap_*.png`
- T15: [uso_pratico.md](uso_pratico.md)
- Notebook: `notebooks/03_avaliacao.ipynb`

---

## T12 — Avaliação no teste e escolha da métrica

- **Tipo:** obrigatório · ML
- **Depende de:** T09, T10
- **Esforço:** 3 h
- **Aceite:**
  - No **teste**: accuracy, recall (sensibilidade), precision, F1, matriz de confusão. AUC-ROC é plus (aparece na disciplina).
  - Tabela comparando os modelos.
  - Parágrafo obrigatório: **qual métrica manda neste problema e por quê**.
    - Ex. câncer: falso negativo (recall baixo) é mais grave que falso positivo.
    - Accuracy sozinha é insuficiente se a classe for desbalanceada.
  - Modelo “vencedor” escolhido com base na validação, métricas de teste reportadas **uma vez** (sem retreinar olhando o teste).

## T13 — Feature importance

- **Tipo:** obrigatório · interpretação
- **Depende de:** T12
- **Esforço:** 2 h
- **Aceite:**
  - Importância do modelo vencedor (coeficientes padronizados **ou** `feature_importances_`).
  - Gráfico + texto ligando as top features ao significado clínico (sem afirmar causalidade).

## T14 — SHAP

- **Tipo:** obrigatório · interpretação
- **Depende de:** T12
- **Esforço:** 3 h
- **Aceite:**
  - `summary plot` (impacto global) + pelo menos 1 `waterfall`/`force` de um caso.
  - Explicação em linguagem de médico: “esta amostra foi classificada X porque…”.
  - Limitação do SHAP citada (explicação do modelo, não da biologia).

## T15 — Discussão crítica de uso na prática

- **Tipo:** obrigatório · produto/ética
- **Depende de:** T12, T13, T14
- **Esforço:** 2 h
- **Aceite (checklist do enunciado):**
  - O modelo **pode** ser usado na prática? Em qual papel (triagem, segunda leitura, priorização de fila)?
  - O que quebra: drift, população diferente do dataset, desbalanceamento, viés.
  - Frase explícita: **o médico tem a palavra final**.
  - LGPD: dataset público, sem PII real; se fosse hospital, dado de saúde é sensível.
