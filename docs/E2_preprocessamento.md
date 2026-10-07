# Epic 2 — Pré-processamento

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E2_preprocessamento`  
Tasks: T05, T06, T07, T08  
Anterior: [E1 Exploração de dados](E1_exploracao_de_dados.md) · Próximo: [E3 Modelagem](E3_modelagem.md)

Artefatos desta branch:

- T05: [limpeza.md](limpeza.md) · `src/preprocess.py` (`load_and_clean`)
- T06: `ColumnTransformer` unfitted · `models/preprocess_unfitted.joblib`
- T07: [correlacao.md](correlacao.md) · `reports/figures/02_correlacao.png`
- T08: [split.md](split.md) · `models/split.joblib` · `src/train.py` (`make_split`)

---

## T05 — Limpeza (ausentes e inconsistentes)

- **Tipo:** obrigatório · dados
- **Depende de:** T04
- **Esforço:** 2 h
- **Aceite:**
  - Estratégia de missing documentada (drop, imputação mediana/moda, indicador de ausência).
  - Checagem de duplicatas, valores impossíveis (ex.: idade negativa, zeros sentinela no Pima Diabetes).
  - Antes/depois do shape registrado.

## T06 — Pipeline de pré-processamento (sklearn)

- **Tipo:** obrigatório · ML
- **Depende de:** T05
- **Esforço:** 3 h
- **Aceite:**
  - `sklearn.pipeline.Pipeline` + `ColumnTransformer`: numéricas (scaler) e categóricas (OneHot/Ordinal) **fit só no treino**.
  - Nenhuma transformação “na mão” no conjunto inteiro antes do split (isso é leakage — o split oficial é T08; aqui o pipeline deve estar **pronto para ser fitado depois do split**).
  - Pipeline serializável (`joblib`).

## T07 — Análise de correlação

- **Tipo:** obrigatório · dados
- **Depende de:** T05
- **Esforço:** 2 h
- **Aceite:**
  - Heatmap (Pearson ou Spearman, justificado).
  - Lista de pares altamente correlacionados e decisão: manter, dropar ou reduzir dimensionalidade.
  - Discussão: correlação ≠ causalidade no contexto clínico.

## T08 — Split treino / validação / teste

- **Tipo:** obrigatório · ML
- **Depende de:** T06
- **Esforço:** 1 h
- **Aceite:**
  - Três conjuntos **disjuntos**. Proporção sugerida: 70/15/15 ou 60/20/20, **estratificado** pelo alvo.
  - Seed fixada.
  - Teste **intocado** até T12. Validação só para hiperparâmetros / comparação de modelos.
  - Documento curto: por que essa proporção e por que estratificar.
