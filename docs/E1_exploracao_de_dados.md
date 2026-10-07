# Epic 1 — Exploração de dados

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E1_exploracao_de_dados`  
Tasks: T03, T04  
Anterior: [E0 Descoberta e setup](E0_descoberta_e_setup.md) · Próximo: [E2 Pré-processamento](E2_preprocessamento.md)

---

## T03 — Carregar a base e mapear características

- **Tipo:** obrigatório · dados
- **Depende de:** T02
- **Esforço:** 2 h
- **Aceite:**
  - Shape, dtypes, cardinalidade, amostra de linhas documentados no notebook.
  - Dicionário de colunas (nome técnico → significado clínico, mesmo que curto).
  - Identificador inútil (`id`) removido da modelagem, se existir.

## T04 — Estatísticas descritivas e visualizações

- **Tipo:** obrigatório · dados
- **Depende de:** T03
- **Esforço:** 3 h
- **Aceite:**
  - Describe numérico + distribuição do alvo.
  - Pelo menos: histograma/box-plot das features mais relevantes, contagem de classes, (opcional) pairplot/subset.
  - Texto discutindo o que as distribuições **implicam** para o diagnóstico (não só “o gráfico mostra X”).
  - Figuras salvas em `reports/figures/`.
