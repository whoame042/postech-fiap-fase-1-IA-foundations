# Epic 6 — Entregáveis FIAP

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E6_entregaveis_fiap`  
Tasks: T17, T18, T19, T20, T21  
Anterior: [E5 EXTRA CNN](E5_extra_cnn.md)

O PDF pede **um arquivo PDF** contendo os itens abaixo + **um vídeo**.

---

## T17 — Dockerfile + README de execução

- **Tipo:** obrigatório · entrega
- **Depende de:** T12 (código estável)
- **Esforço:** 3 h
- **Aceite:**
  - `Dockerfile` que instala deps e permite rodar o notebook **ou** `python src/train.py`.
  - README: pré-requisitos, build, run, onde estão métricas e figuras.
  - Link do repositório Git no PDF final.

## T18 — Dataset no repo ou link de download

- **Tipo:** obrigatório · entrega
- **Depende de:** T01
- **Esforço:** 30 min
- **Aceite:**
  - Arquivo no repo **ou** URL + comando de download + checksum/nome do arquivo.
  - Licença/citação da fonte no README.

## T19 — Relatório técnico (PDF)

- **Tipo:** obrigatório · entrega
- **Depende de:** T15, T17, T18, T20
- **Esforço:** 6 h
- **Seções mínimas (PDF):**
  1. Link do Git
  2. Estratégias de pré-processamento
  3. Modelos usados e **por quê**
  4. Resultados e interpretação (métricas, importance, SHAP)
  5. Uso prático + limite clínico
  6. (Se T16) capítulo EXTRA
- **Aceite:** PDF único, citando figuras de T20; sem código dump interminável (código vive no Git).

## T20 — Resultados visuais no repositório

- **Tipo:** obrigatório · entrega
- **Depende de:** T04, T07, T12, T13, T14
- **Esforço:** 2 h
- **Aceite:**
  - Pasta `reports/figures/` com EDA, correlação, matrizes de confusão, importance, SHAP.
  - Prints ou exportações que o PDF e o vídeo consigam reusar.

## T21 — Vídeo de demonstração (≤ 15 min)

- **Tipo:** obrigatório · entrega
- **Depende de:** T17, T19
- **Esforço:** 4 h
- **Aceite:**
  - Upload YouTube ou Vimeo (público ou não listado).
  - Mostra o sistema **em execução** (Docker ou notebook rodando).
  - Explica o fluxo: dados → pipeline → modelo → métrica → interpretação → “médico decide”.
  - Link no PDF. Duração ≤ 15 min.
