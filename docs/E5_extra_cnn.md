# Epic 5 — EXTRA (sobe nota)

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E5_extra_cnn`  
Tasks: T16  
Anterior: [E4 Avaliação, interpretação e ética](E4_avaliacao_interpretacao_etica.md) · Próximo: [E6 Entregáveis FIAP](E6_entregaveis_fiap.md)

Não trata T16 como substituto da classificação tabular.

Artefatos desta branch:

- T16: [cnn_dataset.md](cnn_dataset.md) · [cnn_extra.md](cnn_extra.md)
- Figuras: `reports/figures/06_cnn_*.png`
- Notebook: `notebooks/03_cnn.ipynb`
- Código/treino: `extra-cnn-pneumonia/`
- Deps isoladas: `requirements-cnn.txt`

---

## T16 — Diagnóstico por imagem com CNN

- **Tipo:** extra
- **Depende de:** T02 (pode andar em paralelo a T09–T15)
- **Esforço:** 8–12 h
- **Candidatos do enunciado:**
  - [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  - [CBIS-DDSM Breast Cancer](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data)
- **Aceite mínimo se for fazer:**
  - Split treino/val/teste (o dataset de pneumonia já vem separado — respeitar).
  - CNN (do zero **ou** transfer learning); accuracy/recall/F1 no teste.
  - Overfitting discutido (treino vs val).
  - Não substitui T01–T15: o tabular continua sendo a entrega principal.
