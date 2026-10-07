# T14 — SHAP

Modelo vencedor: logística. Conjunto: **teste** (holdout).  
Figuras: `reports/figures/05_shap_summary.png` (global) e `05_shap_waterfall_caso.png` (um exame).

## Caso (índice `190` no CSV)

| Campo | Valor |
|---|---|
| `row_idx` | 190 |
| y real | **maligno** (1) |
| y predito | **benigno** (0) |
| Tipo | falso negativo |

Este exame **é maligno** e o modelo classificou **benigno**. No waterfall, features de tamanho/textura (a mesma família da T13: `radius_*`, `texture_worst`, `area_*`) empurram o log-odds; o saldo linear ainda ficou do lado benigno. Em linguagem de médico: “o escore tabular desta lâmina não cruzou o limiar de malignidade, apesar do rótulo real ser maligno — o caso está na zona de overlap da EDA. Eu ainda olho a imagem, a clínica e, se preciso, peço exame complementar. O modelo não fecha o laudo.”

## Limitação

SHAP atribui crédito às features **deste modelo**, nestes 30 números de núcleo. Não é evidência causal, não é biologia, não é laudo. O dataset é público e antigo; não há sociodemografia nem a imagem que o patologista vê. Explicar o modelo ≠ explicar a doença.
