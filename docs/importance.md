# T13 — Importância do vencedor (logística)

Modelo: logística (`C=1`), coeficientes na escala do `StandardScaler` (comparáveis).  
Gráfico: `reports/figures/04_importance.png`. Nomes = features do `ColumnTransformer` (prefixo `num__` removido).

## Top features

| Feature | Coef. | Leitura clínica | Ressalva |
|---|---|---|---|
| `radius_se` | +1.41 | Maior variação de raio entre núcleos do exame empurra para maligno (heterogeneidade de tamanho) | Não prova mecanismo biológico |
| `texture_worst` | +1.24 | Textura mais irregular nos núcleos “piores” alinha com atipia visual | Coeficiente ≠ causa |
| `radius_worst` | +0.96 | Núcleos maiores no pior caso — consistente com a EDA (maligno tende a ser maior) | Colinear com perímetro/área (T07) |
| `area_se` | +0.92 | Dispersão de área entre núcleos | Mesma família de tamanho |
| `area_worst` | +0.91 | Área do pior núcleo | Idem |

As três primeiras batem com o dicionário da T03 (tamanho e textura do núcleo). `worst concave points` não liderou o |coef| desta logística, embora a EDA o destacasse — o modelo linear reparte crédito com features colineares. SHAP (T14) conta a história por exame, não substitui esta tabela.

**Nenhuma destas features causa câncer.** O coeficiente diz o quanto o *modelo* empurra o log-odds.
