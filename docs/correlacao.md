# T07 — Correlação

Método: **Pearson**. As 30 features são contínuas (mean/se/worst do núcleo). Não há sentinela tipo Insulin do Pima que justificasse Spearman como padrão. Spearman fica de reserva se a T05 tivesse apontado outliers extremos; aqui o describe da E1 não mostrou isso.

Figura: `reports/figures/02_correlacao.png` (heatmap sem anotação numérica — 30×30 fica ilegível). Pares na tabela abaixo.

## Limiar e top pares

Limiar de “alto”: **|r| > 0.90**.

| Par | \|r\| | Decisão |
|---|---|---|
| `radius_mean` × `perimeter_mean` | 0.998 | **Manter as duas** no baseline |
| `radius_worst` × `perimeter_worst` | 0.994 | Manter |
| `radius_mean` × `area_mean` | 0.987 | Manter |
| `perimeter_mean` × `area_mean` | 0.987 | Manter |
| `radius_worst` × `area_worst` | 0.984 | Manter |

## Decisão

Não dropar perímetro/área agora e não fazer PCA. Motivo: a árvore/RF (T10) tolera colinearidade; a logística (T09) usa regularização no pipeline e a discussão da banca cobre o custo (coeficientes instáveis), em vez de esconder o fato. Se a T09 sofrer de VIF alto, a revisão documenta dropar `perimeter_*` / `area_*` e **refaz o split (T08)** — não muda feature depois do teste lacrado.

`corrwith(y)` fica à parte (associação com o diagnóstico), não mistura com o heatmap de X.

## Correlação ≠ causalidade

`radius` e `perimeter` andam juntos porque descrevem o **mesmo núcleo** em unidades diferentes, não porque “diminuir o raio cura o câncer”. Nenhuma correlação desta tabela autoriza conduta clínica. O médico tem a palavra final.
