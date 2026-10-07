# T05 — Limpeza

Dataset: `data/data.csv` (Wisconsin Diagnostic). Sem `StandardScaler` / `fillna` com estatística do conjunto inteiro.

## Inventário

| Coluna | Problema | Ação | Momento |
|---|---|---|---|
| `Unnamed: *` | Coluna vazia residual do CSV (Kaggle) | Dropar a coluna, não as linhas | Agora (`load_and_clean`) |
| `id` | Identificador do exame | Fora de X | Agora |
| `diagnosis` | `M`/`B` | Recodificar `M=1` (maligno), `B=0` | Agora |
| Features numéricas | NA | Nenhum NA neste CSV | Pipeline T06: `SimpleImputer(median)` se aparecer no treino |
| Features categóricas | — | Não há neste dataset | Pipeline T06: moda + OneHot (factory pronta) |
| Duplicatas de linha | Risco de vazar o mesmo exame | `drop_duplicates` | Agora |
| Raio / área / perímetro < 0 | Valor impossível | Checagem; neste CSV não ocorre | Agora (alerta) |
| Zeros sentinela (tipo Pima) | Não se aplica ao Wisconsin | Não recodificar 0→NA | — |

Imputação de mediana/moda **não** roda no dataset inteiro. Vai no `ColumnTransformer`, fit só no treino (T09).

## Antes / depois

| Momento | Shape |
|---|---|
| CSV bruto | (569, 33) — inclui `id` e coluna `Unnamed` vazia |
| Depois da limpeza (df com y) | (569, 31) — 30 features + `diagnosis` |
| X | (569, 30) |
| Duplicatas removidas | 0 |
| NA em X | 0 |
| Negativos em X | não |

`y` é `int` 0/1. Código: `src/preprocess.py` → `load_and_clean`.
