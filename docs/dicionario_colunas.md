# Dicionário de colunas — Breast Cancer Wisconsin (Diagnostic)

Fonte: medidas de **núcleo celular** em imagem digitalizada de punção aspirativa (FNA) de massa mamária (UCI).  
Alvo clínico: **maligno vs benigno**. O modelo não diagnostica; o médico decide.

Arquivo: `data/data.csv`. Na modelagem, **`id` sai de X**. `diagnosis` é só o alvo `y`.

## Colunas que não entram em X

| Nome técnico | Significado | Uso |
|---|---|---|
| `id` | Identificador do exame na base (não é prontuário) | Fora de X — vazaria identidade sem valor clínico |
| `diagnosis` | Rótulo do exame: `M` = maligno, `B` = benigno | Alvo `y`. Classe positiva de triagem = **maligno** (`1`) |

## Dez famílias de feature (× mean / se / worst)

Cada família descreve o contorno ou a textura do núcleo. Há três estatísticas por família:

- **mean** — média entre os núcleos medidos no exame
- **se** — erro-padrão (dispersão da medida)
- **worst** — média dos três maiores valores (o “pior” núcleo)

| Família | Colunas | O que mede (clínico, curto) |
|---|---|---|
| radius | `radius_mean`, `radius_se`, `radius_worst` | Distância média do centro à borda do núcleo (tamanho) |
| texture | `texture_mean`, `texture_se`, `texture_worst` | Desvio-padrão da escala de cinza (heterogeneidade visual) |
| perimeter | `perimeter_mean`, `perimeter_se`, `perimeter_worst` | Perímetro do núcleo |
| area | `area_mean`, `area_se`, `area_worst` | Área do núcleo |
| smoothness | `smoothness_mean`, `smoothness_se`, `smoothness_worst` | Variação local do raio (borda mais ou menos lisa) |
| compactness | `compactness_mean`, `compactness_se`, `compactness_worst` | `perímetro² / área − 1` — quão “encolhido”/irregular vs círculo |
| concavity | `concavity_mean`, `concavity_se`, `concavity_worst` | Gravidade dos trechos côncavos do contorno |
| concave points | `concave points_mean`, `concave points_se`, `concave points_worst` | Quantidade de trechos côncavos no contorno |
| symmetry | `symmetry_mean`, `symmetry_se`, `symmetry_worst` | Simetria do núcleo |
| fractal dimension | `fractal_dimension_mean`, `fractal_dimension_se`, `fractal_dimension_worst` | Irregularidade “tipo costa” do contorno (− 1 na definição UCI) |

Total em X: **30** colunas numéricas. Não há unidade física padronizada no CSV (pixels / unidades da imagem digitalizada).

## Mapeamento do alvo

| Fonte | Código | Classe clínica |
|---|---|---|
| CSV (`diagnosis`) | `M` | maligno (positivo de triagem) |
| CSV (`diagnosis`) | `B` | benigno |
| `sklearn.load_breast_cancer` | `target=0` | malignant |
| `sklearn.load_breast_cancer` | `target=1` | benign |

No código deste repo: `y = 1` se `M`, `y = 0` se `B`. Não usar o inteiro do sklearn sem conferir `target_names`.
