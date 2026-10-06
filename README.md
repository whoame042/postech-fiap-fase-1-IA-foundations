# Tech Challenge Fase 1 — Apoio à triagem de câncer de mama com Machine Learning

POSTECH FIAP • IA para Devs • Fase 1

### Storytelling:

Um hospital universitário quer acelerar a triagem de exames sem tirar do médico a decisão. Este projeto treina um modelo que, a partir de **30 medidas do núcleo celular** obtidas em uma punção aspirativa por agulha fina (PAAF), estima a **probabilidade de um nódulo de mama ser maligno**.

> **O modelo não faz diagnóstico.** Ele ajuda a ordenar a fila e a sinalizar casos que merecem atenção. **O médico tem sempre a palavra final.**

## Resultado em uma frase

O modelo escolhido, uma **Regressão Logística** com 24 medidas e limiar de decisão 0,33, encontrou **31 dos 32 tumores malignos** do conjunto de teste (recall 0,969), com **1 alarme falso** em 54 benignos. Os dois erros receberam probabilidades intermediárias (0,32 e 0,38) e cairiam na "faixa de atenção" proposta para revisão médica.

## Estrutura

O projeto segue o padrão do [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/): os **notebooks contam a análise**, e o código que se repete entre eles fica num pacote Python em `src/`.

| Caminho | Conteúdo |
|---|---|
| `notebooks/01-eda.ipynb` | Dataset, limpeza, estatísticas descritivas, distribuições e correlação |
| `notebooks/02-preprocessamento-e-modelagem.ipynb` | Split, pipeline, seleção de medidas, ajuste de hiperparâmetros, limiar e escolha do modelo na validação |
| `notebooks/03-avaliacao-e-interpretacao.ipynb` | Avaliação no teste (uma única vez), ROC/PR, feature importance, SHAP e discussão de uso prático |
| `src/triagem_mama/` | Código compartilhado: `config.py` (caminhos, semente), `dados.py` (carga, limpeza, split) e `modelagem.py` (pipelines, candidatos, métricas) |
| `data/raw/data.csv` | Dataset original, nunca alterado |
| `models/modelo_final.joblib` | Modelo final treinado + limiar (gerado pelo notebook 02) |
| `reports/figures/` | Todas as figuras geradas pelos notebooks |
| `docs/` | Enunciado do desafio |

Os notebooks devem ser executados **na ordem** (01 -> 02 -> 03): o 03 carrega o modelo salvo pelo 02.

## Como executar

Requer Python 3.12 ou superior (testado com 3.13).

### Localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # instala as dependências e o pacote do projeto (src/)
jupyter lab
```

Para executar os três notebooks de ponta a ponta, sem abrir o Jupyter (~4 min):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-eda.ipynb notebooks/02-preprocessamento-e-modelagem.ipynb notebooks/03-avaliacao-e-interpretacao.ipynb
```

### Com Docker

```bash
docker build -t tc-fase1 .
docker run --rm -p 8888:8888 tc-fase1
```

Acesse a URL com o token exibida nos logs (`http://127.0.0.1:8888/lab?token=...`).

Para executar todos os notebooks dentro do container e receber as figuras e o modelo na sua máquina:

```bash
docker run --rm -v "$(pwd)/reports:/app/reports" -v "$(pwd)/models:/app/models" tc-fase1 \
    jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01-eda.ipynb notebooks/02-preprocessamento-e-modelagem.ipynb notebooks/03-avaliacao-e-interpretacao.ipynb
```

Tudo usa `random_state=42`, então os resultados são reprodutíveis.

## Metodologia

1. **Divisão estratificada** em treino (397 exames, 70%), validação (86, 15%) e teste (86, 15%), com ~37% de malignos em cada.
2. **Pipeline** (`ColumnTransformer` + `StandardScaler` + classificador), ajustado **só no treino** para evitar vazamento de dados.
3. **Quatro candidatos:** Regressão Logística, Regressão Logística sem medidas redundantes (sem perímetro e área, que repetem o raio), Random Forest e KNN.
4. **Ajuste de hiperparâmetros** com `GridSearchCV` (validação cruzada de 5 folds no treino) e **limiar de decisão** com `TunedThresholdClassifierCV`, ambos pelo critério **F2**.
5. **Escolha na validação**, por regra definida antes de ver os resultados: maior recall, depois maior F2, depois maior F2 na validação cruzada.
6. **Teste usado uma única vez**, só para o modelo escolhido.

### Por que recall e F2?

Um **falso negativo** (câncer classificado como benigno) atrasa o tratamento. Um **falso positivo** gera um exame ou revisão a mais. Como o primeiro erro é muito mais grave:

- O **recall do maligno** (de todos os malignos, quantos o modelo encontrou) é a métrica principal;
- O **F2**, que combina recall e precisão dando **o dobro de peso ao recall**, é o critério para ajustar e escolher modelos;
- A **accuracy** sozinha engana: como 63% dos exames são benignos, um modelo que nunca encontrasse um maligno ainda teria 63% de acerto.

## Resultados

### Validação: escolha do modelo (86 exames, 32 malignos)

| Modelo | Limiar | Recall (M) | Precisão (M) | F2 (M) | Falsos neg. | Falsos pos. | F2 (CV treino) |
|---|---|---|---|---|---|---|---|
| **Regressão Logística (sem redundância)** | 0,33 | **1,000** | 0,941 | **0,988** | 0 | 2 | 0,957 |
| KNN | 0,20 | 1,000 | 0,941 | 0,988 | 0 | 2 | 0,909 |
| Regressão Logística | 0,34 | 1,000 | 0,865 | 0,970 | 0 | 5 | 0,965 |
| Random Forest | 0,48 | 0,938 | 0,968 | 0,943 | 2 | 1 | 0,944 |

A LR sem redundância empatou com o KNN em recall e F2 na validação e venceu no desempate pela validação cruzada (0,957 × 0,909). Também é o modelo mais fácil de interpretar.

### Teste: avaliação final (86 exames, 32 malignos)

| Limiar | Recall (M) | Precisão (M) | F1 (M) | Accuracy | Falsos neg. | Falsos pos. |
|---|---|---|---|---|---|---|
| **0,33 (escolhido)** | **0,969** | 0,969 | 0,969 | 0,977 | 1 | 1 |
| 0,50 (padrão, referência) | 0,969 | 1,000 | 0,984 | 0,988 | 1 | 0 |

- **AUC-ROC 0,999** e **AP 0,999**: o modelo quase sempre dá probabilidade maior aos malignos do que aos benignos.
- Neste teste, o limiar 0,33 não encontrou malignos a mais do que o 0,50: o único maligno perdido teve probabilidade 0,32. Na validação, porém, o limiar evitou um falso negativo. Com só 32 malignos por conjunto, **cada caso vale ~3 pontos de recall**, e essas diferenças devem ser lidas com cautela.

### No que o modelo se baseia

As medidas que mais empurram para maligno são o **tamanho do núcleo e sua variação** (`radius_se`, `radius_worst`), a **textura heterogênea** (`texture_worst`) e as **reentrâncias do contorno** (`concave points_*`), o que é coerente com a citologia. Algumas medidas de variação (`compactness_se`, `fractal_dimension_se`) têm peso negativo por compensação estatística entre medidas correlacionadas. O SHAP mostra que foi isso que levou ao único falso negativo, um sinal útil para o médico desconfiar do resultado.

### Limitações

569 exames de um único centro (Universidade de Wisconsin, anos 1990), medidos por um software específico; poucos malignos no teste; limiar escolhido por critério técnico, que na prática deve ser definido com a equipe clínica. Antes de qualquer uso real, seriam necessários validação externa, estudo prospectivo, monitoramento e adequação à ANVISA e à LGPD. A discussão completa está no notebook 03.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**: 569 exames, 30 medidas, diagnóstico benigno/maligno. Público, sem dados pessoais.

- Origem: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW28>. Licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Arquivo usado (`data/raw/data.csv`): versão do Kaggle, <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>.