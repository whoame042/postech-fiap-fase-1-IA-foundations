# Tech Challenge Fase 1 — Triagem de câncer de mama com Machine Learning

Classificação tabular (benigno × maligno) no dataset público **Breast Cancer Wisconsin** (569 amostras, 30 features), como apoio à triagem clínica. **O modelo não faz diagnóstico: o médico tem sempre a palavra final.**

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/data.csv` | Dataset Breast Cancer Wisconsin (público) |
| `AnaliseExploratoria.ipynb` | EDA, limpeza, pipeline de pré-processamento, split e modelos (T03–T11) |
| `notebooks/02_avaliacao.ipynb` | Avaliação no teste: accuracy, recall e F1 (T12) |
| `notebooks/03_interpretabilidade.ipynb` | Feature importance e SHAP (T13, T14) |
| `notebooks/04_correlacao_e_discussao_clinica.ipynb` | Correlação, validação cruzada e discussão clínica (T07, T15) |
| `reports/figures/` | Gráficos gerados pelos notebooks |
| `requirements.txt` | Dependências com versões fixas |
| `Dockerfile` | Ambiente reproduzível |

## Como executar

Requer Python 3.12 ou superior (testado com 3.13).

### Localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Abra os notebooks na ordem acima. Todos usam `random_state=42` e são reproduzíveis.

### Com Docker

```bash
docker build -t tc-fase1 .
docker run --rm -p 8888:8888 tc-fase1
```

Acesse a URL com o token exibida nos logs (`http://127.0.0.1:8888/lab?token=...`). Para que os gráficos gerados fiquem na sua máquina, monte a pasta:

```bash
docker run --rm -p 8888:8888 -v "$(pwd)/reports:/app/reports" tc-fase1
```

## Resultados (conjunto de teste, 86 amostras, 32 malignas)

Classe positiva: **maligno**. O recall dela é a métrica principal, pois um falso negativo é o erro mais grave.

| Modelo | Accuracy | Recall | Precisão | F1 |
|---|---|---|---|---|
| Regressão Logística | 0,988 | 0,969 | 1,000 | 0,984 |
| Random Forest | 0,965 | 0,906 | 1,000 | 0,951 |
| KNN | 0,953 | 0,906 | 0,967 | 0,935 |

Com poucos casos malignos no teste, as diferenças entre modelos não são conclusivas; a validação cruzada e as limitações estão em `notebooks/04_correlacao_e_discussao_clinica.ipynb`.

---

# Backlog de tasks do desafio

Fonte: `POSTECH - Tech Challenge - Fase 1.pdf`
Curso: Pós Tech — IA para Devs
Peso: **90% da nota de todas as disciplinas da fase**
Formato: grupo · entrega obrigatória

Processo SDLC genérico (3 tamanhos): process_sdlc/_INDICE.md
Binding deste challenge: process_sdlc/aplicacoes/tech_challenge_fase1.md (recomendado: medium)
Playbooks operacionais: epics/_INDICE.md

## O que o desafio pede

Hospital universitário quer **triagem automática** de exames e documentos clínicos para apoiar o médico — não substituí-lo. Nesta fase a entrega mínima é **classificação tabular com Machine Learning** (doença sim/não). CNN em imagem é **EXTRA** (sobe nota se a parte obrigatória não fechar 100%).

**Restrição clínica (enunciado):** o médico sempre tem a palavra final no diagnóstico. Nenhum artefato pode vender o modelo como diagnóstico autônomo.

## Fora de escopo nesta fase

- Prontuário eletrônico completo, fila de atendimento, login de hospital
- LLM / RAG / GenAI (fases posteriores do curso)
- Deploy em nuvem de produção, monitoramento clínico real
- Usar dado real de paciente identificável (somente dataset **público**)
