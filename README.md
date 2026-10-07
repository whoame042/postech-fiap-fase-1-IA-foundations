# Tech Challenge Fase 1 — triagem tabular de câncer de mama

Pós Tech FIAP — IA para Devs. Peso: **90% da nota** da fase.

Hospital universitário quer **triagem automática** de exames clínicos para apoiar o médico. Entrega mínima: classificação tabular **maligno vs benigno** (Breast Cancer Wisconsin). CNN em imagem é **EXTRA** (`extra-cnn-pneumonia/`).

**O modelo estima P(maligno | features do exame). Não emite diagnóstico. O médico tem a palavra final.**

Detalhe clínico, origem, licença e desbalanceamento: [docs/problema_clinico.md](docs/problema_clinico.md).  
Kickoff (donos, EXTRA, prazo): [docs/kickoff.md](docs/kickoff.md).  
Backlog por épico: [docs/_tech_challenge_fase_1_backlog_de_tasks.md](docs/_tech_challenge_fase_1_backlog_de_tasks.md).

## Como rodar local

Python 3.11+ recomendado. Docker fica na T17.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import sklearn, pandas, shap"
jupyter notebook notebooks/01_eda.ipynb
```

Dataset tabular: `data/data.csv` (já no repo).

## Estrutura

```
.
  README.md
  requirements.txt
  data/data.csv
  notebooks/01_eda.ipynb
  notebooks/02_modelagem.ipynb
  src/preprocess.py
  src/train.py
  src/evaluate.py
  reports/figures/
  docs/
  extra-cnn-pneumonia/    # EXTRA (T16)
```

## Fora de escopo nesta fase

- Prontuário eletrônico, fila de atendimento, login de hospital
- LLM / RAG / GenAI
- Deploy em nuvem de produção
- Dado real de paciente identificável
