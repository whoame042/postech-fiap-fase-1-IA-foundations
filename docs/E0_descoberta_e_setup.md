# Epic 0 — Descoberta e setup

Índice: [_tech_challenge_fase_1_backlog_de_tasks.md](_tech_challenge_fase_1_backlog_de_tasks.md)  
Branch: `epic/E0_descoberta_e_setup`  
Tasks: T00, T01, T02  
Próximo: [E1 Exploração de dados](E1_exploracao_de_dados.md)

Artefatos desta branch:

- T00: [kickoff.md](kickoff.md) · [prazo.md](prazo.md)
- T01: [problema_clinico.md](problema_clinico.md)
- T02: `README.md`, `requirements.txt`, `.gitignore`, `src/`, `notebooks/`

---

## T00 — Kickoff do challenge

- **Tipo:** obrigatório · setup
- **Depende de:** —
- **Esforço:** 30 min
- **Fazer:** alinhar grupo, prazo FIAP, peso 90%, dono de cada epic, dataset candidato.
- **Aceite:**
  - Grupo tem um dono por epic (dados, modelagem, relatório, vídeo, Docker).
  - Prazo de entrega anotado.
  - Decisão: EXTRA CNN entra ou fica de reserva se a nota obrigatória não fechar.

## T01 — Escolher dataset e formular o problema clínico

- **Tipo:** obrigatório · produto
- **Depende de:** T00
- **Esforço:** 2 h
- **Candidatos do enunciado:**
  - [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) — maligno vs benigno (**recomendado:** tabular limpo, features clínicas, SHAP funciona bem)
  - [Diabetes](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
  - Outro público, desde que seja **tabela** + **classificação binária/multiclasse de diagnóstico**
- **Aceite:**
  - Dataset escolhido, licença/origem citadas, alvo (`y`) nomeado em linguagem clínica (ex.: “maligno vs benigno”).
  - Parágrafo de ½ página: problema, por que ML, o que o médico faria sem o modelo, o que o modelo **não** substitui.
  - Desbalanceamento de classes identificado (sim/não + proporção).

## T02 — Scaffold do repositório Python

- **Tipo:** obrigatório · engenharia
- **Depende de:** T01
- **Esforço:** 2 h
- **Estrutura sugerida:**
  ```
  tech-challenge-fase1/
    README.md
    Dockerfile
    requirements.txt
    data/           # ou script de download
    notebooks/01_eda.ipynb
    notebooks/02_modelagem.ipynb
    src/preprocess.py
    src/train.py
    src/evaluate.py
    reports/figures/
  ```
- **Aceite:**
  - `requirements.txt` pinado (pandas, scikit-learn, matplotlib, seaborn, shap, jupyter).
  - README com “como rodar local” (mesmo que o Docker venha em T17).
  - `.gitignore` (`.venv`, `__pycache__`, dados grandes se for o caso).
