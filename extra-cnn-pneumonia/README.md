# Tech Challenge Fase 1 — EXTRA: CNN em radiografias de tórax

Parte **EXTRA** (visão computacional) do Tech Challenge Fase 1 (FIAP Pós Tech, IA para Devs). Classifica radiografias de tórax em **NORMAL** ou **PNEUMONIA** com uma rede neural convolucional (CNN).

A entrega principal (classificação tabular de câncer de mama) está na raiz deste repositório.

**Este modelo é um exercício acadêmico de apoio à triagem. Ele não faz diagnóstico: o médico tem sempre a palavra final.**

## Dataset

[Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), cerca de 5.800 imagens. Não é versionado neste repositório por causa do tamanho.

1. Entre no Kaggle com a sua conta, abra a página do dataset e clique em **Download**.
2. Extraia o conteúdo em `data/`, de modo que fique `data/chest_xray/train`, `val` e `test`.
3. Confira com:

```bash
python src/check_dataset.py
```

## Como executar

Requer Python 3.12 ou superior (testado com 3.13). Em Mac com Apple Silicon, o treino usa a GPU (MPS) automaticamente.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/check_dataset.py
jupyter lab
```

### Executar com Docker

O Docker aqui serve para **explorar os notebooks já treinados** em qualquer máquina, sem instalar Python. Ele **não usa a GPU** (nem MPS do Mac, nem CUDA), então treinar do zero dentro do container é bem mais lento — para treinar, prefira a opção acima, com `.venv` direto na máquina.

```bash
docker build -t cnn-extra .

# -v monta suas pastas locais dentro do container, para ele enxergar o
# dataset e salvar/ler os pesos treinados
docker run -p 8888:8888 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  cnn-extra
```

Depois abra `http://localhost:8888` no navegador (sem senha nem token — é só para uso local).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/` | Dataset (não versionado) |
| `notebooks/` | Notebooks de exploração, treino e avaliação |
| `src/` | Código reutilizável (verificação do dataset e, depois, treino) |
| `models/` | Pesos treinados (não versionados) |
| `reports/figures/` | Gráficos e resultados |
| `reports/relatorio_tecnico.md` | Relatório técnico: pré-processamento, modelos e resultados |
| `requirements.txt` | Dependências com versões fixas |
| `Dockerfile` | Imagem para rodar os notebooks sem instalar Python |

## Plano

1. Explorar as imagens e as classes (a base é desbalanceada: há mais pneumonia que normal)
2. Reorganizar o split: o `val` original tem só 16 imagens, então um novo split treino/validação é feito a partir do `train`, mantendo o `test` intocado
3. CNN com transfer learning (ResNet18 pré-treinada) e comparação com uma CNN simples
4. Avaliação no teste: accuracy, recall, precisão, F1 e matriz de confusão (o recall da pneumonia é a métrica principal)
5. Discussão de overfitting (curvas de treino e validação) e do uso prático

## Status

Em andamento.

- [x] Dataset baixado e conferido
- [x] Exploração dos dados (`notebooks/01_exploracao.ipynb`)
- [x] Novo split treino/validação por paciente e pipeline de pré-processamento (`notebooks/02_split_preprocessamento.ipynb`, código em `src/dados.py`)
- [x] CNN simples (baseline) (`notebooks/03_cnn_simples.ipynb`)
- [x] Transfer learning com ResNet18 (`notebooks/04_resnet18.ipynb`)
- [x] Avaliação no teste e interpretação com Grad-CAM (`notebooks/05_avaliacao_final.ipynb`) — modelo final: ResNet18
- [x] Dockerfile (`Dockerfile`, testado com `docker build` + `docker run`)
- [x] Relatório técnico (`reports/relatorio_tecnico.md`)

Falta apenas o vídeo de demonstração (até 15 min) para fechar os entregáveis pedidos no enunciado.
