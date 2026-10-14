# Atividade extra: CNN em radiografias de tórax

Este diretório contém a atividade **opcional** de visão computacional do Tech Challenge Fase 1. O estudo compara uma CNN simples treinada do zero com uma ResNet18 pré-treinada para classificar radiografias como **NORMAL** ou **PNEUMONIA**.

> Este é um experimento acadêmico de apoio à triagem. Não fornece diagnóstico e não substitui a avaliação de profissionais de saúde.

## Requisitos

- Docker Desktop atualizado, com Docker Compose v2 (`docker compose`).
- Acesso à internet para baixar o dataset do Kaggle e, na primeira execução da ResNet18, os pesos pré-treinados do PyTorch.
- Aproximadamente 1,2 GB para o dataset, além do espaço usado pela imagem Docker, checkpoints e resultados.

O fluxo documentado é CPU-only: não configura GPU NVIDIA/CUDA ou Apple MPS. A reprodução do treino dentro do container pode ser demorada.

## 1. Baixar e preparar o dataset

As radiografias não são armazenadas neste repositório: o conjunto tem cerca de 1,2 GB. Baixe-o diretamente do [Chest X-Ray Images (Pneumonia) no Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), usando sua conta e respeitando os termos da plataforma e do conjunto.

Extraia a pasta `chest_xray` para `extra-cnn-pneumonia/data/`. A estrutura deve ficar assim, sem uma segunda pasta `chest_xray` aninhada:

```text
data/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
```

O validador espera a versão do Kaggle usada no estudo e confere as contagens:

| Conjunto | NORMAL | PNEUMONIA |
|---|---:|---:|
| `train` | 1.341 | 3.875 |
| `val` | 8 | 8 |
| `test` | 234 | 390 |

As imagens são ignoradas pelo Git e pelo build do Docker; somente este README de dados fica versionado. O conjunto de validação original é pequeno; o pipeline separa uma validação maior, por paciente, a partir de `train`, e preserva `test` como a partição original.

## 2. Validar dados e construir a imagem

Se o terminal estiver na raiz do repositório, entre na pasta `extra-cnn-pneumonia`. Execute os comandos seguintes a partir dessa pasta:

```powershell
Set-Location .\extra-cnn-pneumonia
docker compose build
docker compose run --rm app python src/check_dataset.py
```

O validador deve exibir as contagens esperadas e terminar com código de saída `0`. Se o dataset estiver ausente, incompleto ou em uma pasta diferente, o comando informa o problema e falha. Corrija a estrutura antes de iniciar o Jupyter.

Os mesmos comandos funcionam em Bash a partir do diretório `extra-cnn-pneumonia`:

```bash
docker compose build
docker compose run --rm app python src/check_dataset.py
```

## 3. Iniciar o JupyterLab

```text
docker compose up
```

Quando o servidor iniciar, copie do log a URL local com o token gerado pelo Jupyter, normalmente começando por `http://127.0.0.1:8888/lab?token=...`, e abra-a no navegador. O serviço é publicado somente em `127.0.0.1`; não remova esse limite nem exponha a porta à internet.

Para encerrar, pressione `Ctrl+C` e execute:

```text
docker compose down
```

## 4. Explorar ou reproduzir o estudo

No JupyterLab, abra e execute os notebooks nesta ordem:

1. `01_exploracao.ipynb`
2. `02_split_preprocessamento.ipynb`
3. `03_cnn_simples.ipynb`
4. `04_resnet18.ipynb`
5. `05_avaliacao_final.ipynb`

Ou, em outro terminal na pasta `extra-cnn-pneumonia`, execute a sequência completa dentro do container:

```powershell
docker compose run --rm app jupyter nbconvert --to notebook --execute --inplace `
  notebooks/01_exploracao.ipynb `
  notebooks/02_split_preprocessamento.ipynb `
  notebooks/03_cnn_simples.ipynb `
  notebooks/04_resnet18.ipynb `
  notebooks/05_avaliacao_final.ipynb
```

Em Bash, troque os acentos graves de continuação por barras invertidas (`\`). O comando pode levar bastante tempo em CPU. Os notebooks são montados como volume para que suas saídas sejam gravadas no host; checkpoints `.pt` e figuras geradas também ficam nas pastas locais `models/` e `reports/`.

O clone não inclui checkpoints treinados. Portanto, para executar `05_avaliacao_final.ipynb`, execute primeiro os notebooks de treino `03_cnn_simples.ipynb` e `04_resnet18.ipynb`, que criam os pesos necessários.

## Solução de problemas

- **Docker não conecta ao engine:** inicie o Docker Desktop e aguarde até o engine Linux estar pronto; depois repita o comando.
- **Porta 8888 ocupada:** encerre o serviço que a utiliza ou altere a porta do host em `docker-compose.yml`, mantendo `8888` como porta do container (por exemplo, `127.0.0.1:8889:8888`).
- **Validador informa contagens diferentes:** confira se o conteúdo está em `data/chest_xray/` e se baixou a versão do Kaggle indicada acima; uma versão diferente exige revisar as contagens esperadas no código e neste guia.
- **Precisa recuperar o link/token do Jupyter:** consulte a saída do serviço com `docker compose logs app`.

## Observação sobre os resultados

O relatório do estudo compara CNN simples e ResNet18 usando resultados tanto de validação quanto de teste; a escolha da ResNet18 considera o equilíbrio observado nessa comparação. Assim, os valores do teste descrevem o experimento realizado e não devem ser interpretados como uma estimativa independente posterior à escolha do modelo. A reprodução registra o estudo como implementado, não como validação clínica.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/` | Dataset baixado localmente; não versionado |
| `notebooks/` | Exploração, preparação, treinamento e avaliação |
| `src/` | Código de dados, modelos, treino, Grad-CAM e validação do dataset |
| `models/` | Checkpoints de treinamento; não versionados |
| `reports/figures/` | Figuras do estudo |
| `reports/relatorio_tecnico.md` | Relatório técnico do experimento |
| `Dockerfile` | Imagem com dependências e JupyterLab |
| `docker-compose.yml` | Publicação local e montagem de volumes |
