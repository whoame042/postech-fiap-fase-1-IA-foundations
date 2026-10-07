# T16 — Dataset EXTRA (Chest X-Ray Pneumonia)

Fonte: [Kaggle — Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.). ~5.8k radiografias pediátricas, classes **NORMAL** / **PNEUMONIA**.

**Não substitui** T01–T15. A entrega obrigatória continua sendo o tabular de mama na raiz.

## Download (não versionar imagens)

O zip (~1,2 GB) **não entra no Git**. Extrair para `extra-cnn-pneumonia/data/chest_xray/`:

```
extra-cnn-pneumonia/data/chest_xray/
  train/NORMAL  train/PNEUMONIA
  val/NORMAL    val/PNEUMONIA
  test/NORMAL   test/PNEUMONIA
```

```bash
# na pasta extra-cnn-pneumonia/
# 1) baixar o zip no Kaggle (UI ou: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia)
# 2) unzip para data/ de modo que exista data/chest_xray/train
python src/check_dataset.py
```

## Split

O autor já entrega `train` / `val` / `test`. **O `test` original não foi misturado nem usado como early stopping.**

O `val` oficial tem só 16 imagens — insuficiente. Por isso 20% do **train** virou validação **por paciente** (mesmo `person*` não cai nos dois lados). Detalhe: `extra-cnn-pneumonia/reports/relatorio_tecnico.md` §3.1 e `src/dados.py`.

| Conjunto | Imagens | Papel |
|---|---|---|
| treino | 4.172 | fit + augment |
| validação | 1.044 | early stopping / escolha de modelo |
| teste (original) | 624 | **uma vez**, no final |
