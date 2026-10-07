# T08 — Split treino / validação / teste

Proporção: **70 / 15 / 15**, estratificado por `y` (maligno=1). Seed **42**.

Implementação: dois `train_test_split` em `src/train.py` (`make_split`). Artefato: `models/split.joblib` (índices + seed). O split é **único** — T09/T10/T11 não sorteiam de novo.

## Por que 70/15/15

N = 569 é pequeno. Split efetivo: **397 / 86 / 86**. 15% de teste = 86 exames: dá para calcular F1/recall com intervalo largo — assumimos incerteza, não “número mágico”. 15% de validação (~85) serve para comparar modelos e hiperparâmetros sem tocar no teste. 70% (~399) ainda cabe em logística / árvore / KNN neste tabular.

60/20/20 deixaria o treino mais pobre (~341) sem ganho claro de estabilidade no teste. 80/10/10 deixaria o teste pequeno demais (~57) para a discussão da T12.

## Por que estratificar

A classe positiva (maligno) é 37,3%. Sem `stratify`, um teste pode sair quase só benigno e a accuracy mentir. Estratificar mantém a prevalência parecida nos três conjuntos.

## Lacre do teste

Ninguém usa `X_test` / `y_test` para métrica, gráfico de modelo ou `GridSearchCV` até a T12. Validação (`X_val`) é o único holdout para T09–T11. CV, se houver, só em `X_train`.
