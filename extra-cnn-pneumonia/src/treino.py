"""Funções para treinar e avaliar os modelos."""
import copy
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn


def escolher_dispositivo():
    """Usa a GPU do Mac (MPS) ou uma GPU NVIDIA (CUDA) se houver; senão, a CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def treinar_uma_epoca(modelo, loader, criterio, otimizador, dispositivo):
    """Passa por todas as imagens de treino uma vez, ajustando os pesos do modelo."""
    modelo.train()                                   # liga dropout e batch norm em modo treino
    perda_total, acertos, total = 0.0, 0, 0
    for imagens, rotulos in loader:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

        otimizador.zero_grad()                       # limpa os gradientes do lote anterior
        saidas = modelo(imagens)                     # previsão
        perda = criterio(saidas, rotulos)            # o quanto errou
        perda.backward()                             # calcula como ajustar cada peso
        otimizador.step()                            # ajusta os pesos

        perda_total += perda.item() * len(rotulos)
        acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += len(rotulos)
    return perda_total / total, acertos / total


@torch.no_grad()                                     # na avaliação não ajustamos pesos
def avaliar(modelo, loader, criterio, dispositivo):
    """Mede o desempenho em um conjunto. Devolve perda, rótulos reais e previsões."""
    modelo.eval()                                    # desliga dropout
    perda_total, total = 0.0, 0
    todos_rotulos, todas_previsoes = [], []
    for imagens, rotulos in loader:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
        saidas = modelo(imagens)
        perda_total += criterio(saidas, rotulos).item() * len(rotulos)
        total += len(rotulos)
        todos_rotulos += rotulos.cpu().tolist()
        todas_previsoes += saidas.argmax(dim=1).cpu().tolist()
    return perda_total / total, todos_rotulos, todas_previsoes


def calcular_metricas(rotulos, previsoes):
    """Métricas tendo PNEUMONIA (rótulo 1) como a classe que queremos encontrar."""
    return {
        "accuracy": accuracy_score(rotulos, previsoes),
        "recall": recall_score(rotulos, previsoes, pos_label=1),        # dos doentes, quantos achou
        "precisao": precision_score(rotulos, previsoes, pos_label=1),   # dos que disse "doente", quantos eram
        "f1": f1_score(rotulos, previsoes, pos_label=1),
        "recall_normal": recall_score(rotulos, previsoes, pos_label=0), # dos normais, quantos reconheceu
    }


def treinar(modelo, loaders, pesos_classe, epocas, taxa_aprendizado, caminho_pesos, dispositivo):
    """Treina o modelo por várias épocas e guarda os melhores pesos.

    A cada época: treina, mede na validação e imprime o resultado.
    Os pesos salvos são os da época com menor perda na validação; assim, se o
    modelo começar a decorar o treino (overfitting), ficamos com a melhor versão.
    Devolve uma tabela com o histórico, usada depois para desenhar as curvas.
    """
    modelo = modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_classe.to(dispositivo))   # pesos de classe entram aqui
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado)

    historico = []
    melhor_perda_val, melhores_pesos = float("inf"), None
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        perda_treino, acc_treino = treinar_uma_epoca(modelo, loaders["treino"], criterio, otimizador, dispositivo)
        perda_val, rotulos, previsoes = avaliar(modelo, loaders["val"], criterio, dispositivo)
        metricas_val = calcular_metricas(rotulos, previsoes)

        historico.append({
            "epoca": epoca,
            "perda_treino": perda_treino, "acc_treino": acc_treino,
            "perda_val": perda_val, "acc_val": metricas_val["accuracy"],
            "recall_val": metricas_val["recall"], "f1_val": metricas_val["f1"],
        })
        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            torch.save(melhores_pesos, caminho_pesos)
            marca = "  <- melhor até agora"
        else:
            marca = ""
        print(f"Época {epoca:2d}/{epocas} | treino: perda {perda_treino:.3f} acc {acc_treino:.3f} | "
              f"val: perda {perda_val:.3f} acc {metricas_val['accuracy']:.3f} "
              f"recall {metricas_val['recall']:.3f} | {time.time() - inicio:.0f}s{marca}")

    modelo.load_state_dict(melhores_pesos)           # volta para a melhor versão
    return pd.DataFrame(historico)
