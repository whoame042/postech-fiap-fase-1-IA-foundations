"""Modelos de classificação de radiografias."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def bloco_convolucional(canais_entrada, canais_saida):
    """Um bloco básico de CNN: convolução -> normalização -> ReLU -> redução da imagem."""
    return nn.Sequential(
        nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1),  # detecta padrões (bordas, texturas)
        nn.BatchNorm2d(canais_saida),                                        # estabiliza o treino
        nn.ReLU(),                                                           # zera valores negativos
        nn.MaxPool2d(2),                                                     # metade da altura e da largura
    )


class CNNSimples(nn.Module):
    """CNN pequena treinada do zero, usada como baseline (ponto de comparação).

    A imagem passa por 4 blocos convolucionais. A cada bloco ela fica menor
    (224 -> 112 -> 56 -> 28 -> 14) e com mais "mapas de características"
    (16 -> 32 -> 64 -> 128). No fim, os mapas são resumidos em 128 números e uma
    camada linear decide entre NORMAL e PNEUMONIA.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.extrator = nn.Sequential(
            bloco_convolucional(3, 16),
            bloco_convolucional(16, 32),
            bloco_convolucional(32, 64),
            bloco_convolucional(64, 128),
            nn.AdaptiveAvgPool2d(1),   # média de cada mapa -> 128 números
            nn.Flatten(),
        )
        self.classificador = nn.Sequential(
            nn.Dropout(0.5),           # desliga metade dos neurônios ao acaso no treino (evita overfitting)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classificador(self.extrator(x))


def resnet18_transferencia(num_classes=2, congelar_extrator=True):
    """ResNet18 pré-treinada no ImageNet (1,2 milhão de fotos comuns), adaptada
    para NORMAL/PNEUMONIA. Isso é *transfer learning*: em vez de aprender do zero
    a reconhecer bordas, texturas e formas, reaproveitamos o que a rede já
    aprendeu com essas fotos e ensinamos só a parte final a decidir entre as
    nossas duas classes.

    congelar_extrator=True: as camadas convolucionais (o "extrator de padrões")
    ficam travadas, com os pesos do ImageNet, e só a camada final é treinada.
    Isso deixa o treino bem mais rápido e evita estragar o que já foi aprendido,
    já que temos poucas imagens (milhares, não milhões) comparado ao ImageNet.
    """
    modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if congelar_extrator:
        for parametro in modelo.parameters():
            parametro.requires_grad = False   # "não ajuste este peso durante o treino"

    # A camada final original tem 1000 saídas (as 1000 classes do ImageNet).
    # Trocamos por uma camada nova, com 2 saídas (NORMAL, PNEUMONIA) e pesos
    # aleatórios: essa é a única parte que vamos treinar.
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    return modelo
