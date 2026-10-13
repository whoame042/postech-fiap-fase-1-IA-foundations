"""Grad-CAM: mostra em quais regiões da imagem o modelo mais "olhou" para decidir.

Ideia resumida: as últimas camadas convolucionais de uma CNN ainda guardam a
localização espacial (onde, na imagem, cada padrão foi encontrado), mesmo que já
tenham perdido detalhe fino. O Grad-CAM usa o gradiente da previsão em relação a
essa camada para descobrir quais regiões pesaram mais na decisão, e desenha isso
como um "mapa de calor" sobre a imagem original.

Isto substitui, em um modelo de imagem, o papel que SHAP/feature importance têm
em um modelo tabular: os dois respondem "por que o modelo decidiu isso?".
"""
import numpy as np
import torch
from PIL import Image


class GradCAM:
    """Calcula o mapa de calor de uma camada convolucional escolhida."""

    def __init__(self, modelo, camada_alvo):
        self.modelo = modelo
        self.ativacoes = None   # saída da camada durante o forward (o que ela "viu")
        self.gradientes = None  # gradiente que chega nessa camada durante o backward (o quanto ela importou)

        # Hooks são funções que o PyTorch chama automaticamente sempre que essa
        # camada processa dados (forward) ou recebe gradiente (backward).
        camada_alvo.register_forward_hook(self._guardar_ativacoes)
        camada_alvo.register_full_backward_hook(self._guardar_gradientes)

    def _guardar_ativacoes(self, modulo, entrada, saida):
        self.ativacoes = saida.detach()

    def _guardar_gradientes(self, modulo, grad_entrada, grad_saida):
        self.gradientes = grad_saida[0].detach()

    def gerar_mapa(self, imagem_tensor, classe_alvo, dispositivo):
        """Devolve um mapa 2D (valores de 0 a 1) do tamanho da camada escolhida.

        imagem_tensor: uma única imagem já transformada (sem a dimensão de lote).
        classe_alvo: 0 (NORMAL) ou 1 (PNEUMONIA) — de qual classe queremos o "porquê".
        """
        self.modelo.eval()
        entrada = imagem_tensor.unsqueeze(0).to(dispositivo)   # adiciona a dimensão de lote (1 imagem)
        # Na ResNet18 as camadas convolucionais estão congeladas (requires_grad=False).
        # Sem isto, o PyTorch nem calcularia o gradiente até a camada-alvo, porque
        # nenhum peso no caminho pediria gradiente. Pedir gradiente da própria
        # entrada força o cálculo até lá, sem mudar nenhum peso do modelo.
        entrada.requires_grad_(True)

        saida = self.modelo(entrada)
        self.modelo.zero_grad()
        saida[0, classe_alvo].backward()   # calcula o gradiente só em relação a essa classe

        # Para cada um dos mapas de características, resume o gradiente em um único
        # número (a "importância" daquele mapa para a decisão) e faz a média
        # ponderada das ativações por essa importância.
        importancia = self.gradientes.mean(dim=(2, 3), keepdim=True)
        mapa = (importancia * self.ativacoes).sum(dim=1).squeeze(0)

        mapa = torch.relu(mapa)                    # só nos interessa o que empurrou a favor da classe
        mapa = mapa / (mapa.max() + 1e-8)           # normaliza para ficar entre 0 e 1
        return mapa.cpu().numpy()


def redimensionar_mapa(mapa, tamanho):
    """O mapa sai pequeno (ex.: 14x14 ou 7x7). Amplia para o tamanho da imagem original."""
    imagem_mapa = Image.fromarray((mapa * 255).astype(np.uint8))
    imagem_mapa = imagem_mapa.resize(tamanho, Image.BICUBIC)
    return np.asarray(imagem_mapa) / 255.0
