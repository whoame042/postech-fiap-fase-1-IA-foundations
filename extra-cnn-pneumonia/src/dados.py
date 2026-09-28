"""Carregamento e pré-processamento das radiografias.

Este arquivo reúne tudo o que os notebooks precisam para preparar os dados:
  1. listar as imagens e seus rótulos;
  2. criar um novo split treino/validação (sem misturar imagens do mesmo paciente);
  3. definir as transformações aplicadas em cada imagem;
  4. montar os DataLoaders que entregam as imagens ao modelo em lotes.
"""
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RAIZ = Path(__file__).resolve().parent.parent / "data" / "chest_xray"
CLASSES = ["NORMAL", "PNEUMONIA"]          # a posição na lista é o rótulo: NORMAL = 0, PNEUMONIA = 1
EXTENSOES = {".jpeg", ".jpg", ".png"}
TAMANHO_IMAGEM = 224                        # tamanho de entrada da ResNet18 (224 x 224 pixels)

# Média e desvio padrão do ImageNet. A ResNet18 pré-treinada foi treinada com imagens
# normalizadas assim, então usamos os mesmos valores para ela "enxergar" como esperado.
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Listar as imagens
# ---------------------------------------------------------------------------
def id_paciente(nome_arquivo):
    """Extrai o identificador do paciente a partir do nome do arquivo.

    Exemplos: 'person1000_virus_1681.jpeg' -> 'person1000'
              'IM-0115-0001.jpeg'           -> 'IM-0115'
    """
    if nome_arquivo.startswith("person"):
        return nome_arquivo.split("_")[0]
    return "IM-" + re.search(r"IM-(\d+)", nome_arquivo).group(1)


def listar_imagens(conjunto):
    """Devolve um DataFrame com uma linha por imagem de 'train', 'val' ou 'test'."""
    linhas = []
    for rotulo, classe in enumerate(CLASSES):
        for arquivo in sorted((RAIZ / conjunto / classe).iterdir()):
            if arquivo.suffix.lower() in EXTENSOES:
                linhas.append({
                    "caminho": arquivo,
                    "classe": classe,
                    "rotulo": rotulo,
                    "paciente": id_paciente(arquivo.name),
                })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 2. Novo split treino / validação
# ---------------------------------------------------------------------------
def dividir_treino_validacao(df_treino, semente=42):
    """Separa cerca de 20% do treino original para servir de validação.

    Um mesmo paciente aparece em várias imagens (até 30). Se algumas ficassem no
    treino e outras na validação, o modelo "reconheceria" o paciente e a validação
    pareceria melhor do que realmente é. Por isso a divisão é feita por paciente:
    todas as imagens de um paciente vão para o mesmo lado.

    O StratifiedGroupKFold divide em 5 partes (grupos = pacientes) mantendo a
    proporção de NORMAL/PNEUMONIA parecida. Usamos uma parte (20%) como validação.
    """
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
    indices_treino, indices_val = next(
        divisor.split(df_treino, df_treino["rotulo"], groups=df_treino["paciente"])
    )
    return df_treino.iloc[indices_treino].reset_index(drop=True), \
           df_treino.iloc[indices_val].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Transformações
# ---------------------------------------------------------------------------
def transformacoes_avaliacao():
    """Usada em validação e teste: só deixa a imagem no formato que o modelo espera."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),           # cinza copiado em 3 canais (a ResNet espera RGB)
        transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),   # tamanho único para todas
        transforms.ToTensor(),                                 # imagem -> tensor com valores entre 0 e 1
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def transformacoes_treino():
    """Usada no treino: as mesmas etapas, mais variações aleatórias (data augmentation).

    A cada época a mesma imagem sai um pouco diferente. Isso ajuda o modelo a não
    decorar as imagens de treino (overfitting). Usamos variações pequenas e realistas.
    Não giramos a imagem de lado (flip horizontal), porque o coração fica no lado
    esquerdo do paciente e isso não ocorre em uma radiografia real espelhada.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
        transforms.RandomRotation(degrees=10),                                   # gira até 10 graus
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # desloca e amplia um pouco
        transforms.ColorJitter(brightness=0.2, contrast=0.2),                    # varia brilho e contraste
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


# ---------------------------------------------------------------------------
# 4. Dataset e DataLoaders
# ---------------------------------------------------------------------------
class RadiografiasDataset(Dataset):
    """Entrega (imagem transformada, rótulo) para cada posição da tabela."""

    def __init__(self, df, transformacao):
        self.df = df
        self.transformacao = transformacao

    def __len__(self):
        return len(self.df)

    def __getitem__(self, posicao):
        linha = self.df.iloc[posicao]
        imagem = Image.open(linha["caminho"])
        return self.transformacao(imagem), int(linha["rotulo"])


def criar_dataloaders(tamanho_lote=32, semente=42, num_workers=0):
    """Monta os DataLoaders de treino, validação e teste.

    Devolve (loaders, tabelas), dois dicionários com as chaves 'treino', 'val' e 'teste'.
    O conjunto de teste é o original do Kaggle e não é usado para nenhuma decisão.
    """
    df_treino, df_val = dividir_treino_validacao(listar_imagens("train"), semente)
    tabelas = {"treino": df_treino, "val": df_val, "teste": listar_imagens("test")}

    transformacoes = {
        "treino": transformacoes_treino(),
        "val": transformacoes_avaliacao(),
        "teste": transformacoes_avaliacao(),
    }
    gerador = torch.Generator().manual_seed(semente)   # embaralhamento reproduzível
    loaders = {
        nome: DataLoader(
            RadiografiasDataset(tabela, transformacoes[nome]),
            batch_size=tamanho_lote,
            shuffle=(nome == "treino"),                # só o treino é embaralhado
            generator=gerador if nome == "treino" else None,
            num_workers=num_workers,
        )
        for nome, tabela in tabelas.items()
    }
    return loaders, tabelas


def pesos_de_classe(df_treino):
    """Peso maior para a classe menos frequente, para compensar o desbalanceamento.

    Fórmula: total / (número de classes * quantidade da classe). Assim, errar uma
    imagem NORMAL (a classe rara) "custa" mais na hora de treinar.
    """
    contagem = df_treino["rotulo"].value_counts().sort_index()
    pesos = len(df_treino) / (len(CLASSES) * contagem)
    return torch.tensor(pesos.values, dtype=torch.float32)
