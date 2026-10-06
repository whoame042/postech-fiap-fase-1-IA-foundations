"""Carga, limpeza e divisão do ataset BCW (Diagnostic)."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from triagem_mama.config import(
    ALVO, 
    CAMINHO_DADOS,
    PROPORCAO_TESTE,
    PROPORCAO_VALIDACAO,
    SEED
)

#'id' identifica o exame; 'unnamed: 32' vem vazia. nenhuma é informação relevante
COLUNAS_DESCARTADAS = ["id", "Unnamed: 32"]


def carregar_dados_brutos(caminho=CAMINHO_DADOS) -> pd.DataFrame:
    """Lê CSV exatamente como foi baixado, sem nenhum tratamento."""
    return pd.read_csv(caminho)


def carregar_dados(caminho=CAMINHO_DADOS) -> pd.DataFrame:
    """Lê o CSV e aplica a limpeza decidida na EDA (notebook 01)."""
    return carregar_dados_brutos(caminho).drop(columns=COLUNAS_DESCARTADAS)


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as 30 medidas (X) do diagnóstico (y)."""
    return df.drop(columns=[ALVO]), df[ALVO]

@dataclass(frozen=True)
class Divisao:
    """Os três conjuntos disjuntos usados no projeto."""

    X_treino: pd.DataFrame
    X_val: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_val: pd.Series
    y_teste: pd.Series

def dividir_dados(X: pd.DataFrame, y: pd.Series) -> Divisao:
    """Divide em treino (70%), validação (15%) e teste (15%), estratificado por y.
    
    O train_test_split só divide em duas partes por vez, então são dois passos:
    1) separa o teste do restante;
    2) separa o treino e validação do que sobrou.
    A estratificação mantém a proporção benigno/maligno igual nos três conjuntos.
    """
    X_resto, X_teste, y_resto, y_teste = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=SEED
    )
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_resto, 
        y_resto,
        test_size=PROPORCAO_VALIDACAO / (1 - PROPORCAO_TESTE), 
        stratify=y_resto,
        random_state=SEED
    )
    return Divisao(X_treino, X_val, X_teste, y_treino, y_val, y_teste)