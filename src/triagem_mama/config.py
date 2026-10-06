"""Configurações únicas do projeto: caminhos, seed e definição do alvo."""

from pathlib import Path

#src/triagem_mama/config.py -> sobe dois níveis até a raiz do projeto
#Requer instalação editável (pip install -e .), que é o que o requirements.txt faz.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = RAIZ_PROJETO / "data" / "raw" / "data.csv"
PASTA_FIGURAS = RAIZ_PROJETO / "reports" / "figures"
PASTA_MODELOS = RAIZ_PROJETO / "models"
CAMINHO_MODELO_FINAL = PASTA_MODELOS / "modelo_final.joblib"

#seed única para tudo que tem aleatorieade (split, modelos, validadção cruzada.)

SEED = 42

ALVO = "diagnosis"
CLASSE_POSITIVA = "M" #maligno: classe que não pode passar
CLASSE_NEGATIVA = "B" # benigno
CLASSES = [CLASSE_NEGATIVA, CLASSE_POSITIVA]
NOMES_CLASSES = {"B": "Benigno", "M": "Maligno"}
CORES_CLASSES = {"B": "#3B82A0", "M": "#D66A54"} #Mesma cor por classe para todos os gráficos

#Proporção do split (sobre o total de 569 amostras): 70% treino, 15% validação, 15% teste
PROPORCAO_TESTE = 0.15
PROPORCAO_VALIDACAO = 0.15