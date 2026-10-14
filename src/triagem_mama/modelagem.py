"""Modelagem dos pipelines, candidatos a modelo e métricas usadas no projeto."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from triagem_mama.config import CLASSE_POSITIVA, CLASSES, SEED

#F2: Combina recall e precisão dando ao recall o dobro do peso. Usamos F2 porque deixar passar um maligno (falso negativo) é pior que pedir ume xame a mais para um benigno (falso positivo);
scorer_f2 = make_scorer(fbeta_score, beta=2, pos_label=CLASSE_POSITIVA)
scorer_recall = make_scorer(recall_score, pos_label=CLASSE_POSITIVA)


def criar_cv(n_splits: int = 5) -> StratifiedKFold:
    """Validação cruzada estratificada e reprodutível."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def colunas_sem_redundancia(colunas) -> list[str]:
    """Remove perímetro e área, mantendo o raio.
    Raio, perímetro e área medem o tamanho do núcleo e têm correlação > 0,96 
    entre si (notebook 01).Ficar só com o raio testa se a redundância atrapalha a 
    Regressão Logística.
    """
    return [c for c in colunas if not c.startswith(("perimeter_", "area_"))]


def criar_pipeline(classificador, colunas) -> Pipeline:
    """Pré-processamento + classificador num único objeto.
    
    O ColumnTransformer seleciona as colunas usadas e padroniza cada uma 
    (média 0, desvio 1).Como tudo está dentro do Pipeline, o scaler só 
    aprende média e desvio com os dados passados no fit(), 
    o que evita vazamento de informação da validação e do teste.
    """
    preprocessador = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(colunas))],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return Pipeline(
        steps=[("preprocessador", preprocessador), ("classificador", classificador)]
    )

def criar_candidatos(colunas) -> dict:
    """Modelos candidatos e as grades de hiperparâmetros testadas para cada um.

    Retorna {nome: (pipeline, grade)}. Os nomes da grade usam o prefixo
    'classificador__' porque o hiperparâmetro pertence a essa etapa do pipeline.
    """
    colunas = list(colunas)
    grade_lr = {
        #C controla a regrularização: valores menores = pesos mais contidos.
        "classificador__C": [0.01, 0.1, 1, 10, 100],
        # 'balanced' dá mais peso à classe minoritária (maligno) no treino.
        "classificador__class_weight": [None, "balanced"]
    }
    return {
        "Regressão Logística": (
            criar_pipeline(LogisticRegression(max_iter=5000, random_state=SEED), colunas),
            grade_lr
        ),
        "Regressão Logística (sem redundância)": (
            criar_pipeline(
                LogisticRegression(max_iter=5000, random_state=SEED), 
                colunas_sem_redundancia(colunas)
                ),
            grade_lr
        ),
        "Random Forest":(
            criar_pipeline(
                RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                colunas 
            ),
            {
                "classificador__max_depth": [None, 5, 10],
                "classificador__min_samples_leaf": [1, 3, 5],
                "classificador__class_weight": [None, "balanced"]
            },
        ),
        "KNN": (
            criar_pipeline(KNeighborsClassifier(), colunas),
            {
                "classificador__n_neighbors": [3, 5, 7, 9, 11, 15],
                "classificador__weights": ["uniform", "distance"]    
            },
        ),
    }

def calcular_metricas(y_real, y_previsto) -> dict:
    """Métricas do projeto, sempre com maligno (M) como classe 'positiva'."""
    vn, fp, fn, vp = confusion_matrix(y_real, y_previsto, labels=CLASSES).ravel()
    return {
        "Recall (M)": recall_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA),
        "Precisão (M)": precision_score(
            y_real, y_previsto, pos_label=CLASSE_POSITIVA, zero_division=0
        ),
        "F1 (M)": f1_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA),
        "F2 (M)": fbeta_score(y_real, y_previsto, beta=2, pos_label=CLASSE_POSITIVA),
        "Accuracy": accuracy_score(y_real, y_previsto),
        "Falsos negativos": int(fn),
        "Falsos positivos": int(fp),
    }