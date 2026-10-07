# Problema clínico

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Origem:** UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic). Espelho operacional: [Kaggle — Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) e `sklearn.datasets.load_breast_cancer` (mesma origem, sem conta Kaggle).
- **Licença:** conjunto **público** para uso acadêmico (UCI). Sem PII de paciente real; `id` da tabela é identificador do exame, não dado assistencial.
- **Arquivo no repo:** `data/data.csv` (T18: versionado aqui).
- **Alvo y:** `diagnosis` — **maligno (M)** vs **benigno (B)**. Classe positiva para triagem = **maligno**. O modelo estima P(maligno | features do exame de núcleo de mama). **Não emite diagnóstico.**
- **Mapeamento (não inverter):** no CSV, `M` = maligno e `B` = benigno. No `sklearn.datasets.load_breast_cancer`, `target_names` é `['malignant', 'benign']` (`0` = malignant, `1` = benign).
- **Desbalanceamento:** sim.
- **Proporção (N=569, contagem em `data/data.csv`):** benigno (B) = 357 (62,74%) · maligno (M) = 212 (37,26%).
- **Papel do modelo:** apoio à **triagem** / segunda leitura — priorizar casos com alta P(maligno) para o médico revisar primeiro.
- **O médico tem a palavra final.**

## Problema (½ página)

Um hospital universitário recebe exames citológicos de mama (núcleo de tumor) descritos por medidas de núcleo celular — raio, textura, perímetro, área, concavidade, pontos côncavos, simetria e dimensão fractal, cada uma em média, erro-padrão e “pior” valor. A pergunta clínica é binária: o exame aponta para **maligno** ou **benigno**?

Hoje o médico faz isso com laudo, experiência e, quando preciso, exames complementares. O volume de casos e a variação entre observadores tornam útil um classificador que rode as mesmas 30 features numéricas e devolva uma probabilidade calibrada de malignidade, com interpretação (importance / SHAP) para o médico questionar o modelo.

ML entra porque o espaço de features é tabular, rotulado e público: dá para treinar, validar e testar sem leakage, e comparar pelo menos duas técnicas (T09–T11) com métricas que respeitam o custo clínico do falso negativo (deixar passar um maligno). Accuracy sozinha não basta: a classe maligna é minoria (37,26%).

O que o médico faria sem o modelo: ler o exame, correlacionar com clínica e imagem, e decidir conduta. O modelo **não** substitui essa decisão, **não** vê a paciente, **não** indica tratamento e **não** afirma causalidade biológica a partir de correlação ou SHAP. Serve para triagem e transparência do raciocínio estatístico.

**O médico tem a palavra final no diagnóstico.**
