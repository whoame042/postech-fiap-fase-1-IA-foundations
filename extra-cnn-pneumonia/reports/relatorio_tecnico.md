# Relatório técnico — Classificação de pneumonia em radiografias de tórax

Tech Challenge Fase 1 (FIAP Pós Tech, IA para Devs) — parte **EXTRA** (visão computacional).

## 1. Contexto e objetivo

O desafio pede uma base de sistema de apoio ao diagnóstico para um hospital. A entrega principal (dados em tabela) está em outro repositório; esta parte extra faz a mesma ideia com **imagens médicas**: dizer se uma radiografia de tórax mostra um pulmão **NORMAL** ou com **PNEUMONIA**, usando uma rede neural convolucional (CNN — o tipo de rede neural feito para reconhecer padrões em imagens).

Este modelo é um exercício acadêmico de apoio à triagem. Ele não substitui o diagnóstico médico, como discutimos na Seção 6.

## 2. Dataset

[Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia): 5.856 radiografias de crianças pequenas (1 a 5 anos), já divididas pelo autor original em `train` (treino), `val` (validação) e `test` (teste).

A exploração inicial (`notebooks/01_exploracao.ipynb`) mostrou três pontos que guiaram todas as decisões seguintes:

1. **Classes desbalanceadas no treino:** 1.341 imagens NORMAL (25,7%) contra 3.875 PNEUMONIA (74,3%). Um modelo que respondesse sempre "pneumonia" já acertaria 74% das vezes, sem aprender nada de verdade.
2. **`val` original pequeno demais:** só 16 imagens (8 de cada classe) — não dá para confiar em nenhum resultado calculado com tão pouco dado.
3. **Tamanhos e formatos variados:** largura entre 384 e 2.916 pixels, altura entre 127 e 2.713; a maioria em tons de cinza (5.573 imagens), mas 283 coloridas (RGB).

O brilho médio das duas classes é parecido (123,0 para NORMAL, 125,3 para PNEUMONIA), então o brilho sozinho não seria um "atalho" fácil para o modelo diferenciar as classes.

## 3. Estratégias de pré-processamento

Implementado em `src/dados.py` e mostrado em `notebooks/02_split_preprocessamento.ipynb`.

### 3.1 Nova divisão treino/validação, por paciente

Como o `val` original é pequeno demais, separamos 20% do `train` para servir de validação nova. O `test` original não foi tocado em nenhuma etapa e só foi olhado uma única vez, no final (Seção 5.2).

Um cuidado importante aqui: os nomes dos arquivos revelam qual paciente é qual (por exemplo, `person1000_bacteria_2931.jpeg`), e um mesmo paciente aparece em até 30 fotos diferentes. Se a divisão fosse por foto solta, fotos do mesmo paciente cairiam ao mesmo tempo no treino e na validação — e aí a validação pareceria melhor do que realmente é, porque o modelo já teria "visto" aquele paciente antes. Por isso a divisão foi feita **por paciente**: conferimos que nenhum paciente aparece nos dois lados ao mesmo tempo.

Resultado da divisão:

| Conjunto | Imagens | NORMAL | PNEUMONIA | % PNEUMONIA |
|---|---|---|---|---|
| treino | 4.172 | 1.072 | 3.100 | 74,3% |
| validação | 1.044 | 269 | 775 | 74,2% |
| teste (original) | 624 | 234 | 390 | 62,5% |

### 3.2 Preparando cada imagem

Toda imagem passa por três etapas antes de entrar no modelo:
1. **Deixar em tons de cinza, mas com 3 canais:** algumas imagens são coloridas e outras não; padronizamos todas para o mesmo formato, porque a rede pré-treinada espera receber 3 canais de cor.
2. **Redimensionar para 224×224 pixels:** o tamanho de entrada que a ResNet18 espera.
3. **Normalizar os valores dos pixels:** ajustamos os números para a mesma escala usada quando a ResNet18 foi treinada originalmente — isso ajuda ela a "reconhecer" as imagens.

Só no treino, adicionamos pequenas variações aleatórias em cada imagem (rotação leve, deslocamento, zoom, brilho e contraste) — a chamada *data augmentation*. A ideia é a mesma foto nunca aparecer duas vezes exatamente igual, para o modelo aprender o padrão geral em vez de decorar as fotos de treino. **Não usamos espelhamento horizontal**, porque isso inverteria a posição do coração na imagem, o que nunca acontece numa radiografia real.

### 3.3 Balanceando as classes

Em vez de duplicar ou descartar imagens, aplicamos pesos diferentes na conta do erro: um erro numa foto NORMAL (a classe rara) pesa 1,95 vezes mais que um erro numa foto PNEUMONIA (peso 0,67). Assim o modelo é "cobrado" mais forte para acertar a classe rara, sem mexer nos dados em si.

## 4. Modelos usados e por quê

Treinamos e comparamos dois modelos (`src/modelos.py`, `src/treino.py`), com as mesmas configurações de treino (mesmo otimizador, mesma conta de erro com pesos), para a comparação ser justa.

### 4.1 CNN simples (baseline), treinada do zero

Uma rede pequena, montada e treinada do zero para servir de ponto de comparação: 4 blocos que vão detectando padrões cada vez mais complexos na imagem, terminando numa camada que decide entre NORMAL e PNEUMONIA. Tem cerca de 98 mil números ajustáveis, todos aprendidos durante o treino. A ideia de ter esse modelo simples é conseguir responder: "vale a pena o esforço extra de usar uma rede pronta e mais complexa, ou uma rede simples já resolve?"

### 4.2 ResNet18 com transfer learning

Aqui usamos uma rede já treinada antes (a ResNet18), que aprendeu a reconhecer padrões gerais de imagem (bordas, texturas, formas) usando 1,2 milhão de fotos comuns (o ImageNet). Deixamos essa parte "congelada" (sem mudar) e treinamos só a última camada para decidir entre as nossas duas classes — só 1.026 números ajustáveis, de um total de mais de 11 milhões. A aposta é que os padrões gerais que essa rede já sabe reconhecer também ajudam a entender radiografias, mesmo sem ela nunca ter visto uma antes.

## 5. Resultados

### 5.1 Validação

| Métrica | CNN simples | ResNet18 |
|---|---|---|
| Acerto geral (accuracy) | 96,3% | 94,8% |
| Recall PNEUMONIA (achou quantos casos reais) | 98,2% | 93,8% |
| Precisão PNEUMONIA (dos que disse "pneumonia", quantos acertou) | 96,8% | 99,2% |
| Recall NORMAL | 90,7% | 97,8% |

Olhando só esses números, a CNN simples pareceria a vencedora: ela encontra mais casos reais de pneumonia, que é a métrica que definimos como mais importante (deixar passar uma doença é o erro mais grave numa triagem).

### 5.2 Teste (a medida que realmente vale, com dado nunca visto)

| Métrica | CNN simples | ResNet18 |
|---|---|---|
| Acerto geral (accuracy) | 80,3% | 90,2% |
| Recall PNEUMONIA | 99,2% | 95,4% |
| Precisão PNEUMONIA | 76,3% | 89,6% |
| Recall NORMAL | **48,7%** | 81,6% |

O teste virou a história de cabeça para baixo. O recall de NORMAL da CNN simples caiu de 90,7% para 48,7%: ela passou a classificar quase tudo como PNEUMONIA. Isso até faz o recall de pneumonia parecer ainda melhor, mas é enganoso — é o comportamento de um modelo que "chuta pneumonia" na dúvida, não de um modelo que aprendeu a diferenciar as duas classes de verdade. A ResNet18 perdeu bem menos desempenho entre validação e teste, e chegou ao teste bem mais equilibrada entre as duas classes.

Essa queda mostra que as fotos do `test` têm características um pouco diferentes das fotos de `train`/`val`, mesmo sendo o mesmo dataset — algo já conhecido sobre este conjunto de dados específico. A ResNet18, por já vir com um "olho treinado" em milhões de imagens variadas, lidou melhor com essa diferença. A CNN simples, treinada do zero só com as nossas poucas milhares de radiografias, foi mais sensível a ela.

**Modelo final escolhido: ResNet18.** Mesmo com um recall de pneumonia um pouco menor no teste (95,4% contra 99,2%), a CNN simples só chega nesse número quase deixando de reconhecer pacientes normais — o que na prática sobrecarregaria a equipe médica com alarmes falsos e diminuiria a confiança no sistema. A ResNet18 mantém um recall de pneumonia alto com muito menos alarmes falsos.

### 5.3 Interpretação com Grad-CAM

Para entender em que parte da imagem o modelo (ResNet18) está se baseando, usamos uma técnica chamada Grad-CAM (`src/gradcam.py`). Ela gera um "mapa de calor" sobre a imagem, mostrando quais regiões pesaram mais na decisão do modelo — é o equivalente, para imagens, ao que "importância das variáveis" (feature importance) ou SHAP fazem para modelos de dados em tabela.

O que encontramos (`notebooks/05_avaliacao_final.ipynb`, imagens em `reports/figures/05_gradcam_*.png`):

- **Nos acertos**, o mapa de calor sempre se concentra nos pulmões e na região do coração — exatamente onde um médico olharia.
- **Em parte dos erros**, o mapa de calor se concentra numa letra "R" no canto da imagem (uma marcação que o técnico de radiologia coloca para indicar o lado do paciente), em vez do pulmão. Isso é uma evidência real de que, nesses casos, o modelo pode ter usado essa marcação como um atalho, em vez de olhar de fato para a anatomia — um risco conhecido quando o dataset de treino tem marcações consistentes desse tipo.

## 6. Discussão crítica

**O modelo pode ser usado na prática? Como?**

Só com cuidado, e só como **apoio à triagem**, nunca como diagnóstico:

- Pode ajudar a **priorizar a fila de leitura** dos exames, adiantando os casos com maior chance de pneumonia.
- Pode servir como uma **segunda opinião automática**: se o modelo discorda muito do laudo inicial, o exame é marcado para uma segunda checagem.
- **Não deve decidir sozinho.** Com recall de pneumonia de 95,4% no teste, cerca de 1 em cada 20 pacientes com pneumonia passaria despercebido — inaceitável como decisão final, mas administrável como uma camada extra de triagem, sempre revisada por um médico.

**Limitações:**

- O dataset vem de um único grupo de hospitais e é majoritariamente de crianças pequenas (1 a 5 anos); não sabemos se o modelo funcionaria bem em adultos ou com outros aparelhos de raio-X.
- Encontramos uma diferença real entre as fotos de treino/validação e as de teste dentro do próprio dataset (Seção 5.2), o que reforça a necessidade de testar o modelo com dados de outras fontes antes de qualquer uso real.
- O Grad-CAM mostrou pelo menos um caso concreto de possível uso de atalho (a marcação de lateralidade) em vez de sinal clínico — algo que precisaria ser investigado e corrigido (por exemplo, removendo essas marcações antes do treino) antes de qualquer aplicação prática.

## 7. Conclusão

O projeto percorreu o caminho completo de um problema de visão computacional aplicado à saúde: explorar os dados, dividir com cuidado (por paciente), preparar e balancear as imagens, comparar dois modelos com abordagens bem diferentes (rede simples do zero vs. rede pronta reaproveitada), avaliar no dado nunca visto e entender as decisões do modelo com Grad-CAM.

O achado mais importante não foi "qual modelo ganhou", mas a diferença entre validação e teste: um modelo pode parecer ótimo na validação e não funcionar tão bem fora dela — por isso é essencial guardar um conjunto de teste intocado até a decisão final. A ResNet18, escolhida como modelo final, entrega o melhor equilíbrio entre encontrar casos de pneumonia e não gerar alarmes falsos em excesso. Ainda assim, como qualquer modelo desse tipo, ela deve ser tratada como uma ferramenta de apoio — o médico sempre com a palavra final.
