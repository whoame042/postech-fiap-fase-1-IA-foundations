# Tech Challenge Fase 1 — backlog de tasks

Fonte: `POSTECH - Tech Challenge - Fase 1.pdf`  
Curso: Pós Tech — IA para Devs  
Peso: **90% da nota de todas as disciplinas da fase**  
Formato: grupo · entrega obrigatória

Processo SDLC genérico (3 tamanhos): `process_sdlc/_INDICE.md`  
Binding deste challenge: `process_sdlc/aplicacoes/tech_challenge_fase1.md` (recomendado: medium)  
Playbooks operacionais: `epics/_INDICE.md`

## Épicos

| Épico | Branch | Tasks | Arquivo |
|---|---|---|---|
| E0 Descoberta e setup | `epic/E0_descoberta_e_setup` | T00–T02 | [E0_descoberta_e_setup.md](E0_descoberta_e_setup.md) |
| E1 Exploração de dados | `epic/E1_exploracao_de_dados` | T03–T04 | [E1_exploracao_de_dados.md](E1_exploracao_de_dados.md) |
| E2 Pré-processamento | `epic/E2_preprocessamento` | T05–T08 | [E2_preprocessamento.md](E2_preprocessamento.md) |
| E3 Modelagem | `epic/E3_modelagem` | T09–T11 | [E3_modelagem.md](E3_modelagem.md) |
| E4 Avaliação, interpretação e ética | `epic/E4_avaliacao_interpretacao_etica` | T12–T15 | [E4_avaliacao_interpretacao_etica.md](E4_avaliacao_interpretacao_etica.md) |
| E5 EXTRA CNN | `epic/E5_extra_cnn` | T16 | [E5_extra_cnn.md](E5_extra_cnn.md) |
| E6 Entregáveis FIAP | `epic/E6_entregaveis_fiap` | T17–T21 | [E6_entregaveis_fiap.md](E6_entregaveis_fiap.md) |

## O que o desafio pede

Hospital universitário quer **triagem automática** de exames e documentos clínicos para apoiar o médico — não substituí-lo. Nesta fase a entrega mínima é **classificação tabular com Machine Learning** (doença sim/não). CNN em imagem é **EXTRA** (sobe nota se a parte obrigatória não fechar 100%).

**Restrição clínica (enunciado):** o médico sempre tem a palavra final no diagnóstico. Nenhum artefato pode vender o modelo como diagnóstico autônomo.

## Fora de escopo nesta fase

- Prontuário eletrônico completo, fila de atendimento, login de hospital
- LLM / RAG / GenAI (fases posteriores do curso)
- Deploy em nuvem de produção, monitoramento clínico real
- Usar dado real de paciente identificável (somente dataset **público**)

---

## Mapa requisito → task

| Requisito do PDF | Task | Épico |
|---|---|---|
| Escolher dataset médico público e discutir o problema | T01 | [E0](E0_descoberta_e_setup.md) |
| Projeto Python estruturado; notebook ou scripts | T02 | [E0](E0_descoberta_e_setup.md) |
| Carregar a base e explorar características | T03 | [E1](E1_exploracao_de_dados.md) |
| Estatísticas descritivas + visualizações + discussão | T04 | [E1](E1_exploracao_de_dados.md) |
| Limpeza (ausentes, inconsistentes) | T05 | [E2](E2_preprocessamento.md) |
| Pipeline de pré-processamento em Python; categóricas/numéricas | T06 | [E2](E2_preprocessamento.md) |
| Análise de correlação | T07 | [E2](E2_preprocessamento.md) |
| Split claro treino / validação / teste | T08 | [E2](E2_preprocessamento.md) |
| Duas ou mais técnicas de classificação | T09, T10 (+ T11 recomendado) | [E3](E3_modelagem.md) |
| Treino no conjunto de treinamento | T09–T11 | [E3](E3_modelagem.md) |
| Avaliação no teste: accuracy, recall, F1 + discussão da métrica | T12 | [E4](E4_avaliacao_interpretacao_etica.md) |
| Interpretação: feature importance e SHAP | T13, T14 | [E4](E4_avaliacao_interpretacao_etica.md) |
| Discussão crítica de uso prático + papel do médico | T15 | [E4](E4_avaliacao_interpretacao_etica.md) |
| EXTRA: diagnóstico por imagem com CNN | T16 | [E5](E5_extra_cnn.md) |
| Repo Git + código-fonte + Dockerfile + README | T17 | [E6](E6_entregaveis_fiap.md) |
| Dataset (ou link de download) | T18 | [E6](E6_entregaveis_fiap.md) |
| Relatório técnico (pré-processamento, modelos, resultados) | T19 | [E6](E6_entregaveis_fiap.md) |
| Resultados (prints, gráficos, análises) | T20 | [E6](E6_entregaveis_fiap.md) |
| Vídeo ≤ 15 min (YouTube/Vimeo, público ou não listado) | T21 | [E6](E6_entregaveis_fiap.md) |

---

## Ordem de execução

```
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08
                                              ↓
                                    T09 / T10 / T11 (em paralelo depois do split)
                                              ↓
                                         T12 → T13 → T14 → T15
                                              ↓
                              T16 (EXTRA, em paralelo a T12–T15 se houver capacidade)
                                              ↓
                                    T17 + T18 + T20 → T19 → T21
```

Estimativa do caminho obrigatório: **~40 h**. EXTRA CNN: **+8–12 h**.

---

## Definição de pronto da fase (gate de entrega)

- [ ] Repo Git com código rodando via `docker compose up` **ou** `docker build` + comando no README
- [ ] Notebook/script reproduz o fluxo EDA → pipeline → treino → métricas → SHAP
- [ ] Pelo menos **dois** classificadores, split treino/validação/teste **sem leakage**
- [ ] Relatório PDF cobre as três seções pedidas (pré-processamento, modelos+porquê, resultados+interpretação)
- [ ] Discussão deixa explícito: modelo é **apoio**, médico decide
- [ ] Vídeo ≤ 15 min mostra o sistema rodando e o fluxo
- [ ] Dataset versionado no repo **ou** link estável + hash/instruções de download

---

## Prompt de execução (por task)

Ao implementar uma task, o critério de aceite do épico é a Definition of Done. Não pular T08 antes de treinar. Não olhar o teste antes de T12. Não tratar T16 como substituto da classificação tabular.
