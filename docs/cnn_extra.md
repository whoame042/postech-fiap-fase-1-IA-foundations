# T16 — EXTRA CNN (não é a entrega principal)

Código e treino: [`extra-cnn-pneumonia/`](../extra-cnn-pneumonia/). Dependências isoladas: [`requirements-cnn.txt`](../requirements-cnn.txt) (não misturar torch no `requirements.txt` do tabular / Docker T17).

**O médico tem a palavra final.** CNN = pré-triagem de raio-X; o tabular (T01–T15) é outra tarefa.

## Modelos

- CNN simples **do zero** (baseline).
- **Transfer learning** ResNet18 (ImageNet, cabeça treinada). Vencedor.

Augment só no treino. Early stopping na val. Teste intocado até a avaliação final.

## Métricas no teste (ResNet18, uma vez)

| Métrica | CNN simples | ResNet18 (final) |
|---|---|---|
| Accuracy | 80,3% | **90,2%** |
| Recall PNEUMONIA | 99,2% | **95,4%** |
| Precisão PNEUMONIA | 76,3% | **89,6%** |
| Recall NORMAL | 48,7% | **81,6%** |

Recall de pneumonia manda (FN = pneumonia perdida). A CNN simples “chuta pneumonia” no teste (recall NORMAL 48,7%) — por isso o vencedor é a ResNet18. Fonte: `extra-cnn-pneumonia/reports/relatorio_tecnico.md` §5.

## Overfitting (treino vs val)

Curvas: `reports/figures/06_cnn_curvas_train_val.png` (ResNet18) e `06_cnn_simples_curvas.png`.  
A CNN simples fecha a val com 96% e cai no teste (acc 80%, NORMAL 49%): overfit / shift do `test`. A ResNet18 perde menos (val 94,8% → teste 90,2%). Dropout, pesos de classe e augment no treino estão no relatório §3–4.

## Ética

Não fundir CNN + tabular num “diagnóstico único”. Radiologista decide. Capítulo extra do PDF (T19) aponta para este doc.
