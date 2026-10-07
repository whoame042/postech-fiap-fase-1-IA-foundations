# T15 — Uso prático e ética

## 1. Papel

No hospital universitário do enunciado o score serviria para **ordenar a fila de revisão**: casos com alta P(maligno) primeiro (priorização / segunda leitura). **Não** fecha diagnóstico e **não** substitui o patologista.

## 2. Pode ir para a prática?

Só como piloto em dados *como* o Wisconsin (mesmo tipo de medida), com humano no loop, depois de calibrar o limiar pelo custo de FN vs FP (T12). Não há deploy em produção neste trabalho.

## 3. O que quebra

- **População / drift:** Wisconsin é FNA de mama (núcleo celular, EUA, décadas atrás) ≠ fluxo de um SUS 2026, ≠ outro laboratório, ≠ outro microscópio.
- **Desbalanceamento:** maligno é 37%; accuracy 0.99 no teste **não** autoriza “o sistema diagnostica câncer”. A métrica que manda é recall (T12).
- **Viés / o que falta:** sem idade, sexo, raça, clínica ou imagem como feature social/assistencial. O modelo só vê 30 números.
- **Matriz do vencedor no teste (logística, uma vez):** TN 54 · FP 0 · **FN 1** · TP 31 (N_teste=86). Um falso negativo em 32 malignos — exatamente o custo que o recall tenta reduzir. Não retreinamos depois de ver isso.

Evidências amarradas: importância liderada por `radius_se` / `texture_worst` (T13); waterfall do FN índice 190 (T14).

## 4. LGPD

Este repo usa dataset **público** de pesquisa (UCI), sem PII de paciente real. Em hospital, dado de saúde é **sensível** (LGPD art. 5º, II e art. 11). Sem base legal, sem minimizar, sem responsável, o sistema não sobe.

## 5. Fecho

*O médico tem a palavra final no diagnóstico.*
