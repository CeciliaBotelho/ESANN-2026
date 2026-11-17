# ESANN — cópia limpa para reprodução

## O que está pronto

- Código portátil do método do artigo, sem dependência das pastas dos estudos originais.
- Experimento inicial, cinco datasets iniciais e cinco de confirmação, com dados sintéticos, circuitos e previsões.
- Explicação das alterações no Q-FLAIR: `docs/ALTERACOES.md`.
- Seeds completas e suas funções: `config/seeds.json`.
- Arquivos originais de terceiros, licenças e patches preservados.
- Comandos de reprodução: `README.md`.

## O que foi executado em 17/11/2026

1. Regeneração/verificação dos dados dos 11 conjuntos.
2. Reavaliação dos circuitos arquivados e da tabela do artigo.
3. Nova execução de RAW nos cinco conjuntos de confirmação.
4. Nova execução de RAW/CAL na partição disponibilizada.
5. Nova execução de RAW/CAL/SEQ/SEQ_REF nos cinco conjuntos iniciais.
6. Replay independente em Qiskit de um circuito treinado da referência.

As 27 novas construções do nosso método/controles produziram as mesmas estruturas,
prefixos selecionados e previsões históricas, com diferença máxima zero nesta máquina.
O replay Qiskit apresentou diferença máxima de aproximadamente 1,2e-15.
Resultados e registros estão em `validation/2026-11-17/`.

**O treino completo dos cinco Q-FLAIR não foi repetido nesta preparação.** O código e
os comandos estão disponíveis; essa repetição é mais demorada. Reavaliar circuitos
salvos é diferente de refazer seu treinamento. Os tempos de execução históricos
não foram substituídos pelos tempos desta verificação.

## Preservação

Repositório de destino: https://github.com/CeciliaBotelho/ESANN-2026
O manuscrito e os estudos originais ficaram intactos.
Não foram incluídos dados de pacientes, credenciais ou ambientes virtuais.
