# ESANN — Feature-conditioned quantum block search

Pacote de reprodução de **Block Search Conditioned on Features for Sequential Quantum Data Encoding**.

Este pacote contém somente os experimentos Bars and Stripes apresentados no artigo.
Não contém INbreast, dados de pacientes, MLPs, ambientes virtuais ou credenciais.
É uma cópia separada: os estudos e o manuscrito originais não foram modificados.

## Começar

Use Python 3.11. Os dois ambientes são separados porque o Q-FLAIR usa versões antigas e incompatíveis de NumPy/Qiskit.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py tables
```

`verify` regenera os dados dos 11 conjuntos, confere arquivos por hash, compara
as definições científicas com as originais preservadas, executa os circuitos
salvos, confere o replay do Q-FLAIR e recalcula a tabela do artigo.
Isso **não** é um novo treinamento da referência. Não depende do servidor ou de caminhos do Mac.

Resultados: `runs/verify/verification.json`, `runs/tables/TABLES.md`, `metrics.csv` e `summary.json`.
As métricas por conjunto incluem a acurácia balanceada dos cinco datasets.

## Executar novamente nosso método

```bash
# Cinco conjuntos de confirmação: RAW, de ponta a ponta
python reproduce.py rerun --stage confirmation --mode raw --output runs/confirmation_raw

# Avaliação inicial: RAW, CAL e os controles parciais SEQ/SEQ_REF
python reproduce.py rerun --stage initial --mode all --output runs/initial_all

# Partição disponibilizada: RAW e CAL
python reproduce.py rerun --stage released --mode cal --output runs/released
```

Os comandos recusam sobrescrever saídas existentes. Para repetir, use outra pasta.
Nenhum parâmetro é escolhido pelo teste. Diferenças numéricas entre plataformas
são registradas em `prediction_max_difference`; não são escondidas nem corrigidas
alterando os resultados históricos.

## Executar a referência Q-FLAIR

Em outro ambiente Python 3.11:

```bash
python3.11 -m venv .venv-qflair
source .venv-qflair/bin/activate
python -m pip install -r requirements-qflair.txt

# Inferência do circuito histórico: Qiskit versus simulador NumPy + QF+CAL
python run_reference.py --seed 20263001 --saved-circuit --output runs/qflair_replay_20263001

# Treino oficial novamente, limitado às mesmas 32 operações (pode levar horas)
python run_reference.py --seed 20263001 --output runs/qflair_train_20263001
```

O segundo comando refaz somente o dataset indicado. Execute separadamente as
cinco sementes `20263001`–`20263005` para uma repetição integral da referência.
Não usa IBM Cloud, credenciais, shots nem hardware pago: somente simulação local.
O custo histórico da referência foi medido com outro mecanismo de execução;
os tempos deste pacote não devem ser apresentados como repetição desses tempos.

## Organização

- `src/esann/`: simulador, reconstrução, busca RAW/CAL, controles e calibração.
- `artifacts/released/42/`: partição disponibilizada, incluindo circuitos e previsões.
- `artifacts/initial/`: cinco conjuntos iniciais; CAL principal, RAW secundário.
- `artifacts/confirmation/`: cinco novos conjuntos; RAW principal pré-especificado.
- `third_party/`: gerador e Q-FLAIR com licenças, originais e patches.
- `provenance/`: funções originais e protocolos preservados, com mapa de origem.
- `docs/ALTERACOES.md`: mudanças científicas versus mudanças de organização.
- `config/seeds.json`: sementes e sua função.
- `validation/`: verificações efetivamente executadas nesta preparação.

## Ler os resultados

Na confirmação, RAW versus QF+CAL apresenta médias de aproximadamente:

| Método | Acurácia balanceada | BCE | AUC |
|---|---:|---:|---:|
| QF+CAL | 78,28% | 0,43861 | 0,84230 |
| Busca por blocos, RAW | 83,81% | 0,36916 | 0,90791 |

O critério é ganho de pelo menos 0,01 em acurácia balanceada e diferença de BCE
menor ou igual a zero: quatro de cinco conjuntos e as médias o satisfazem.
Os testes t pareados e de permutação são análises posteriores, sem ajuste para
múltiplos testes. Os cinco conjuntos são realizações do mesmo gerador, não cinco
domínios independentes. A comparação envolve procedimentos completos e não isola
causalmente somente o tamanho do bloco.

## Código de terceiros e publicação

O gerador pertence ao projeto Xanadu `qml-benchmarks`; o Q-FLAIR pertence aos
autores da referência. Suas licenças Apache-2.0 e avisos foram preservados.
A licença MIT preexistente do projeto também foi preservada, sem mudar sua
atribuição. Ver `THIRD_PARTY_NOTICES.md`.

Repositório: https://github.com/CeciliaBotelho/ESANN-2026

As verificações executadas estão em `validation/2026-11-17/`. A versão vinculada
ao artigo deve ser identificada por uma release ou commit. Nenhum DOI foi atribuído.
Mantenha os arquivos de proveniência e licenças nas redistribuições.
