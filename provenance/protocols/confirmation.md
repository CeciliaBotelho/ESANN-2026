# PROTOCOLO — confirmação prospectiva BLOCO_RAW vs QF+CAL (congelado antes de qualquer modelo)

**10/11/2026, ~15:58.** Liberado pelo ROOT por volta das 15:50, com atualização de paralelismo por volta das 15:54. Revisão de Claude, não independente.

**Natureza.** É uma **nova etapa primária**. Na bateria de replicação (`bs_artigo_proximos_testes/bateria`), o BLOCO_RAW era secundário; aqueles resultados ficam preservados e **não** são promovidos retrospectivamente. Este teste confirma o **desempenho do RAW contra o QF**. Ele **não** isola o efeito de blocos, por isso não há CAL, SEQ, SEQ_REF nem controle extra.

**Dados**
- 5 conjuntos NOVOS, com sementes de geração **20263001–20263005**.
  - Antes do uso, verifiquei que essas sementes não aparecem como valor inteiro em nenhum manifesto ou params do projeto (`VERIFICACAO_SEMENTES.json`). Não há colisão.
- Geração: `gera_par(s, 1000, 1000)` do gerador v2 (`bs_artigo_proximos_testes/gerador.py`), versão original qml-benchmarks f06e20c com σ = 0,5. Não se usa a versão transposta antiga.
- Divisão estratificada 800/200 dentro dos 1000 de treino, com semente própria registrada por conjunto: **20263101–20263105**.
- A Escala [0, π] é ajustada só no treino. O teste de 1000 fica intacto.
- Não há nenhuma linha bruta em comum entre partições, com os conjuntos já usados (semente 42 e 20261201–05) nem entre os conjuntos novos.
- Manifesto e hashes estão em `bateria/rS/dados/dados_meta.json`.

**Braços (cada um roda uma vez)**
- **BLOCO_RAW (primário):** original e congelado, `OB.constroi('BLOCO_RAW')` + `OB.final_c1`.
  - C = 1; prefixos pares + CONST escolhidos pela validação.
  - Inalterados: LIB, eixos, no-reuse, pesos, sementes de refino, refit e readout (o hash de `objetivo_blocos.py` é conferido).
- **QF+CAL (comparador primário):** oficial, 4 qubits, mesmo pool, seed 12345, parallel 2 e teto 32. Usa C = 1, prefixo + CONST pela validação e limiar ROC no treino.
- **Secundários:** QF_ENTREGUE e CONST. O CONST é a referência nula; as diferenças RAW − CONST e QF+CAL − CONST são descritivas.
- **Excluído:** 64 operações, ajuste da biblioteca, retune, sementes extras e substituição de conjunto.

**Métricas**
- AccAver no teste, com limiar ROC escolhido no treino sobre as probabilidades de cada braço.
- BCE no teste.
- AUC no teste, apenas descritiva.

**Critério prospectivo (igual ao anterior)**
- Por conjunto, o critério operacional é atingido se ΔAccAver(RAW − QF+CAL) ≥ +0,010 **E** ΔBCE ≤ 0.

**Resumo**
- **confirmado:** 5/5 conjuntos COMPLETOS **E** critério em ≥ 4/5 **E** média ΔAccAver ≥ +0,010 **E** média ΔBCE ≤ 0.
- **não confirmado:** os 5 estão completos e alguma dessas condições falha.
- **NÃO AVALIADO (n/5):** caso contrário. Um conjunto interrompido tecnicamente não conta como falha nem como sucesso.
- Todos os conjuntos são relatados. As métricas de um conjunto só entram no critério se a auditoria dele passar: hashes, CSVs, BLOCO congelado, replay A (custo), replay B (simulador), guarda do predict, certificações da calibração e paridade da AccAver oficial.

**Checks pré-execução (`checks.json`): todos ok.**
- Decoder do gerador: a semente 42 reproduz o Q-FLAIR com erro ≤ 1e-12 e rótulos exatos.
- Dados novos e sementes distintas.
- **Canário do avaliador_raw** nas 5 réplicas encerradas:
  - reproduz exatamente (diferença 0) RAW, QF+CAL, QF_ENTREGUE e CONST;
  - os ΔRAW − QF são corretos e o primário **não** é CAL;
  - a auditoria inclui replay e certificação;
  - o limiar usa só o treino (n = 800);
  - a escolha do prefixo QF independe do teste;
  - uma auditoria falhando é detectada.
- Isolamento artificial: o construtor RAW só recebe treino e é determinístico, e o final_c1 independe do teste.

**Execução (`scripts/fila.py`)**
- **Fase A:** BLOCO_RAW com ≤ 5 processos, antes dos QF.
- **Fase B:** até **5 QF × parallel 2 = 10 workers**, BLAS 1 e ≤ 2 GB de RSS por processo.
  - Admissão: memória livre − 5 ≥ 30%. A 5ª vaga exige ainda livre ≥ 35% antes de entrar.
  - Sob contenção (load1 ≥ 11) admite menos jobs, sem reiniciar nenhum.
  - Guarda: memória livre < 25% em 3 leituras corta o último QF admitido.
- **Prazo e teto:** prazo absoluto **19:50:08**; teto de CPU agregada **43.200 s**, incluindo o preparo.
- **Controle:** caffeinate, PARAR e monitor.
- **Fase C, sempre:** `avalia.py` (avaliação única) e `relatorio.py`, mesmo se a bateria estiver incompleta. Nenhum outro estudo roda em paralelo.

Isto é exploratório: sem IC, sem vencedor eleito, sem vantagem quântica, e o teto de 32 não é convergência.
