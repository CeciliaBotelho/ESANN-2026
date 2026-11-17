# PLANO — próximos testes para o artigo (método ORIGINAL CONGELADO por blocos; Bars & Stripes)

**10/11/2026, cerca de 11:15.** Revisão de Claude, não independente. Esta é uma proposta revisável. **Nada foi executado** além da extração descritiva de artefatos (§3).

Não houve treino, consulta nova ao teste para escolher, instalação, rede ou gasto. Os resultados e códigos originais, `sources/` e o `.tex` ficam intactos. INbreast e a biblioteca ampliada são investigações **separadas** e não entram aqui.

## 0. Ponto de partida (tudo será relatado, inclusive o desfavorável)

**Comparação BS 800/200/200** (`reproducao_benchmarks_qflair/comparacao_bs`), com 4 qubits, teto de 32 e C = 1:

| braço | AccAver | BCE | AUC |
|---|---|---|---|
| BLOCO_CAL | 0,780 | 0,499 | 0,823 |
| QF+CAL | 0,725 | 0,512 | 0,757 |
| QF_ENTREGUE | 0,741 | 0,501 | 0,756 |

Critério prefixado **atingido**: ΔAccAver = +0,054 e ΔBCE = −0,014.

- **É um único conjunto de teste, de 200 imagens.** A incerteza posterior calculada pelo ROOT (`INCERTEZA_POSTERIOR_ROOT.json`; bootstrap pareado, exploratório, condicional aos modelos) dá IC 95% de ΔbAcc [−0,031; +0,140], ΔBCE [−0,113; +0,087] e ΔAUC [−0,028; +0,159]. **Os três intervalos incluem 0.** O ganho observado ainda não está replicado.
- **Comparação a 32 operações abstratas, não na convergência:**
  - a BCE de validação do QF ainda caía na porta 31 (0,515);
  - o BLOCO se estabiliza entre k = 28 e 32 (0,471–0,474).

  A comparação é "a teto igual de operações abstratas". O artigo autoral leva o QF até 150 portas.
- **Assimetrias já declaradas:**
  - canal constante π no oficial;
  - o BLOCO usa só 16 colunas;
  - 4 qubits nos dois braços, enquanto o artigo autoral usa 5;
  - a reprodução R1/R2 do artigo é relatada à parte.

## 1. Replicação em dados Bars & Stripes NOVOS (prioridade 1)

> **ATUALIZAÇÃO 10/11 ~11:40 (prevalece sobre o texto abaixo onde houver conflito).** A proveniência foi resolvida em `VERIFICACAO_FONTE_PENNYLANE.md` e verificada independentemente pelo ROOT.
> - Os CSVs do Q-FLAIR são a saída do script do artigo qml-benchmarks (seed 42) com o gerador **original** f06e20c: igualdade numérica ≤ 1e-12 e rótulos exatos, mas não identidade de bits em todos os valores.
> - A geração das réplicas usa `gerador.py` v2: `gera_par(semente, 1000, 1000)`, com **sementes novas** 20261201–20261205 (≠ 42), treino e teste na mesma sequência, achatamento direto e rótulo −1→0.
> - A divisão interna 800/200 sai do treino de 1000, como no original.
> - A "validação estatística" anterior foi substituída pela verificação de proveniência; as falhas ficam como histórico.
> - Os resultados BS já medidos não mudam.
> - **A bateria continua NÃO autorizada.**

### Proveniência (verificada só localmente)

- O gerador oficial do Q-FLAIR (`Datasets/DataGeneration.py`) **não sintetiza** os dados: ele baixa `qml.data.load("other", name="bars-and-stripes")`, o dataset do PennyLane, e grava os CSVs `*_4d_*`, imagens 4×4 com 16 features.
  - O repositório Q-FLAIR é Apache-2.0 (`LICENSE` local).
  - O dataset do PennyLane vem do pacote qml-benchmarks (Bowles et al. 2024).
- **Pendência:** obter o código-fonte exato do gerador (`generate_bars_and_stripes`), o commit e a licença. Não há cópia local; isso exige uma consulta de rede de leitura, a autorizar.
  - Sem a fonte, a alternativa é uma **reimplementação a partir da descrição publicada**, declarada como tal e não chamada de "gerador oficial".
- **Checagem leve nos CSVs autorais de treino (1000):**
  - σ estimado pelo 2º momento (E[x²] = 1 + σ²) = 0,493, compatível com um ruído gaussiano de σ = 0,5 sobre pixels ±1 (a confirmar na fonte);
  - classes 514/486;
  - cerca de 8,7% das imagens têm sinal constante, ou seja, rótulo intrinsecamente ambíguo, então o teto de Bayes fica abaixo de 1;
  - a orientação separa o rótulo em cerca de 94%.
- **Validação do gerador antes de usar** (sem métrica de modelo): comparar as estatísticas acima, por exemplo σ, a fração constante e o balanço, entre o gerado e o autoral, com tolerância fixada antes.
  - Bit-identidade não é possível, porque a semente do dataset PennyLane é desconhecida.

### Desenho (fixado antes; número finito)

- **R = 5 conjuntos novos**, com sementes de geração 20261201, 20261202, 20261203, 20261204 e 20261205, gravadas antes de qualquer treino.
- **Para cada réplica:**
  - geram-se 1000 imagens de treino e um teste novo com **1000 imagens** (maior que 200, para reduzir o ruído do teste);
  - a divisão interna é estratificada 800/200, com semente própria da réplica, fixada;
  - a escala, os CSVs e o formato seguem exatamente `prepara_comparacao_bs.py`.
- **Não é replicação:** novas divisões do mesmo teste de 200, ou do mesmo treino de 1000. Isso fica proibido de ser chamado assim.
- **Braços, idênticos ao protocolo aprovado:**
  - QF_ENTREGUE e QF+CAL: oficial, 4 qubits, teto de 32, seed 12345, parallel 2;
  - BLOCO_CAL, congelado (hash conferido);
  - BLOCO_RAW e CONST.

  Além deles, os braços de ablação do §2, que são baratos.
- **Critério por réplica, o mesmo do original:** ΔAccAver(BLOCO_CAL − QF+CAL) ≥ +0,010 E ΔBCE ≤ 0.
- **Resumo prefixado:** "**replicado**" se o critério for atingido em ≥ 4/5 réplicas E a média de ΔAccAver for ≥ +0,010 com a média de ΔBCE ≤ 0. Caso contrário, "**não replicado**".
  - As 5 réplicas são relatadas **sempre**, inclusive as desfavoráveis.
  - Secundário prefixado: bootstrap pareado por réplica, com limiar fixo do treino.
- **Limites declarados:**
  - a replicação varia os dados, não a semente do otimizador (o QF continua com seed 12345, e o BLOCO é determinístico);
  - continua sendo a teto de 32, sem convergência.
- **Parada:**
  - são 5 réplicas, sem extensão nem substituição;
  - uma falha técnica (processo morto, truncamento) faz a réplica ficar INCOMPLETA, relatada separadamente. Ela só pode ser refeita com a **mesma** semente e a falha declarada;
  - nenhum retune.
- **Custo (base medida no BS original):**
  - QF com cerca de 2.700 s de relógio e 5.050 s de CPU por réplica de 800;
  - predição e replays com cerca de 40 s, um pouco mais com teste de 1000;
  - BLOCO com cerca de 20 s por braço.
  - **Total de cerca de 4 h de relógio e cerca de 7,5 h de CPU**, com no máximo 2 workers e um QF por vez, mais cerca de 30 min de implementação e checks.

## 2. Ablação: decisão conjunta de 2 operações contra sequencial (prioridade 2; roda dentro das réplicas)

**O que o BLOCO decide por etapa:**
- o par (feature nova k, posição j entre 10: RY_q→RX_q ou RY_r→RXX_{q,q+1});
- os 2 pesos (w₁, w₂) otimizados **juntos**, com a mesma feature nas duas portas;
- sem reutilizar a feature em etapas seguintes.

**Admissibilidade equivalente.** Uma linha de base que proíba reutilizar a feature com 1 porta por decisão ficaria com 16 portas contra 32. Isso seria injusto, e por isso fica proibido.

Os braços sequenciais mantêm **a mesma estrutura**: cada feature nova ocupa exatamente 2 slots consecutivos, com o mesmo pool, as mesmas posições, a mesma conectividade em linha, o mesmo objetivo J_C1, até 32 operações e o mesmo prefixo e calibração.

| braço | seleção | otimização |
|---|---|---|
| **BLOCO** (congelado) | (k, j) escolhidos juntos, pelo ganho do par | 2D conjunta (L-BFGS-B) |
| **SEQ** | sub-etapa 1: escolhe k e a 1ª porta RY_q pelo ganho **sozinha** (1D). Sub-etapa 2: com k fixo, escolhe a 2ª porta entre as posições compatíveis com q (RX_q, RXX com q) | 1D, depois 1D, com w₁ congelado |
| **SEQ+REF** | **as mesmas escolhas do SEQ** | depois da sub-etapa 2, reotimiza (w₁, w₂) juntos em 2D a partir dos valores do SEQ |

**O que cada contraste isola e o que não isola:**
- **BLOCO − SEQ+REF:** efeito da **seleção** conjunta do par, com a otimização final 2D igual.
  - Continua confundido com a miopia gulosa (a sub-etapa 1 julga RY sozinha) e com a trajetória diferente nas etapas seguintes.
- **SEQ+REF − SEQ:** efeito da **otimização** conjunta dos 2 pesos, com a seleção igual.
- **Não se alega isolamento total:** as escolhas mudam o estado, e o estado muda as etapas seguintes.

**Orçamento (declarado; não se iguala tudo):**
- **BLOCO**, por etapa: 80 consultas de 2 ângulos, 4 sementes × posições × features livres na pré-triagem e 8 refinos 2D.
- **SEQ:**
  - 1 ângulo usa 3 pontos, ou seja, 2 consultas novas;
  - 4 posições × features livres × 4 sementes 1D na sub-etapa 1, com 8 refinos 1D;
  - 2 a 3 posições na sub-etapa 2, com o mesmo número de refinos.
- Para cada braço se relatam consultas, avaliações, refinos e CPU. O SEQ deve sair mais barato.

**Critério prefixado, nas 5 réplicas:**
- "decisão conjunta contribui" se BLOCO − SEQ+REF tiver média de ΔBCE ≤ −0,010 e Δ < 0 em ≥ 4/5;
- "otimização conjunta contribui" com a mesma regra para SEQ+REF − SEQ;
- caso contrário, "não demonstrado". Tudo é relatado.

**Implementação e checks (antes de qualquer réplica):**
- reconstrução de 1 ângulo contra a simulação direta;
- SEQ+REF = BLOCO quando as escolhas coincidem (canário artificial);
- determinismo;
- canário artificial em que o par conjunto é necessário: um sinal que só aparece com as 2 portas, para verificar que o teste tem poder;
- isolamento de teste.

**Custo:** segundos por braço e réplica. A estimativa de cerca de 10 min inclui os checks.

## 3. Qualidade × custo (já extraído; `custos_curva_bs.json`, script `extrai_custos_curva.py`)

| | BLOCO_CAL | QF (oficial) |
|---|---|---|
| operações | 32 abstratas: 16 RY, 5 RX, 11 RXX | 32 abstratas: 30 Rxx_x, 2 Ryy_x, cada uma com rz adicional no circuito qiskit |
| 2 qubits | 11 (10 no prefixo k = 30) | 32 rotuladas como 2q (rxx/ryy entre j−1 e j) |
| profundidade (nossa contagem de camadas) | 30 (28 no prefixo) | não extraída (depende de transpilação) |
| construção / treino | cerca de 13 s de CPU de construção (livro: 18 s de relógio e 16 s de CPU) | 2.697 s de relógio e 5.050 s de CPU (parallel 2; cerca de 84 s por porta) |
| consultas | 16 etapas × 80 = 1.280 avaliações de circuito em lote, e 128 refinos | **não extraídas** (exige instrumentar o oficial) |
| prefixo escolhido | k = 30 (BCE val 0,471) | k = 31 (QF+CAL) |

**Limites:**
- 32 operações abstratas **não são** custo nativo igual. O QF tem mais portas de 2 qubits e um rz extra por porta.
- Os tempos vêm de **implementações e simuladores diferentes** (numpy próprio contra qiskit oficial), então **não** se infere speedup.
- Uma comparação defensável de custo exige uma métrica comum: contagem de avaliações de circuito, ou contagem nativa após a mesma transpilação. Isso é pendência e não usa hardware.

A curva por prefixo, com calibração no treino e validação e teste lado a lado, é descritiva e não foi usada para escolher. Ela está no JSON e mostra:
- o BLOCO se estabiliza entre k = 28 e 32;
- o QF ainda melhora na validação até a porta 31.

## 3b. Execução paralela máxima (pedido da usuária, 10/11 ~11:25)

**Máquina:** 10 núcleos (6 de performance e 4 de eficiência), 16 GB de RAM, 46% livres e carga de cerca de 2,4 no momento da medida.

**O que é paralelizável:**
- **As 5 réplicas são independentes:** dados, BLOCO, QF, replay e predição. É o principal ganho.
- **Os braços BLOCO, SEQ e SEQ+REF de todas as réplicas** custam segundos cada e rodam juntos, num pool de até 4 processos, antes dos QF.
- **Dentro de um QF**, o `parallel` oficial (hoje 2) **não** será aumentado, a menos que um check prove resultado idêntico (Gate_collect e Cost_collect com Δ = 0) entre parallel 2 e 4 em poucas portas. Do contrário, isso mudaria a execução registrada. O ganho vem de rodar várias réplicas ao mesmo tempo.

**Quem limita é a memória, não a CPU.** O BS original teve pico de 7 processos × cerca de 386 MB por QF, um teto conservador de cerca de 2,7 GB por réplica, porque o RSS soma páginas compartilhadas. Com o piso de 25% livres, isso dá cerca de 3,4 GB utilizáveis hoje.

**Fila com admissão dinâmica:**
- Uma nova réplica QF só inicia se a memória livre for ≥ 25% + a margem medida de uma réplica, e se houver pelo menos 2 núcleos livres.
- Teto de **4 QF simultâneos**, ou seja, 8 núcleos, deixando 2 livres.
- Arquivo PARAR, livro de custos por réplica e manifesto, como na bateria INbreast.
- Um bench curto mede a memória real de 2 QF simultâneos (poucas portas) e fixa a margem antes da bateria.

**Estimativa de relógio:**

| QF simultâneos | relógio dos QF | observação |
|---|---|---|
| 1 (sequencial, como antes) | cerca de 3,8 h | |
| 2 | cerca de 2,3 h | 3 rodadas |
| 3 | cerca de 1,7 h | 2 rodadas, mais cerca de 20% de contenção |
| 4–5 | cerca de 1,2 h | usam os núcleos de eficiência, mais lentos |

A **CPU agregada não muda** (cerca de 7,5 h); paralelizar só reduz o relógio. Os resultados são idênticos ao sequencial, porque cada réplica é determinística e isolada em sua pasta, e a auditoria confere isso.

**Exige autorização nova do ROOT:** o limite anterior era de ≤ 2 workers e um QF oficial por vez.

**Não paraleliza o bloqueio:** sem a fonte e a licença do gerador, a replicação não começa. O que dá para adiantar em paralelo agora é implementar o SEQ e o SEQ+REF com checks, e o bench de memória e paridade do `parallel`.

## 4. Recomendação e pendências

**Ordem recomendada:**
1. Validar o gerador. Depende de obter a fonte e a licença, o que exige autorização de leitura em rede; senão, reimplementação declarada.
2. Implementar o SEQ e o SEQ+REF, com checks e canários artificiais.
3. Fazer um bench curto.
4. Submeter à revisão do ROOT.
5. Rodar uma única bateria de 5 réplicas, com braços originais e de ablação (cerca de 4 h de relógio e cerca de 7,5 h de CPU).
6. Escrever o relatório automático com todas as réplicas.

**Opcional, a decidir antes da bateria:** QF com teto maior (por exemplo, 64) em **uma** réplica pré-designada (a 1ª), para mostrar a curva além de 32. Custo de cerca de +1,5 h. Não muda o critério.

**Pendências:**
- fonte, commit e licença do gerador qml-benchmarks;
- tolerâncias da validação do gerador;
- contagem de consultas e de portas nativas do QF;
- decidir o teste de 1000 contra o de 200 (recomendo 1000);
- revisão do ROOT antes de qualquer execução longa.
