# O que foi modificado

## 1. Referência Q-FLAIR realmente executada no artigo

Base: commit `4f331783fc5bda5492fa3fffff0a845b0805eb39` de
https://github.com/phels23/Q-FLAIR .

Mudanças no código-fonte: somente a atribuição do número de qubits para **4**,
em `QNN_logLoss.py` e `QNN_predict.py`. Os arquivos originais, as cópias usadas
e os dois patches estão em `third_party/qflair/`.

Mantidos: ordem `Rxx_x,Ryy_x,U2_x,U3_x,H_y`, grade de 1.001 pesos, seed 12345,
limiar de parada do código 1e-8, conectividade em anel e reutilização de features.
O número de trabalhadores foi configurado em 2; não é um parâmetro científico.

O teto de 32 operações foi aplicado externamente nos experimentos originais,
aguardando registros completos de porta e custo. É truncamento deliberado, não
convergência do algoritmo. O arquivo de parâmetros original usa 8 trabalhadores;
o benchmark empacotado usa 2, como a execução do estudo.

Foram fornecidas 16 colunas variáveis e uma coluna extra constante. O código
Q-FLAIR substitui a última por pi. Assim, nenhuma feature variável foi perdida.
A busca por blocos usa as 16 variáveis e não o canal constante.

## 2. Avaliação final comum adicionada

`QF+CAL` não é a saída original do Q-FLAIR: usamos as mesmas portas treinadas,
calibração logística C=1 ajustada só no treino, e escolha de prefixo ou constante
pela BCE de validação. O limiar ROC para acurácia balanceada é ajustado no treino.
A referência considera prefixos de uma operação; nosso método, blocos completos.

A saída bruta final do Q-FLAIR foi conservada como resultado descritivo separado.

## 3. Nosso método do artigo

Blocos locais RY -> RX, ou RY em um endpoint -> RXX no par vizinho de uma cadeia
de quatro qubits. Cada inserção usa uma feature nova nas duas operações.
Dez posições ordenadas: quatro locais e seis orientações de interação.

Reconstrução de nove pontos por posição, oito novas avaliações; coeficientes
compartilhados entre features. Quatro inicializações fixas, oito finalistas,
L-BFGS-B em [-1,1]^2; critério de parada 1e-12, no máximo 16 blocos/32 operações.
Os pesos dos blocos anteriores não são reotimizados.

RAW usa BCE direta de P0 na seleção. CAL ajusta a calibração dentro da seleção.
Ambos usam a mesma calibração final. Só a inclinação a é penalizada:
sigma(a*(2*P0-1)+b), com a^2/(2*n_train), C=1.

SEQ e SEQ_REF são controles parciais da etapa CAL. Trajetórias, inicializações e
esforço diferem: não constituem isolamento causal do ganho de usar blocos.

## 4. Dados e sementes

Gerador Xanadu qml-benchmarks `f06e20c`, noise_std=0.5, imagens 4x4.
Seed 42: verificação de reprodução do dataset disponibilizado.
Seeds próprias e partições: `config/seeds.json`.

Primeira partição: 800/200/200; cinco iniciais e cinco de confirmação: 800/200/1000.
Min-max/clipping ajustado só nos 800 exemplos de treino. Na preparação do
benchmark do Q-FLAIR, a escala também é processada pela rotina oficial; o replay
de referência conserva exatamente esse comportamento.

## 5. Reorganização para este pacote

As definições das funções científicas foram extraídas dos arquivos executados,
sem alteração de seus corpos. Ver `provenance/sources.json`. Imports relativos e
constantes explícitas substituem a cadeia de imports de estudos não relacionados.
O verificador compara as árvores sintáticas das funções com as fontes preservadas.
Foi acrescentada uma interface de linha de comando portátil; nenhuma MLP foi incluída.

`run_reference.py` aplica o teto de 32 de forma síncrona numa cópia temporária do
script: interrompe depois de escrever a 32ª porta e seu custo. Esse ajuste de
empacotamento fica em `runtime_cap.patch` de cada nova execução; não altera os
arquivos originais ou executados preservados e não foi usado para obter a tabela
histórica. O loop de seleção anterior ao teto não é modificado.

O pacote permite recalcular artefatos históricos, refazer o método e repetir a
referência; esses três níveis de reprodução são identificados separadamente.
