#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"


#define PadroesValidacao 39
#define PadroesTreinamento 39 
#define Sucesso 0.04		    // 0.0004
#define NumeroCiclos 100000     // Exibir o progresso do treinamento a cada NumeroCiclos ciclos

//Sigmoide
#define TaxaAprendizado 0.3     //0.3 converge super rápido e com uma boa precisão (sigmoide na oculta).
#define Momentum 0.9            // Dificulta a convergencia da rede em minimos locais, fazendo com que convirja apenas quando realmente se tratar de um valor realmente significante.
#define MaximoPesoInicial 0.5


//Saidas da rede neural (Exemplo): Vocês precisam definir os intervalos entre 0 e 1 para cada uma das saídas de um mesmo neuronio.
//Alem disso, esses sao exemplos, voces podem ter mais tipos de saidas, por exemplo defni que o robo ira rotacionar para a direita, esquerda ou nao rotacionar, no geral isso nao teria como alterar, entao deixei como exemplo.

//Direcao de rotacao (Neuronio da camada de saida 1)
//   Direita              Reto            Esquerda
//0.125 - 0.375      0.375 - 0.625      0.625 - 0.875
//    0,25				  0,5                0,75
#define OUT_DR_DIREITA    0.25    
#define OUT_DR_ESQUERDA   0.5   
#define OUT_DR_FRENTE     0.75

//Para a direcao de movimento nao ha muita diferenca, entao acredito que voces possam adotar esses valores
//Direcao de movimento (Neuronio da camada de saida 2)
//	  Frente		    Re
//   0.1 - 0.5      0.5 - 0.9
#define OUT_DM_FRENTE     0.3      
#define OUT_DM_RE         0.7

//O angulo nao possui receita de bolo, voces podem altera-lo em diferentes niveis, ou ate lidar com valores continuos
//Angulo de rotacao  (Neuronio da camada de saida 3)
// 0.1-0.3 = 0 graus, 0.3-0.5 = 5 graus, 0.5-0.7 = 15 graus, 0.7-0.9 = 45 graus
#define OUT_AR_SEM_ROTACAO  0.2    // 0 graus (intervalo 0.1-0.3)
#define OUT_AR_LATERAL      0.4    // 5 graus (intervalo 0.3-0.5)
#define OUT_AR_DIAGONAL     0.6    // 15 graus (intervalo 0.5-0.7)
#define OUT_AR_FRONTAL      0.8    // 45 graus (intervalo 0.7-0.9)

//Essa e uma sugestao, voces tambem podem trabalhar com a velocidade de movbvimento tambem sendo retornada pela rede neural, pois quanto mais proximo dos obstaculos, mais lento deveria ser o movimento
//Velocidade de movimento (Neuronio da camada de saida 4)

#define ALCANCE_MAX_SENSOR 5000

//Sobre o numero de neuronio das camadas, a camada de entrada ira refletir o numero de sensores, entao seriam esses 8. Se voces possuissem mais variaveis relevantes para essa operacao, poderiam utiliza-las. 
//Pensem que ate mesmo a velocidade de movimento atual do robo poderia ser utilizada como entrada para decidir no momento t+1
// Camada de entrada
#define NodosEntrada 8

//A quantidade de neuronios nessa camada esta fortemente vinculada a complexidade do problema, sendo uma boa pratica iniciar os esperimentos com pelo menos um neuronio a mais do que na camada de entrada.
// Camada oculta
#define NodosOcultos 9

//Essa camada ira definir a quantidade de diferentes variaveis de saida, nesse meu exemplo sao elas  direcao de rotacao (DR), direcao de movimento (DM) e angulo de rotacao (AR).
//Mas como eu disse no comentario acima, a rede poderia ter um quarto neuronio na camada de saida, para definir a velocidade de mopvimento do robo, ou ate outras saidas que voces condiderem importanes para a resolucao do problema.
// Camada de saída
#define NodosSaida 3

//Estrutura da rede neural, sintam-se livres para adicionar novas camadas intermediarias, alterar a funcao de ativacao, bias e etc.
class NeuralNetwork {
public:
    int i, j, p, q, r;
    int IntervaloTreinamentosPrintTela;
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    float Rando;
    float Error;
    float AcumulaPeso;

    int esquerda = 0;
    int diagonal_esquerda_lateral = 0;
    int diagonal_esquerda_frontal = 0;
    int frente_esquerda = 0;
    int direita = 0;
    int diagonal_direita_lateral = 0;
    int diagonal_direita_frontal = 0;
    int frente_direita = 0;

    // Camada oculta
    float Oculto[NodosOcultos];
    float PesosCamadaOculta[NodosEntrada + 1][NodosOcultos];
    float OcultoDelta[NodosOcultos];
    float AlteracaoPesosOcultos[NodosEntrada + 1][NodosOcultos];
    ActivationFunction* activationFunctionCamadasOcultas;

    // Camada de saída
    float Saida[NodosSaida];
    float SaidaDelta[NodosSaida];
    float PesosSaida[NodosOcultos + 1][NodosSaida];
    float AlterarPesosSaida[NodosOcultos + 1][NodosSaida];
    ActivationFunction* activationFunctionCamadaSaida;

    float ValoresSensores[1][NodosEntrada] = {{0, 0, 0, 0, 0, 0, 0, 0}};

    //Exemplo de dadod de treinamento, cada um representando a distancia lida por um sensor
    const float Input[PadroesTreinamento][NodosEntrada] = {
    //ESQUERDA 							  FRENTE								  DIREITA
    // {0, 		1, 		2, 		3, 		4, 		5, 		6, 		7}
        // Obstáculo à esquerda - virar direita (6 padrões)
        {1000,       900,       1500,       4000,    5000,      5000,       5000,       5000},
        {800,        900,       1000,      3000,    4000,      4000,       5000,       5000},
        {600,        700,       800,      2000,    3000,      3000,       5000,       5000},
        // Obstáculo à esquerda + frente - virar a direita
        {2000,       1000,       1200,       1000,     1300,       3000,       4000,       5000},
        {1200,        900,       1000,       800,     1000,       2000,       3000,       5000},
        {600,        900,       1000,       600,     800,        1000,       2000,       5000},
        // Obstáculo à esquerda + frente - virar a direita (novos padrões)
        {1465,       1511,       1929,       3162,     4540,       4607,       5000,       5000},
        {1350,       1450,       1800,       3000,     4400,       4500,       4900,       5000},
        {1600,       1650,       2100,       3300,     4700,       4800,       5000,       5000},

        // Obstáculo à direita - virar esquerda (6 padrões)
        {5000, 5000, 5000, 5000, 4000, 1500, 900, 1000},
        {5000, 5000, 4000, 4000, 3000, 1000, 900, 800},
        {5000, 5000, 3000, 3000, 2000, 800, 700, 600},
        
        // Obstáculo à direita + frente - virar à direita
        {5000, 4000, 3000, 1300, 1000, 1200, 1000, 2000},
        {5000, 3000, 2000, 1000, 800, 1000, 900, 1200},
        {5000, 2000, 1000, 800, 600, 1000, 900, 600},
        
        // Caminho livre - seguir em frente (6 padrões)
        {5000,      5000,       5000,    5000,    5000,      5000,       5000,       5000},
        {4000,      4200,       4400,    4000,    4000,      4400,       4200,       4000},
        {3000,      3200,       3400,    3000,    3000,      3400,       3200,       3000},
        {2000,      2200,       2400,    2500,    2500,      2400,       2200,       2000},
        {2500,      2700,       1900,    2000,    2000,      1900,       2700,       2500},
        {2000,      2200,       2400,    3500,    3500,      2400,       2200,       2000},
        
        // Corredor estreito - apenas extremidades livres (sensor0 e sensor7 > 3000, outros < 2000)
        // Virar esquerda (3 padrões) - sensor0 > sensor7
        {4500,      1500,       1800,    1900,    1700,      1600,       1400,       3500},
        {4200,      1200,       1500,    1800,    1600,      1400,       1300,       3200},
        {4800,      1800,       1900,    1950,    1800,      1700,       1500,       3800},
        // Virar direita (3 padrões) - sensor7 > sensor0
        {3500,      1500,       1800,    1900,    1700,      1600,       1400,       4500},
        {3200,      1200,       1500,    1800,    1600,      1400,       1300,       4200},
        {3800,      1800,       1900,    1950,    1800,      1700,       1500,       4800},
        
        // Obstáculo à esquerda/frente, caminho livre à direita - virar direita (3 padrões)
        {2163,      2203,       1703,    1548,    1612,      1945,       2884,       2935},
        {2100,      2150,       1650,    1500,    1550,      1900,       2800,       2900},
        {2250,      2250,       1750,    1600,    1680,      2000,       2950,       3000},
        // Obstáculo à direita/frente, caminho livre à esquerda - virar esquerda (3 padrões - espelhado)
        {2935,      2884,       1945,    1612,    1548,      1703,       2203,       2163},
        {2900,      2800,       1900,    1550,    1500,      1650,       2150,       2100},
        {3000,      2950,       2000,    1680,    1600,      1750,       2250,       2250},
        
        // Obstáculo à esquerda, caminho livre à frente/direita - virar direita (3 padrões)
        {2114,      1442,       1476,    5000,    5000,      5000,       5000,       5000},
        {2050,      1400,       1450,    4900,    4950,      5000,       5000,       5000},
        {2200,      1500,       1520,    5000,    5000,      5000,       5000,       5000},
        // Obstáculo à direita, caminho livre à frente/esquerda - virar esquerda (3 padrões - espelhado)
        {5000,      5000,       5000,    5000,    5000,      1476,       1442,       2114},
        {5000,      5000,       5000,    4950,    4900,      1450,       1400,       2050},
        {5000,      5000,       5000,    5000,    5000,      1520,       1500,       2200}
    };
    float InputNormalizado[PadroesTreinamento][NodosEntrada];

    //Exemplo de output esperado para os dados de treinamento acima
    // Ângulos de rotação: 0.1-0.3 = 0graus, 0.3-0.5 = 5graus, 0.5-0.7 = 15graus, 0.7-0.9 = 45graus
    const float Objetivo[PadroesTreinamento][NodosSaida] = {
    //   DR,  AR,  DM
        // Obstáculo à esquerda - virar direita (6 padrões)
        // DR = direita, AR = 15 graus (diagonal)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à esquerda + frente - virar a direita
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à esquerda + frente - virar a direita (novos padrões)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à direita - virar esquerda (6 padrões)
        // DR = esquerda, AR = 15 graus (diagonal)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita + frente - virar a esquerda
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Caminho livre - seguir em frente (6 padrões)
        // DR = frente, AR = 0 graus (sem rotação)
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        
        // Corredor estreito - apenas extremidades livres - virar esquerda (3 padrões)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Corredor estreito - apenas extremidades livres - virar direita (3 padrões)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à esquerda/frente, caminho livre à direita - virar direita (3 padrões)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita/frente, caminho livre à esquerda - virar esquerda (3 padrões - espelhado)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à esquerda, caminho livre à frente/direita - virar direita (3 padrões)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita, caminho livre à frente/esquerda - virar esquerda (3 padrões - espelhado)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE}
    };
    
    //Aqui eu utilizei os mesmos valores, mas o correto sera definir dados de validacao diferentes daqueles apresentados a rede em seu treinamento, para garantir que ela nao tenha apenas "decorado" as respostas.
    //Dados de validação
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
        // Obstáculo à esquerda - virar direita (6 padrões)
        {950,        850,        1400,       3800,    4800,      4900,       4900,       4900},
        {750,        850,        950,       2800,    3800,      3900,       4900,       4900},
        {550,        650,        750,      1800,    2800,      2900,       4900,       4900},
        // Obstáculo à esquerda + frente - virar a direita
        {1900,       950,        1150,       950,     1250,       2800,       3900,       4900},
        {1150,        850,        950,       750,     950,       1900,       2900,       4900},
        {550,        850,        950,       550,     750,        950,       1900,       4900},
        // Obstáculo à esquerda + frente - virar a direita (novos padrões de validação)
        {1400,       1450,       1850,       3100,     4500,       4550,       4950,       4950},
        {1300,       1400,       1750,       2950,     4350,       4450,       4850,       4950},
        {1550,       1600,       2050,       3250,     4650,       4750,       4950,       5000},

        // Obstáculo à direita - virar esquerda (6 padrões)
        {4900, 4900, 4900, 4900, 3900, 1400, 850, 950},
        {4900, 4900, 3900, 3900, 2900, 950, 850, 750},
        {4900, 4900, 2900, 2900, 1900, 750, 650, 550},
        
        // Obstáculo à direita + frente - virar à esquerda
        {4900, 3900, 2900, 1250, 950, 1150, 950, 1900},
        {4900, 2900, 1900, 950, 750, 950, 850, 1150},
        {4900, 1900, 950, 750, 550, 950, 850, 550},
        
        // Caminho livre - seguir em frente (6 padrões)
        {4900,      4900,       4900,    4900,    4900,      4900,       4900,       4900},
        {3900,      4100,       4300,    3900,    3900,      4300,       4100,       3900},
        {2900,      3100,       3300,    2900,    2900,      3300,       3100,       2900},
        {1900,      2100,       2300,    2400,    2400,      2300,       2100,       1900},
        {2400,      2600,       1800,    1900,    1900,      1800,       2600,       2400},
        {1900,      2100,       2300,    3400,    3400,      2300,       2100,       1900},
        
        // Corredor estreito - apenas extremidades livres (sensor0 e sensor7 > 3000, outros < 2000)
        // Virar esquerda (3 padrões de validação) - sensor0 > sensor7
        {4400,      1400,       1700,    1800,    1600,      1500,       1300,       3400},
        {4100,      1100,       1400,    1700,    1500,      1300,       1200,       3100},
        {4700,      1700,       1800,    1850,    1700,      1600,       1400,       3700},
        // Virar direita (3 padrões de validação) - sensor7 > sensor0
        {3400,      1400,       1700,    1800,    1600,      1500,       1300,       4400},
        {3100,      1100,       1400,    1700,    1500,      1300,       1200,       4100},
        {3700,      1700,       1800,    1850,    1700,      1600,       1400,       4700},
        
        // Obstáculo à esquerda/frente, caminho livre à direita - virar direita (3 padrões de validação)
        {2100,      2150,       1650,    1500,    1550,      1900,       2800,       2900},
        {2050,      2100,       1600,    1450,    1500,      1850,       2750,       2850},
        {2200,      2200,       1700,    1580,    1630,      1950,       2900,       2950},
        // Obstáculo à direita/frente, caminho livre à esquerda - virar esquerda (3 padrões de validação - espelhado)
        {2900,      2800,       1900,    1550,    1500,      1650,       2150,       2100},
        {2850,      2750,       1850,    1500,    1450,      1600,       2100,       2050},
        {2950,      2900,       1950,    1630,    1580,      1700,       2200,       2200},
        
        // Obstáculo à esquerda, caminho livre à frente/direita - virar direita (3 padrões de validação)
        {2050,      1400,       1450,    4900,    4950,      5000,       5000,       5000},
        {2000,      1350,       1420,    4850,    4900,      4950,       5000,       5000},
        {2150,      1480,       1500,    5000,    5000,      5000,       5000,       5000},
        // Obstáculo à direita, caminho livre à frente/esquerda - virar esquerda (3 padrões de validação - espelhado)
        {5000,      5000,       5000,    4950,    4900,      1450,       1400,       2050},
        {5000,      5000,       5000,    4900,    4850,      1420,       1350,       2000},
        {5000,      5000,       5000,    5000,    5000,      1500,       1480,       2150}
    };
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
        // Obstáculo à esquerda - virar direita (6 padrões)
        // DR = direita, AR = 15 graus (diagonal)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à esquerda + frente - virar a direita
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à esquerda + frente - virar a direita (novos padrões de validação)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à direita - virar esquerda (6 padrões)
        // DR = esquerda, AR = 15 graus (diagonal)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita + frente - virar a esquerda
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Caminho livre - seguir em frente (6 padrões)
        // DR = frente, AR = 0 graus (sem rotação)
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        
        // Corredor estreito - apenas extremidades livres - virar esquerda (3 padrões de validação)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Corredor estreito - apenas extremidades livres - virar direita (3 padrões de validação)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à esquerda/frente, caminho livre à direita - virar direita (3 padrões de validação)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita/frente, caminho livre à esquerda - virar esquerda (3 padrões de validação - espelhado)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        
        // Obstáculo à esquerda, caminho livre à frente/direita - virar direita (3 padrões de validação)
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        // Obstáculo à direita, caminho livre à frente/esquerda - virar esquerda (3 padrões de validação - espelhado)
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL, OUT_DM_FRENTE}
    };
    
    //--

public:
    NeuralNetwork();
    void treinarRedeNeural();
    void inicializacaoPesos();
    int treinoInicialRede();
    void PrintarValores();
    ExpectedMovement testarValor();
    ExpectedMovement definirAcao(int sensor0, int sensor1, int sensor2, int sensor3, int sensor4, int sensor5, int sensor6, int sensor7);
    void validarRedeNeural();
    void treinarValidar();
    void normalizarEntradas();
    void setupCamadas() ;
};

#endif // NEURALNETWORK_H