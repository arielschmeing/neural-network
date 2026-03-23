#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"

#define PadroesValidacao 18
#define PadroesTreinamento 18 
#define Sucesso 0.04        
#define NumeroCiclos 20000    

#define TaxaAprendizado 0.3    
#define Momentum 0.9           
#define MaximoPesoInicial 0.5

#define OUT_DR_DIREITA    0.25  
#define OUT_DR_ESQUERDA   0.5   
#define OUT_DR_FRENTE     0.75

#define OUT_DM_FRENTE     0.3       
#define OUT_DM_RE         0.7

#define OUT_AR_SEM_ROTACAO  0.2   
#define OUT_AR_LATERAL      0.4    
    bool carregarPesos();
#define OUT_AR_DIAGONAL     0.6    
#define OUT_AR_FRONTAL      0.8   

#define ALCANCE_MAX_SENSOR 5000

#define NodosEntrada 8
#define NodosOcultos 9
#define NodosSaida 3

class NeuralNetwork {
public:
    int i, j, p, q, r;
    int IntervaloTreinamentosPrintTela;
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    float Rando;
    float Error;
    float AcumulaPeso;

    float Oculto[NodosOcultos];
    float PesosCamadaOculta[NodosEntrada + 1][NodosOcultos];
    float OcultoDelta[NodosOcultos];
    float AlteracaoPesosOcultos[NodosEntrada + 1][NodosOcultos];
    ActivationFunction* activationFunctionCamadasOcultas;

    float Saida[NodosSaida];
    float SaidaDelta[NodosSaida];
    float PesosSaida[NodosOcultos + 1][NodosSaida];
    float AlterarPesosSaida[NodosOcultos + 1][NodosSaida];
    ActivationFunction* activationFunctionCamadaSaida;

    float ValoresSensores[1][NodosEntrada] = {{0, 0, 0, 0, 0, 0, 0, 0}};

    const float Input[PadroesTreinamento][NodosEntrada] = {
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        
        {5000, 5000, 1000, 300,  300,  1000, 5000, 5000},
        
        {5000, 5000, 2000, 900,  900,  2000, 5000, 5000},
        
        {400,  400,  400,  5000, 5000, 5000, 5000, 5000},
        {5000, 600,  600,  1000, 5000, 5000, 5000, 5000},
        {700,  5000, 5000, 5000, 5000, 5000, 5000, 5000},

        {5000, 5000, 5000, 5000, 5000, 400,  400,  400},
        {5000, 5000, 5000, 5000, 1000, 600,  600,  5000},
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 700},

        {400,  400,  400,  400,  2000, 5000, 5000, 5000},
        {5000, 5000, 5000, 2000, 400,  400,  400,  400},

        {800,  1000, 5000, 5000, 5000, 5000, 2000, 2000},
        {2000, 2000, 5000, 5000, 5000, 5000, 1000, 800},

        {5000, 5000, 500,  1000, 5000, 5000, 5000, 5000},
        {5000, 5000, 5000, 5000, 1000, 500,  5000, 5000},
        
        {300, 300, 300, 300, 300, 300, 300, 300},

        {2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000},

        {500, 500, 1000, 5000, 5000, 1000, 500, 500}
    };
    
    float InputNormalizado[PadroesTreinamento][NodosEntrada];

    const float Objetivo[PadroesTreinamento][NodosSaida] = {
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_RE},
        
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},

        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},

        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL,     OUT_DM_FRENTE},

        {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},

        {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},

        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_RE},


        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},

        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}
    };
    
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000, 1000, 300,  300,  1000, 5000, 5000},
        {5000, 5000, 2000, 900,  900,  2000, 5000, 5000},
        {400,  400,  400,  5000, 5000, 5000, 5000, 5000},
        {5000, 600,  600,  1000, 5000, 5000, 5000, 5000},
        {700,  5000, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000, 5000, 5000, 5000, 400,  400,  400},
        {5000, 5000, 5000, 5000, 1000, 600,  600,  5000},
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 700},
        {400,  400,  400,  400,  2000, 5000, 5000, 5000},
        {5000, 5000, 5000, 2000, 400,  400,  400,  400},
        {800,  1000, 5000, 5000, 5000, 5000, 2000, 2000},
        {2000, 2000, 5000, 5000, 5000, 5000, 1000, 800},
        {5000, 5000, 500,  1000, 5000, 5000, 5000, 5000},
        {5000, 5000, 5000, 5000, 1000, 500,  5000, 5000},
        {300, 300, 300, 300, 300, 300, 300, 300},
        {2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000},
        {500, 500, 1000, 5000, 5000, 1000, 500, 500}
    };
    
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_RE},
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FRONTAL,     OUT_DM_RE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}
    };

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
    void setupCamadas();
    
    void salvarPesos();
    bool carregarPesos();
};

#endif // NEURALNETWORK_H