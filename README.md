# Classificação de Planos com Inteligência Artificial

Este repositório contém uma implementação de uma Rede Neural Artificial utilizando **TensorFlow.js** em ambiente **Node.js**. 
O modelo categoriza os usuários em três tipos de planos (Premium, Medium e Basic) baseando-se nos dados fornecidos como modelo.

## Tecnologias

- **Node.js**
- **TensorFlow.js for Node** (@tensorflow/tfjs-node)

## Objetivo

Prever a probabilidade de o usuário pertencer a um dos três perfis de plano.

## Arquitetura do Modelo

1. Camada Oculta: 
○ Input shape: 07 (um para cada atributo). 
○ 80 neurônios: escolhidos empiricamente para aumentar a 
capacidade de aprendizado mesmo com poucos dados. 
○ Função de ativação: ReLU (Rectified Linear Unit), que permite que 
apenas valores positivos passem adiante. 
2. Camada de Saída: 
○ 3 neurônios: um para cada categoria. 
○ Função de ativação: Softmax, que transforma os valores de saída 
em probabilidades normalizadas.

## Como Executar

1. Instale as dependências:
   ```bash
   npm install
   ```
2. rode o programa:
   ```bash
   npm start
   ```
