# Classificação de Planos com Inteligência Artificial

Este repositório contém uma implementação de uma Rede Neural Artificial utilizando **TensorFlow.js** em ambiente **Node.js**. 
O modelo categoriza os usuários em três tipos de planos (Premium, Medium e Basic) baseando-se nos dados fornecidos como modelo.

## Tecnologias

- **Node.js**
- **TensorFlow.js for Node** (@tensorflow/tfjs-node)

## Objetivo

O sistema recebe 07 variáveis de entrada (features) que descrevem o perfil de um usuário:

- **Preferências:** Cor favorita.
- **Demográficas:** Idade e Localização.

O objetivo é prever a probabilidade de o usuário pertencer a um dos três perfis de plano.

## Arquitetura do Modelo

O modelo foi construído utilizando a API Sequencial do TensorFlow:

1.  **Input Layer:** 07 dimensões.
2.  **Hidden Layer:** 100 neurônios com ativação **ReLU** (para capturar padrões não-lineares).
3.  **Output Layer:** 03 neurônios com ativação **Softmax**, entregando uma distribuição probabilística entre as categorias.

## Como Executar

1. Instale as dependências:
   ```bash
   npm install
   ```
2. rode o programa:
   ```bash
   npm start
   ```
