import tf from '@tensorflow/tfjs-node';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
 const pessoas = [
     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" },
     { nome: "Zé", idade: 28, cor: "verde", localizacao: "Curitiba" },
     { nome: "Maria", idade: 40, cor: "vermelho", localizacao: "Rio" }
 ];

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = pessoas.map(pessoa => normalizaPessoa(pessoa));

console.log("Dados de entrada normalizados:");
console.log(tensorPessoasNormalizado);

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const categorias = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1],  // basic - Carlos
    [0, 0, 1],  // basic - Zé
    [0, 1, 0] // medium - Maria
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

// quanto mais dados de treino, melhor o modelo aprende, mas mais tempo leva pra treinar
const model = await trainModel(inputXs, outputYs); 

// Exemplo de novas pessoas para previsão
const pessoasPrevisao = [
    { nome: "João", idade: 22, cor: "azul", localizacao: "São Paulo" } ,
    { nome: "Luiza", idade: 35, cor: "vermelho", localizacao: "Rio" },
    { nome: "Pedro", idade: 45, cor: "verde", localizacao: "Curitiba" }
]
const pessoasNormalizado = pessoasPrevisao.map(pessoa => normalizaPessoa(pessoa));

// Agora vamos usar o modelo treinado para fazer previsões sobre novas pessoas
const predicions = await predict(model, pessoasNormalizado);

// Exibindo as previsões para as novas pessoas
console.log("Previsões para as novas pessoas:");
predicions.forEach((predicao, indice) => {
    console.log(`${pessoasPrevisao[indice].nome}:`);
    predicao.forEach(p => {
        console.log(`  ${p.categoria}: ${(p.probabilidade * 100).toFixed(2)}%`);
    });
});


// Função para treinar o modelo
async function trainModel(inputXs, inputYs) {
    const model = tf.sequential();
   
    //Primeira camada da rede:
    // entrada de 7 posições (idade normalizada +3 cores+3 localizações)
    
    // 100 neuronios = aqui coloquei tudo isso, porque tem pouca base de treino
    // quanto mais reuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai ter


    // A ReLU age como um filtro
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegouy nesse neurônio é positiva, passa pra frente!
    // Se for zero ou negativa, pode jogar fora, não vai servir pra nada

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)
    //Activation: softmax normaliza a saida em probrabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    //Compilando o modelo
    //optimizer Adam (adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais
    //loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium vai ser sempre [1,0,0]
    //Metrica: accuracy (precisão)
    // Quanto mais distante da previsão do modelo da resposta correta,
    // maior o erro (loss)
    // Exemplo clássico: classificação de imagens, recomendação, categorização de usuário
    // qualquer coisa emque a resposta certa é "apenas uma entre várias posssíveis"


    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    //Treinando o modelo
    //Verbose: desabilita o log interno (e usa só callback)
    //epocjs: quantiade de vezes que vai rodar no dataset
    //shuffle: embaralha os dados, para evitar viés de ordem

    await model.fit(inputXs, inputYs, {
        verbose: 0,
        epochs: 100, 
        shuffle: true ,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    })

    return model;
}

// Função para fazer previsões com o modelo treinado
async function predict(model, pessoas) {
    
    //transformar a pessoa em um tensor de entrada (mesmo formato do treino)
    const tensorPessoa = tf.tensor2d(pessoas);

    //Fazendo a previsão
    const previsao = model.predict(tensorPessoa);

    //A previsão é um tensor, precisamos extrair os valores
    const previsaoArray = await previsao.array();
    console.log("Previsão:", previsaoArray);

    return previsaoArray.map(predicao => {

        return categorias.map((categoria, indice) => ({
            categoria,
            probabilidade: predicao[indice]
        }));
    });
}

// Função para normalizar os dados de entrada
function normalizaPessoa(pessoa) {
    const idadeMaxima = Math.max(...pessoas.map(p => p.idade));
    const idadeMinima = Math.min(...pessoas.map(p => p.idade));
    const idadeNormalizada = Number(((pessoa.idade - idadeMinima) / (idadeMaxima - idadeMinima)).toFixed(2));
    const corAzul = pessoa.cor === "azul" ? 1 : 0;
    const corVermelho = pessoa.cor === "vermelho" ? 1 : 0;
    const corVerde = pessoa.cor === "verde" ? 1 : 0;
    const localizacaoSP = pessoa.localizacao === "São Paulo" ? 1 : 0;
    const localizacaoRio = pessoa.localizacao === "Rio" ? 1 : 0;
    const localizacaoCuritiba = pessoa.localizacao === "Curitiba" ? 1 : 0;
    
    return [idadeNormalizada, corAzul, corVermelho, corVerde, localizacaoSP, localizacaoRio, localizacaoCuritiba];
}