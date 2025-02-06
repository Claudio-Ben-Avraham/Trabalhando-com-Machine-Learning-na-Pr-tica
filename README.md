# Trabalhando com Machine Learning na Prática

Este repositório contém um passo a passo detalhado sobre como criar, treinar e implantar um modelo de previsão usando **Machine Learning**. O modelo foi criado com a plataforma **Azure Machine Learning** e os pontos de extremidade (endpoints) estão configurados para acessar o modelo de forma prática.

## Objetivo

O objetivo é demonstrar como criar um modelo preditivo, configurar os pontos de extremidade para fazer previsões e entender o fluxo de trabalho de Machine Learning em uma plataforma de nuvem.

## Tecnologias Usadas

- **Azure Machine Learning**: Plataforma de nuvem para treinamento, implantação e gerenciamento de modelos de ML.
- **Python**: Linguagem usada para desenvolver o modelo de aprendizado de máquina.
- **Scikit-Learn**: Biblioteca de aprendizado de máquina para treinamento e avaliação do modelo.
- **Azure SDK for Python**: Para interação com a Azure Machine Learning.
- **Jupyter Notebook**: Ambiente para escrever e executar o código.

## Passos para Criar o Modelo

### 1. Preparar o Ambiente

Primeiro, você precisa configurar o ambiente de desenvolvimento. Certifique-se de ter uma conta no **Azure** e instale o SDK do **Azure Machine Learning**. Execute o seguinte comando para instalar as dependências necessárias:

```bash
pip install azureml-sdk
pip install scikit-learn

2. Criar e Configurar o Workspace no Azure

Acesse o Azure Portal e crie um Workspace no Azure Machine Learning. Isso é necessário para armazenar recursos como modelos, experimentos e recursos computacionais.

    Acesse o Azure Machine Learning e crie um novo Workspace.
    Anote o ID do Workspace, Grupo de Recursos e Nome da Assinatura, pois serão necessários no código.

3. Carregar os Dados

Carregue seus dados de treinamento. Para este exemplo, utilizamos um conjunto de dados fictício que contém informações sobre vendas de um produto, com variáveis como preço, quantidade e condições de mercado.

import pandas as pd

# Carregar dados de exemplo
data = pd.read_csv("dados_vendas.csv")

# Exibir as primeiras linhas dos dados
print(data.head())

4. Pré-processamento dos Dados

Antes de treinar o modelo, os dados precisam ser preparados. Isso pode incluir a remoção de valores ausentes, normalização e divisão em variáveis independentes (X) e dependentes (y).

from sklearn.model_selection import train_test_split

# Separar variáveis independentes e dependentes
X = data[['preco', 'quantidade', 'condicao_mercado']]
y = data['vendas']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

5. Treinar o Modelo

Escolha um modelo para treinar. Neste exemplo, usamos uma regressão linear simples, mas você pode usar qualquer modelo adequado para o seu problema.

from sklearn.linear_model import LinearRegression

# Criar o modelo
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

6. Avaliar o Modelo

Depois de treinar o modelo, é importante avaliá-lo para verificar seu desempenho em dados de teste.

from sklearn.metrics import mean_squared_error

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular o erro quadrático médio
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio: {mse}")

7. Implantar o Modelo no Azure

Após treinar o modelo, você pode implantá-lo no Azure Machine Learning. Primeiro, registre o modelo no workspace do Azure:

from azureml.core import Workspace, Model

# Conectar-se ao workspace
ws = Workspace.from_config()

# Registrar o modelo
model = Model.register(workspace=ws, model_path="modelo.pkl", model_name="modelo_vendas")

Agora, crie uma imagem de contêiner (container image) para o modelo:

from azureml.core.image import ContainerImage

# Definir os detalhes do contêiner
image_config = ContainerImage.image_configuration(execution_script="score.py")

# Criar a imagem
image = Image.create(workspace=ws, name="modelo_vendas_imagem", models=[model], image_config=image_config)
image.wait_for_creation(show_output=True)

8. Criar o Ponto de Extremidade (Endpoint)

Com o modelo implantado, é hora de criar um ponto de extremidade para que você possa fazer previsões.

from azureml.core.webservice import AciWebservice, Webservice

# Configurar o serviço ACI
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Criar e implantar o serviço
service = Webservice.deploy_from_image(workspace=ws, name="modelo-vendas-service", image=image, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)

Agora você pode acessar o modelo através do endpoint REST gerado.
9. Fazer Previsões Usando o Endpoint

Com o modelo implantado e o ponto de extremidade configurado, você pode fazer previsões usando chamadas REST para o endpoint.

import requests
import json

# Ponto de extremidade do serviço
endpoint = "https://<nome-do-serviço>.azurewebsites.net/score"

# Dados de entrada para previsão
data = {"data": [[100, 20, 3]]}  # Exemplo de entrada

# Enviar solicitação POST
response = requests.post(endpoint, json=data, headers={"Content-Type": "application/json"})
prediction = response.json()
print(f"Previsão: {prediction}")

10. Limpeza de Recursos

Após o uso, é importante remover os serviços implantados para evitar custos adicionais.

# Excluir o serviço
service.delete()

Conclusão

Agora você tem um modelo de previsão treinado, implantado e acessível através de um ponto de extremidade REST. Esse processo abrange a criação de um modelo preditivo, a configuração de um ponto de extremidade no Azure Machine Learning, e como consumir o modelo em tempo real para fazer previsões.
