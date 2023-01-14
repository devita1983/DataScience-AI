 <h1 align="center"> DataScience-AI </h1>
 
<h1 align="center"> PROJETO CIÊNCIA DE DADOS - Previsão de Vendas </h1>


<h1 align="center"> Step by Step - Previsão de Vendas </h1>

* Passo 1: Entendimento do Desafio
* Passo 2: Entendimento da Área/Empresa
* Passo 3: Extração/Obtenção de Dados
* Passo 4: Ajuste de Dados (Tratamento/Limpeza)
* Passo 5: Análise Exploratória
* Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
* Passo 7: Interpretação de Resultados

# Importar a base de dados

* !pip install matplotlib
* !pip install seaborn
* !pip install scikit-learn

#Depois vamos importar o arquivo que vamos analisar

* import pandas as pd
* tabela = pd.read_csv("nome_do_arquivo.csv")
* display(tabela)

obs: para usar o pd.read_csv("nome_do_arquivo.csv) primeiro você deve importar seus arquivos no Jupyter 
caso não tenha o arquivo importado deverá indicar o caminho que seu arquivo esta

# Agora vamos fazer uma Análise Exploratória dos arquivos
* Vamos tentar visualizar como as informações de cada item estão distribuídas
* Vamos ver a correlação entre cada um dos itens

Para isso , vamos importar o seaborn e o matplotlib

* import seaborn as sns
* import matplotlib.pyplot as plt

* sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
* plt.show()

outra forma de ver a mesma análise

* sns.pairplot(tabela)
* plt.show()

![image01](https://user-images.githubusercontent.com/117879893/212478668-62419066-f849-436e-beee-13a7f303b723.png)

# Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
* Separando em dados de treino e dados de teste

# Depois de treinar os dados e preparar o modelo de teste e resolver um problema de regressão, temos a tabela abaixo


![Imagem02](https://user-images.githubusercontent.com/117879893/212482984-290cfed6-a60b-4e7c-80aa-788ebc62f748.png)



