#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa : TV, Jornal e Rádio
# 
# 

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados
#### Importar a Base de dados
# In[1]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# In[12]:


import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

# outra forma de ver a mesma análise
# sns.pairplot(tabela)
# plt.show()


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[3]:


from sklearn.model_selection import train_test_split

y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[5]:


from sklearn import metrics

# criar as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))  


# #### Visualização Gráfica das Previsões

# In[17]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Como fazer uma nova previsao?

# In[7]:


# Como fazer uma nova previsao
# importar a nova_tabela com o pandas (a nova tabela tem que ter os dados de TV, Radio e Jornal)
# previsao = modelo_randomforest.predict(nova_tabela)
# print(previsao)
nova_tabela = pd.read_csv("novos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)


# #### Qual a importância de cada variável para as vendas?

# In[19]:


sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())


# In[13]:


import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)


# In[23]:


#Correlação das informações / explorar as informações, olhar e ver como estão funcionando, acontecendo. Vamos calcular a Correlação.
#Conceito: O numero vai de zero a 1. E ela diz o quão correlacionado estão duas coisas, o quão caminhando na mesma direção
# Se estão caminhando juntos, a correlação é 1. Quando um aumentar, o outro também tem que aumentar. 
# A correlação, quando há discrepancia o numero é quase 1, tipo 0,9. 
# Quando a correlação estao com ordem de grandeza muito perto, é mais proximo de 0
# A correlação vai de -1 até o 1

#código para o print da correlação
import matplotlib.pyplot as plt
import seaborn as sns

#criar um gráfico
sns.heatmap(tabela.corr(), annot=True, cmap="Greens")

print()
#exibir o gráfico
plt.show()

# Importante sempre separar uma parte dos dados para testar e não todos para evitar viciar 
# fazendo a correlação, percebemos que a TV é muito mais expressivo em investimentos e vendas


# In[ ]:


#base com data, como fazer correlação mensal
# pega a data e transforma de acordo com o mês ou dias, sazonalidade, comportamento ao longo dos anos


# In[ ]:


#criar um gráfico com python da tabela acima

#utilizamos plotly
#existe também o matplotlib
# seaborn 

#!pip install matplotlib / plt / outros códigos estaram assim com esse apelido
#!pip install seaborn /; sns / ---> idem
#!pip install sciki


# In[ ]:


#Ai - Passar informações de uma base de dados para ela aprender uma previsão
#Vamos fazer duas divisões na base de dados! 
# usar X e Y ---> Y é quem eu quero fazer a previsão (vendas, radio ou tv?) Y é a coluna verde, X é o resto todo que
# eu quero usar para fazer a previsão

#Mas precisaremos fazer outra divisão na base de dados: dividir em 2 pedaços, um pedaço para treinar, aprender 
# e outro pedaço de verificação: teste

from sklearn.model_selection import train_test_split



y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# In[24]:


# Criando a inteligencia Artificial
# Regressão Linear e Arvores de Decisão
# Modelos de Inteligencia acima. 

#Importar a inteligencia artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

#Machine Learning, Processo de aprendizado de maquina, dentro do .fit para aprender

#one hot encode para transformar um texto em numero e depois voltar (ferramenta)


# In[26]:


#Testar a Inteligencia - 
#Como funciona? Como um gráfico cartesiano, onde no seu espaço, tem varias informações lineares (regressão linear)
# E depois tenta traçar a melhor reta que representa as informações

#Já o modelo de arvore de decisão é diferente
# Que são em formas de perguntas, Tal coisa ou tal coisa, homem ou mulher, ou seja, ele começa a fazer perguntas para sua
#base de dados

#Ja no nosso exemplo, O valor de Radio é menor que 10? o Valor de Jornal é maior do que 1, e na arvore de decisão
#Vai dizer se Sim ou se Não. 
# existe outras ferramentas, rede neural entre outras

from sklearn.metrics import r2_score

# ou usar
#from sklearn import metrics <--- apenas

# criar as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#r2 diz quanto que o nosso modelo consegue explicar esse modelo e o quanto esta proximo do nosso modelo de teste. 
#Ex: 95% , 98%? quanto maior melhor. 

# quando fazemos uma previsão, ela pode sair viciada. overfitting -> voce treinou tanto ela, que ela ficou sobreajustada
#Não esta conseguindo mais aprender. E o que nos diz se esta viciada é o (r2_score),os dados de testes tem 30%
# Um numero bacana de 70% e 30% para testes

# essa previsão é boa? Depende! Chegamos em 96% de precisão, dá para melhorar, desde que você pegue uma quantidade de dados
# Pegar uma quantidade de dados maior. 
# comparar os modelos

# Agora se fosse na Area de saúde a analise tem que ser mais detalhada, porque 90 e 96% é muito pouco
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao)) 



# In[27]:


#Previsão gráfica
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear
display(tabela_auxiliar)


# In[28]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# In[29]:


# Como fazer uma nova previsao
# importar a nova_tabela com o pandas (a nova tabela tem que ter os dados de TV, Radio e Jornal)
# previsao = modelo_randomforest.predict(nova_tabela)
# print(previsao)
nova_tabela = pd.read_csv("novos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)


# In[30]:


previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)


# In[31]:


sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())


# In[ ]:




