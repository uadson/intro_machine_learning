# 1. Importando bibliotecas
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# 2. Carregando base de dados
IOWA_FILE_PATH = 'train.csv'

data = pd.read_csv(IOWA_FILE_PATH)


# 2.1 Imprimindo resultado
print(data.head())
print()


# 2.2 Consultando os nomes das colunas
print(data.columns)
print()


# 3. Criando um objeto alvo e chamando-o de y.
# O objeto alvo é o valor de imóveis que se deseja prever com base nos dados carregados
# coluna base para o objetivo = SalePrice
y = data.SalePrice

# imprimindo o resultado
print(y.head())
print()


# 4. Criando X
# Serão as colunas "nós", cujos valores condicionarão o valor de venda do imóvel
# São elas:
'''
	LotArea
	YearBuilt
	1stFlrSF
	2ndFlrSF
	FullBath
	BedroomAbvGr
	TotRmsAbvGrd
'''
features = 'LotArea YearBuilt 1stFlrSF 2ndFlrSF FullBath BedroomAbvGr TotRmsAbvGrd'.split()

X = data[features]

# imprimindo o resultado
print(X.head())
print()


# 5. Particionando os dados para treinamento e validação
# Parte dos dados serão utilizados para treinamento e uma outra parte será utilizada
# para validar os dados treinados, ou seja, para verificar se ao informar os dados
# externos, as predições possam chegar o mais próximo dos valores reais
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Especificando o número de random_state, faz com que se obtenha os mesmos resultados
# durante o treinamento pois se estabelece um certo padrão no treinamento.
# Caso não haja nenhum valor informado, ocorrerá uma aleatoriedade no treinamento 
# gerando assim resultados obviamente resultados aleatórios.


# 6. Especificando um modelo
data_model = DecisionTreeRegressor(random_state=1)


# 7. Ajustando o modelo
data_model.fit(train_X, train_y)


# 8. Validando as predições
val_predict = data_model.predict(val_X)


# 8.1 Calculando o Erro Médio Absoluto
val_mae = mean_absolute_error(val_predict, val_y)

# imprimindo resultados
print('Valores das Predições:\n')
for p in val_predict[0:5]:
	print(p)
print()

print('Validação MAE: {:,.0f}'.format(val_mae))
print()


# 9. Controlando os particionamentos dos dados através da função get_mae()
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
	
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
	
	model.fit(train_X, train_y)
	
	preds_val = model.predict(val_X)
	
	mae = mean_absolute_error(preds_val, val_y)
	
	return mae


# 10. Comparando os diferentes tamanhos de árvores(particionamentos, nós)
nodes = [5, 25, 50, 100, 250]

for max_leaf_nodes in nodes:

	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

	print('Máximo de folhas nos nós: {} \t\t Erro Médio Absoluto: {:.0f}'.format(max_leaf_nodes, my_mae))

print()

best_tree_size = 100

print(f'O menor valor de Erro Médio Absoluto se encontra em um tamanho de árvore = {best_tree_size}')
print()


# 11. Ajustando o modelo usando todos os dados
# Agora que temos o melhor número de folhas para o balanceamento dos dados
# vamos ajustar o modelo utilizando todos os dados e objter uma predição 
# mais precisa dos valores
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X, y)

final_pred = final_model.predict(X)

final_mae = mean_absolute_error(final_pred, y)

print()

print('Predições Anteriores:\n')
for p in val_predict[0:5]:
	print(p)
print()

print('Validação MAE: {:,.0f}'.format(val_mae))
print()


print('Novas Predições: \n')
for p in final_pred[0:5]:
	print('{:.0f}'.format(p))
print()


print('Validação MAE: {:,.0f}'.format(final_mae))
print()


print('Comparando com os valores reais: \n')
print(y.head())