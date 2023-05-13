import sqlite3
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# Função de fitness
def fitness_function(individual, X_train, y_train, X_test, y_test):
    print(individual[0])
    print(individual[1])
    print(individual[2])
    param = {
        'max_depth': individual[0],
        'eta': individual[1],
        'objective': 'multi:softprob',
        'num_class': 3,
        'gpu_id': 0,
        'tree_method': 'gpu_hist',
        'subsample': 1 ,
        'min_child_weight': individual[2]
    }
    epochs = 500

    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test)

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = [np.argmax(z) for z in predictions]

    acc = accuracy_score(y_test, y)
    print(f"{acc}%")
    model.__del__()
    return acc

# Função de seleção
def selection(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        max_fitness_index = np.argmax(fitnesses)
        parents.append(population[max_fitness_index])
        fitnesses[max_fitness_index] = -1
    return parents

# Função de crossover
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        offspring.append([
            random.choice([parent1[0], parent2[0]]),
            random.choice([parent1[1], parent2[1]]),
            random.choice([parent1[2], parent2[2]]),
        ])
    return offspring

# Função de mutação
def mutation(offspring, mutation_rate):
    for child in offspring:
        if random.random() < mutation_rate:
            child[random.randint(0, 2)] = random.choice([3, 4, 5])
    return offspring

# Carregando dados
dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()
OU = data['OU-Cover']
data.drop(['Score', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover'], axis=1,
          inplace=True)
data = data.values
data = data.astype(float)

x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.20)

# Parâmetros do algoritmo genético
population_size = 20
num_generations = 10
num_parents = 5
mutation_rate = 0.2

# Inicialização da populaçãocd 
population = [[random.randint(3, 10), random.uniform(0.01, 0.1), random.randint(1, 10)]
              for _ in range(population_size)]

for generation in tqdm(range(num_generations), desc="Processando", ncols=80, leave=False):
    # Avaliando a aptidão da população
    fitnesses = [fitness_function(individual, x_train, y_train, x_test, y_test) for individual in population]

    # Selecionando os melhores indivíduos como pais
    parents = selection(population, fitnesses, num_parents)

    # Gerando a prole por crossover
    offspring = crossover(parents, population_size - num_parents)

    # Aplicando a mutação à prole
    mutated_offspring = mutation(offspring, mutation_rate)

    # Atualizando a população para a próxima geração
    population = parents + mutated_offspring

# Encontrando o melhor indivíduo após todas as gerações
best_individual = max(population, key=lambda ind: fitness_function(ind, x_train, y_train, x_test, y_test))
best_fitness = fitness_function(best_individual, x_train, y_train, x_test, y_test)

print("Melhor indivíduo:", best_individual)
print("Melhor aptidão:", best_fitness)

# Salvando o modelo com os melhores parâmetros
best_param = {
    'max_depth': best_individual[0],
    'eta': best_individual[1],
    'objective': 'multi:softprob',
    'num_class': 3,
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    'subsample': 1 ,
    'min_child_weight': best_individual[2]
}
epochs = 1000
train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test)

model = xgb.train(best_param, train, epochs)
predictions = model.predict(test)
y = [np.argmax(z) for z in predictions]
acc = round(accuracy_score(y_test, y) * 100, 1)

print(f"{acc}%")
model.save_model(f'../../Models/XGBoost_{acc}%_UO-6.json')

