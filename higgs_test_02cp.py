# Тестирование полигармонического каскада на датасете Higgs Boson
# Тест 2 Обучение каскада из 20-х пакетов: 28-200-200-...-200-1 на 500 эпохах
# В отличии от теста 1 производится предварительная обработка
# выбранных признаков, чтобы сделать распределение их значений более равномерным
# версия теста с чекпоинтами 
# первый запуск теста с zapusk = True
# каждый следующий запуск производится с zapusk = False и продолжеает обучения с заданным количеством эпох
# для ранее сохраненной модели
import torch
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

import collective

# Загрузка датасета (сохраненного ранее для удобства в виде Numpy массивов)
# Чтобы можно было таким образом загрузить данные, сначала нужно выполнить higgs_load.py

X = np.load("higgs_X.npy")
y = np.load("higgs_y.npy")

# Обработка выбранных признаков, чтобы сделать распределение их значений более равномерным
id = np.array([0, 5, 9, 13, 17, 21, 22, 23, 24, 25, 26, 27])
X[:,id] = np.log(X[:,id])
X[:,3] = np.log1p(X[:,3])

y = np.reshape(y,(y.shape[0],1))

X = torch.tensor(X)
y = torch.tensor(y)
y = y.repeat(1,1)


# Разделение на train/test (
n = 10500000

X_train = X[:n,:]
y_train = y[:n,:]
X_test = X[n:,:]
y_test = y[n:,:]

#----------------------------------------------------------------------------------------------------------------
# Создание модели

# Настройки
type = "float"    # тип float
mode = "gpu"      # выбрана видеокарта
batch = 14000     # размер батча
epoh = 50         # количество эпох
func = "class"    # задача - классификация

alpha = 1000     # коэффициент "мягкости" решения уравнений

# схема модели (полигармонический каскад в 20 слоев)
schema = [28, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 1] 


start = dt.datetime.now()

# Метка о том, первый это запуск или продолжение
zapusk = True

if zapusk:
    # инициализация модели
    pc = collective.Collective(schema, type, mode, func)

    # обучение модели
    erl, erv = pc.Disciplina(X_train, y_train, X_test, y_test, batch, epoh, alpha)

    data = {"pc":pc, "erl": erl, "erv": erv}
    with open("data.pickle", "wb") as file:
        pickle.dump(data, file)    
else:
    with open("data.pickle", "rb") as file:
        data = pickle.load(file)
    pc = data["pc"]
    erl = data["erl"]
    erv = data["erv"]
    erl2, erv2 = pc.Disciplina(X_train, y_train, X_test, y_test, batch, epoh, alpha) # обучение модели
    erl.extend(erl2[1:])
    erv.extend(erv2[1:])
    data = {"pc":pc, "erl": erl, "erv": erv}
    with open("data.pickle", "wb") as file:
        pickle.dump(data, file)     

print(f'Создание модели: {dt.datetime.now() - start}')

# график обучения
plt.plot(erl, color='b')
plt.plot(erv, color='r')
plt.grid(True)
plt.xlabel('эпохи')
plt.ylabel('ROC AUC')
plt.legend(['train','test','итоговая модель test'], loc=4)

plt.show()

# Делать прогнозы на тестовом множестве
start = dt.datetime.now()

# y_pred = pc.Flamma(X_test)

# Обработка с разделением на части для экономии видеопамяти:
part = 10000
y_pred = pc.FlammaPars(X_test, part)

k = y_test.shape[0]

auc = roc_auc_score(y_test.reshape(k),y_pred.reshape(k))
print("Итоговая проверка auc:", auc)

print(f'Обработка данных: {dt.datetime.now() - start}')