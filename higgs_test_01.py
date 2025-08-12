# Тестирование полигармонического каскада на датасете Higgs Boson
# Тест 1 Обучение каскада из 20-х пакетов: 28-200-200-...-200-1 на 500 эпохах
# Выполнение 500 эпох может занять продолжительое время
# поэтому есть аналогичная версия теста с чекпоинтами higgs_test_01cp.py
import torch
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import roc_auc_score
import numpy as np

import collective

# Загрузка датасета (сохраненного ранее для удобства в виде Numpy массивов)
# Чтобы можно было таким образом загрузить данные, сначала нужно выполнить higgs_load.py

X = np.load("higgs_X.npy")
y = np.load("higgs_y.npy")

y = np.reshape(y,(y.shape[0],1))

X = torch.tensor(X)
y = torch.tensor(y)
y = y.repeat(1,1)


# Разделение на train/test
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
epoh = 500        # количество эпох
func = "class"    # задача - классификация

alpha = 1000     # коэффициент "мягкости" решения уравнений

# схема модели (полигармонический каскад в 20 слоев)
schema = [28, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 1] 


start = dt.datetime.now()

# Инициализация модели
pc = collective.Collective(schema, type, mode, func)

# Обучение модели
erl, erv = pc.Disciplina(X_train, y_train, X_test, y_test, batch, epoh, alpha)
 

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