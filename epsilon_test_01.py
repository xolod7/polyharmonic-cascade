# Тестирование полигармонического каскада на датасете Epsilon
# Тест 1
# Обучение каскада из 4-х пакетов: 2000-3-20-20-1 на 10 эпохах

import torch
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import roc_auc_score
import numpy as np

import collective

# Загрузка датасета (сохраненного ранее для удобства в виде Numpy массивов)
# Чтобы можно было таким образом загрузить данные, сначала нужно выполнить epsilon_load.py

X = np.load("epsilon_X.npy")
y = np.load("epsilon_y.npy")

# Преобразование в тензоры
X = torch.tensor(X)
y = torch.tensor(y, dtype=torch.int32)

# Разделение на train/test (первые 400 000 примеров - train, оставшиеся 100 000 - test)
n = 400000

X_train = X[:n,:]
y_train = y[:n]
X_test = X[n:,:]
y_test = y[n:]

#----------------------------------------------------------------------------------------------------------------
# Создание модели

# Настройки
type = "float"    # тип float
mode = "gpu"      # выбрана видеокарта
batch = 2000      # размер батча
epoh = 10         # количество эпох
func = "class"    # задача - классификация

alpha = 10        # коэффициент "мягкости" решения уравнений

# схема модели 
schema = [2000, 3, 20, 20, 1]


start = dt.datetime.now()

# инициализация модели
pc = collective.Collective(schema, type, mode, func)

# обучение модели
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


# Дополнительная проверка
start = dt.datetime.now()

# y_pred = pc.Flamma(X_test)

# Обработка с разделением на части для экономии видеопамяти:
part = 10000
y_pred = pc.FlammaPars(X_test, part)

k = y_test.shape[0]
# Вычислить точность классификатора
auc = roc_auc_score(y_test.reshape(k), y_pred.reshape(k))
print("Итоговая проверка точности ROC AUC:", auc)

print(f'Обработка данных: {dt.datetime.now() - start}')