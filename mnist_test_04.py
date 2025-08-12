# Тестирование полигармонического каскада на датасете MNIST
# Тест 4
# Обычное каскада из 500 слоев 784-100-25-25-...-25-25-10 на 10 эпохах
# Цель: проверка работоспособности каскада из большого количества пакетов, 
# способности обучаться по тому же самому алгоритму, как и для каскада меньшего размера

import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
import numpy as np

import collective


# Подготовка выборок

data_train = datasets.MNIST("", train=True, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))
data_test = datasets.MNIST("", train=False, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

X_train = data_train.data
X_train = X_train.reshape(60000, 784)

y_train = data_train.targets

X_test = data_test.data
X_test = X_test.reshape(10000,784)

y_test = data_test.targets


# Создание модели

# Настройки
type = "float"    # тип float
mode = "gpu"      # выбрана видеокарта
batch = 2000      # батч из 2000 векторов (30 шагов обучения на эпоху)
epoh = 10         # 10 эпох
func = "class"    # выбрана задача классификации

alpha = 2000      # коэффициент "мягкости" решения уравнений

schema = 25*np.ones(501, dtype=np.int64)
schema[0] = 784
schema[1] = 100
schema[-1] = 10

start = dt.datetime.now()

# Инициализация и обучение
# Создать модель (полигармонический каскад)
pc = collective.Collective(schema, type, mode, func)

# Обучить модель
erl, erv = pc.Disciplina(X_train, y_train, X_test, y_test, batch, epoh, alpha)

print(f'Создание модели: {dt.datetime.now() - start}')

# Делать прогнозы на тестовом множестве
start = dt.datetime.now()

# исходный вариант обработки
# y_pred = pc.Flamma(X_test)

# Обработка с разделением на части для экономии видеопамяти:
part = 1000
y_pred = pc.FlammaPars(X_test, part)

# график обучения
plt.plot(erl, color='b')
plt.plot(erv, color='r')
plt.grid(True)
plt.xlabel('эпохи')
plt.ylabel('точность')
plt.legend(['точность train','точность test'], loc=4)

plt.show()

# Вычислить точность классификатора (если отобрать лучший вариант для каждого класса)
accuracy = accuracy_score(data_test.targets, y_pred)
print("Итоговая проверка точности:", accuracy)
print(f'Обработка данных: {dt.datetime.now() - start}')