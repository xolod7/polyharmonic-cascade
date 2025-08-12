# Тестирование полигармонического каскада на датасете MNIST
# Тест 1
# Обучение небольшого каскада (4 слоя) 784-100-20-20-10 на 10 эпохах

import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets

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

alpha = 200      # коэффициент "мягкости" решения уравнений

schema = [784, 100, 20, 20, 10]  # схема каскада


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