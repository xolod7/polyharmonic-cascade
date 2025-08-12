# Код для предварительного локального сохранения выборки по датасету Epsilon (вызывается однократно)
# Позволяет в дальнейшем быстро загружать данные (в виде Numpy массивов)
import numpy as np
from sklearn.datasets import fetch_openml

# Загрузка датасета
epsilon = fetch_openml(name="epsilon", version=1)
X = epsilon.data.values.astype(np.float32)
y = epsilon.target

# Преобразование категорий в числа: {-1, +1} → {0, 1}
# (в таком формате требуется для модуля обучения при задаче классификации)
# (при бинарной классификации внутри модуля будет снова преобразование в {-1, +1})
y = y.cat.codes.astype(np.float32)  
y = np.array(y)

# Локальное сохранение в виде Numpy массивов epsilon_X.npy и epsilon_y.npy
np.save("epsilon_X.npy",X)
np.save("epsilon_y.npy",y)