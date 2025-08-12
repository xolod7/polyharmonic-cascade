# Код для предварительного локального сохранения выборки по датасету HIGGS (вызывается однократно)
# Позволяет в дальнейшем быстро загружать данные (в виде Numpy массивов)
# Для исполнения кода необходимо иметь файл HIGGS.csv.gz
# Скачать его можно по адресу: https://archive.ics.uci.edu/ml/datasets/HIGGS
import numpy as np
import pandas as pd

#путь к файлу
file_path = "HIGGS.csv.gz"

# Загрузка
dtypes = {col: "float32" for col in range(1, 29)}
dtypes[0] = "int32" 

df = pd.read_csv(file_path, header=None, dtype=dtypes)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Сохранение в виде Numpy массивов higgs_X.npy и higgs_y.npy
np.save("higgs_X.npy",X)
np.save("higgs_y.npy",y)