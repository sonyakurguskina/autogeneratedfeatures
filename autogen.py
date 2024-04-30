# импорт пакетов
import numpy as np
import pandas as pd
import h2o

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from h2o.estimators import H2ORandomForestEstimator



import matplotlib.pyplot as plt
plt.style.use('ggplot')

# чтение данных
df = pd.read_csv('datasets/sberbank.csv')
print(df.shape)
print("types", df.dtypes.unique())
print(df)

df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Преобразование вещественных значений в целые числа
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].fillna(0).astype(int)
for col in df_non_numeric.columns:
    # Проверка типа данных столбца
    if df_non_numeric[col].dtype == 'O':  # Проверка, является ли тип данных 'timestamp'
        try:
            pd.to_datetime(df_non_numeric[col])
            # Извлечение признаков из временных меток
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df[col + '_hour'] = df[col].dt.hour
            df[col + '_minute'] = df[col].dt.minute
            # Удаление столбца timestamp
            df.drop(columns=[col], inplace=True)
        except:
            pass

print(df.shape)
print(df)

# selector = VarianceThreshold(threshold=0.4)
# selected_features = selector.fit_transform(df)
#
# # Получите индексы выбранных признаков
# selected_indices = selector.get_support(indices=True)
#
# # Создайте DataFrame с выбранными признаками
# df_selected = pd.DataFrame(selected_features, columns=df.columns[selected_indices])
#
# # Выведите наиболее полезные признаки
# print(df_selected.head())
# print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=42)
h2o.init(nthreads=-1, max_mem_size=8)
training = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
validation = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
training.describe()
# Удаление столбцов с типом данных timestamp
df = df.select_dtypes(exclude=['datetime'])

# Преобразование категориальных признаков в числовые значения
df = pd.get_dummies(df)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=42)

# Инициализация H2O
h2o.init(nthreads=-1, max_mem_size=8)

# Преобразование данных во фреймы H2O
training = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
validation = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# Указание целевой переменной
y = 'Churn'

# Указание всех переменных, кроме целевой
X = training.columns
X.remove(y)

# Преобразование целевой переменной в факторный тип для обучающего набора данных
training[y] = training[y].asfactor()

# Преобразование целевой переменной в факторный тип для валидационного набора данных
validation[y] = validation[y].asfactor()

# Обучение модели
rf = H2ORandomForestEstimator(model_id='tutorial1', ntrees=800, seed=152)
rf.train(X, y, training_frame=training, validation_frame=validation)

# Оценка модели
print(rf.auc(valid=True))


forest = RandomForestClassifier(n_estimators=800, random_state=152, n_jobs=-1)
forest.fit(X_train, y_train)
print('AUC для sklearn RandomForestClassifier: {:.4f}'.format(roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])))

X = training.columns
X.remove('Churn')
y = 'Churn'

rf1 = H2ORandomForestEstimator(model_id='tutorial1', ntrees=800, seed=152)
rf1.train(X, y, training_frame=training, validation_frame=validation)
print(rf1)
print(rf1.auc(valid=True))
