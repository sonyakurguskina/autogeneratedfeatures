import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OKMeansEstimator, H2OGradientBoostingEstimator, \
    H2OGeneralizedLinearEstimator
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, r2_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def detect_task_type(data: pd.DataFrame):
    """
    Этот метод определяет тип задачи машинного обучения на основе данных. Если все признаки являются категориальными
    или текстовыми, возвращает "classification". Если хотя бы один признак числовой и количество уникальных значений
    в числовых признаках менее 5% от общего числа наблюдений, возвращает "clustering". В противном случае,
    если последний столбец в данных является числовым, возвращает "regression". В противном случае возвращает
    "classification"
    :param data: DataFrame, набор данных для анализа
    :return: str, тип задачи машинного обучения: 'classification', 'clustering' или 'regression'
    """
    feature_types = data.dtypes.unique()
    if all([isinstance(dtype, pd.CategoricalDtype) or dtype == 'object' for dtype in feature_types]):
        return 'classification'
    if any([pd.api.types.is_numeric_dtype(dtype) for dtype in feature_types]):
        unique_counts = data.select_dtypes(include=['int', 'float']).nunique()
        if unique_counts.min() < len(data) * 0.05:
            return 'clustering'
    target = data.select_dtypes(include=['int', 'float']).columns[
        -1] if 'int' in feature_types or 'float' in feature_types else None
    if target and not isinstance(data[target], pd.CategoricalDtype):
        return 'regression'
    return 'classification'


def determine_target_variable(data: pd.DataFrame, task_type: str):
    """
    Этот метод определяет целевую переменную для модели в зависимости от типа задачи машинного обучения.
    Если задача классификации или регрессии, возвращает название последнего столбца данных. Если задача кластеризации,
    возвращает None.
    :param data: DataFrame, набор данных для анализа
    :param task_type: str, тип задачи машинного обучения: 'classification', 'clustering' или 'regression'
    :return: str or None, название целевой переменной или None для задачи кластеризации
    """
    if task_type == 'classification' or task_type == 'regression':
        return data.columns[-1]
    elif task_type == 'clustering':
        return None
    else:
        raise ValueError("Unsupported machine learning task type")


def preprocess_data(data):
    """
    Этот метод предварительно обрабатывает числовые признаки в данных, заполняя пропущенные значения исходя из их
    среднего значения.
    :param data: DataFrame, набор данных для анализа
    :return: DataFrame, предварительно обработанные числовые признаки
    """

    data_numeric = data.select_dtypes(include=[np.number])
    data_numeric.fillna(data_numeric.mean(), inplace=True)
    return data_numeric


def generate_features(data):
    """
    Этот метод генерирует новые признаки на основе числовых данных, вычисляя среднее значение, медиану и
    стандартное отклонение для каждого числового столбца.
    :param data: DataFrame, набор данных для анализа
    :return: DataFrame, новые сгенерированные признаки
    """

    # Example: Calculate mean, median, and standard deviation for each numerical column
    aggregated_features = data.select_dtypes(include=[np.number]).agg(['mean', 'median', 'std']).transpose()
    return aggregated_features


def train_models(X_train, y_train, X_test, y_test, models):
    """
    Этот метод обучает модели машинного обучения на заданных обучающих данных и оценивает их производительность на
    тестовых данных. Выводит ROC-кривую и матрицу ошибок для задачи классификации, а также график регрессии для
    задачи регрессии.
    :param X_train: DataFrame, обучающие признаки
    :param y_train: Series, целевая переменная для обучения
    :param X_test: DataFrame, тестовые признаки
    :param y_test: Series, целевая переменная для тестирования
    :param models: list, список моделей для обучения и оценки
    :return: None
    """

    for name, model in models:
        if "H2O" in name:
            h2o.init(nthreads=-1, max_mem_size=8)
            training = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
            validation = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            for col in training.columns:
                training[col] = training[col].asnumeric()
            for col in validation.columns:
                validation[col] = validation[col].asnumeric()
            model.train(x=X_train.columns.tolist(), y=target_variable, training_frame=training,
                        validation_frame=validation)
            print(f'{name} R^2: {model.r2(valid=True)}')
            if task_type == 'classification':
                plot_roc_curve(y_test, model.predict(validation).as_data_frame().values.flatten())
            elif task_type == 'regression':
                plot_regression_results(y_test, model.predict(validation).as_data_frame().values.flatten())
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                plot_roc_curve(y_test, y_pred)
                plot_confusion_matrix(y_test, y_pred)
            elif task_type == 'regression':
                plot_regression_results(y_test, y_pred)


def plot_roc_curve(y_true, y_pred):
    """
    Этот метод строит ROC-кривую для задачи классификации на основе истинных и предсказанных меток.
    :param y_true: Series, истинные метки
    :param y_pred: Series, предсказанные метки
    :return: None
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Этот метод строит матрицу ошибок для задачи классификации на основе истинных и предсказанных меток.
    :param y_true: Series, истинные метки
    :param y_pred: Series, предсказанные метки
    :return: None
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_regression_results(y_true, y_pred):
    """
    Этот метод строит график регрессии для задачи регрессии на основе истинных и предсказанных значений.
    :param y_true: Series, истинные значения
    :param y_pred: Series, предсказанные значения
    :return: None
    """

    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()


def perform_clustering(df_numeric):
    """
    Этот метод выполняет кластеризацию на числовых данных с помощью алгоритма K-means. Выводит оценку силуэта и
    визуализирует кластеры с использованием метода главных компонент.
    :param df_numeric: DataFrame, числовые данные для кластеризации
    :return: None
    """

    h2o.init()
    h2o_df = h2o.H2OFrame(df_numeric)
    numeric_columns = df_numeric.columns.tolist()
    kmeans = H2OKMeansEstimator(k=3, seed=42)
    kmeans.train(x=numeric_columns, training_frame=h2o_df)
    predictions = kmeans.predict(h2o_df)
    silhouette = silhouette_score(df_numeric, predictions.as_data_frame()["predict"].values)
    print(f'Silhouette score for K-means in H2O: {silhouette}')

    # Visualize clusters
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[numeric_columns])
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = predictions.as_data_frame()['predict'].values
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'], cmap='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    h2o.cluster().shutdown()


# Load data
df = pd.read_csv('datasets/lintrain.csv')
task_type = detect_task_type(df)
print("Machine Learning Task Type:", task_type)
target_variable = determine_target_variable(df, task_type)
print("Target Variable:", target_variable)

# Preprocess the data
df_numeric = preprocess_data(df)

# Generate new features
new_features = generate_features(df_numeric)
print("Generated Features:")
print(new_features)

if target_variable:
    # If classification or regression task, prepare the data
    df.dropna(subset=[target_variable], inplace=True)
    # Разбиение данных на обучающие и тестовые наборы
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_variable]), df[target_variable],
                                                        test_size=0.4, random_state=42)

    # List of models for training
    models = [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=800, random_state=152, n_jobs=-1)),
        ("H2ORandomForestEstimator", H2ORandomForestEstimator(model_id='rf_model', ntrees=800, seed=152)),
        ("H2OGradientBoostingEstimator", H2OGradientBoostingEstimator(model_id='gbm_model', ntrees=800, seed=152)),
        ("H2OGeneralizedLinearEstimator", H2OGeneralizedLinearEstimator(model_id='glm_model', seed=152))
    ]

    # Train models
    train_models(X_train, y_train, X_test, y_test, models)

else:
    if task_type == "clustering":
        perform_clustering(df_numeric)
    else:
        print("Unsupported machine learning task type")
