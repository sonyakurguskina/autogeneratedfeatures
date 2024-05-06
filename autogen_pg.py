from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, stats, chi2_contingency
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, PowerTransformer
from sklearn.decomposition import PCA
from h2o.automl import H2OAutoML
import h2o

from config.config import Config

config = Config()


CSV = 'datasets/classification.csv'
TABLE_NAME = str(CSV).split('.csv')[0].split('/')[1].lower()
METADATA_TABLE_NAME = "metadata_" + TABLE_NAME
DATA_TABLE_NAME = "data_" + TABLE_NAME
PCA_DATA_TABLE_NAME = "pca_" + TABLE_NAME


def load_csv_to_postgres(engine):
    """
    Этот метод загружает данные из файла CSV в базу данных PostgreSQL. Он выполняет следующие шаги:
    1. Чтение CSV файла с помощью библиотеки pandas.
    2. Приведение названий колонок к нижнему регистру.
    3. Очистка датасета от выбросов в числовых признаках.
    4. Замена отсутствующих значений средними значениями.
    5. Удаление строк с пропущенными значениями в строковых столбцах и удаление дубликатов строк.
    6. Проверка наличия столбца 'id' и добавление его, если не существует.
    7. Создание метаданных и запись их в таблицу метаданных в базе данных PostgreSQL.
    8. Запись данных в таблицу данных в базе данных PostgreSQL.
    :param engine: SQLAlchemy Engine, подключение к базе данных PostgreSQL
    :return:
    """
    df = pd.read_csv(CSV, encoding='ISO-8859-1')
    df.columns = map(str.lower, df.columns)
    print(df.info)

    numeric_columns = df.select_dtypes(include=np.number).columns
    for feature in numeric_columns:
        outliers_indices = get_outliers_indices(df[feature])
        df.drop(outliers_indices, inplace=True)

    means = df[numeric_columns].mean()
    df.fillna(means, inplace=True)
    df.dropna(subset=df.select_dtypes(include=['object']).columns, inplace=True)
    df.drop_duplicates(inplace=True)

    if 'id' in df.columns:
        print("Столбец 'id' уже существует в DataFrame. Не будет создан новый столбец.")
    else:
        df['id'] = range(1, len(df) + 1)
    print(df.info)

    metadata_df = pd.DataFrame({'column_name': df.columns, 'dtype': df.dtypes})
    metadata_df['is_target'] = metadata_df['column_name'] == 'target_column_name'
    metadata_df['dtype'] = metadata_df['dtype'].astype(str)
    metadata_df.to_sql(METADATA_TABLE_NAME, engine, if_exists='replace', index=False)
    df.to_sql(DATA_TABLE_NAME, engine, if_exists='replace', index=False)


def get_outliers_indices(feature_series):
    """
    Метод для определения выбросов в данных определяет индексы строк с выбросами в указанном числовом столбце данных.
    Для этого используется метод трех сигм (3-sigma), основанный на среднем значении и стандартном отклонении.
    :param feature_series: Series, числовой столбец данных
    :return: Index, индексы строк с выбросами
    """
    mid = feature_series.mean()
    sigma = feature_series.std()
    return feature_series[(feature_series < mid - 3 * sigma) | (feature_series > mid + 3 * sigma)].index


def get_column_type(column):
    """
    Метод для определения типа данных колонки. Этот метод определяет тип данных колонки на основе ее содержимого.
    :param column: Series, столбец данных
    :return:  str, тип данных колонки
    """
    if pd.api.types.is_integer_dtype(column):
        return 'INT'
    elif pd.api.types.is_float_dtype(column):
        return 'FLOAT'
    elif pd.api.types.is_string_dtype(column):
        return 'VARCHAR'
    elif pd.api.types.is_datetime64_any_dtype(column):
        return 'DATE'
    else:
        return 'VARCHAR'


def find_most_correlated_feature(data_df):
    """
    Корреляционный анализ для признаков.
    Этот метод находит пару наиболее коррелирующих между собой числовых признаков в DataFrame.
    Для этого используется корреляционный коэффициент Пирсона.
    :param data_df: DataFrame, набор данных
    :return: tuple, пара признаков с наибольшей корреляцией и их корреляционный коэффициент
    """
    max_corr = 0
    most_correlated_feature = None
    numerical_features = data_df.select_dtypes(include=['float64', 'int64']).columns

    if len(data_df) > 5000:
        data_df = data_df.sample(n=5000)

    for feature1, feature2 in combinations(numerical_features, 2):
        corr = np.corrcoef(data_df[feature1], data_df[feature2])[0, 1]
        if abs(corr) > abs(max_corr):
            max_corr = corr
            most_correlated_feature = (feature1, feature2)
    print("Корреляционный анализ:", most_correlated_feature, max_corr)
    return most_correlated_feature, max_corr


def perform_anova(data_df):
    """
    Анализ ANOVA для признаков.
    Этот метод находит пару признаков с наибольшим значением F-статистики при проведении однофакторного дисперсионного
    анализа (ANOVA). ANOVA используется для оценки различий между средними значениями групп.
    :param data_df: DataFrame, набор данных
    :return: tuple, пара признаков с наибольшим значением F-статистики и само значение F-статистики
    """
    numerical_features = data_df.select_dtypes(include=['float64', 'int64']).columns
    max_f_value = 0
    most_correlated_features = None

    if len(data_df) > 5000:
        data_df = data_df.sample(n=5000)

    for feature1, feature2 in combinations(numerical_features, 2):
        f_value, p_value = f_oneway(data_df[feature1], data_df[feature2])
        if f_value > max_f_value:
            max_f_value = f_value
            most_correlated_features = (feature1, feature2)
    print("ANOVA:", most_correlated_features, max_f_value)
    return most_correlated_features, max_f_value


def compare_results(data_df):
    """
    Сравнение результатов анализа для выбора наиболее предпочтительного метода. Этот метод сравнивает результаты
    корреляционного анализа и анализа ANOVA, а также анализа V-значений (V-value analysis) для определения наиболее
    важных признаков. Возвращает признак, который был выбран в качестве наиболее предпочтительного.
    :param data_df: DataFrame, набор данных для сравнения результатов анализа
    :return: str, наиболее предпочтительный признак для использования в дальнейшем анализе
    """
    try:
        corr_result = find_most_correlated_feature(data_df)
        anova_result = perform_anova(data_df)
        corr_feature, corr_value = corr_result
        anova_feature, anova_value = anova_result
        gen_feature = None

        if corr_feature is None and anova_feature is None:
            gen_feature = find_most_correlated_categorical_feature(data_df)
            return gen_feature[0]

        for feature in corr_feature:
            if feature in anova_feature:
                gen_feature = feature
        if abs(corr_value) > abs(anova_value):
            print("Корреляционный анализ предпочтительнее")
            if gen_feature:
                return gen_feature
            else:
                return corr_feature[0]
        elif abs(anova_value) > abs(corr_value):
            print("ANOVA предпочтительнее")
            if gen_feature:
                return gen_feature
            else:
                return anova_feature[0]
        else:
            print("Результаты корреляционного анализа и ANOVA примерно равны")
            if gen_feature:
                return gen_feature
            else:
                return corr_feature[0]
    except Exception as e:
        print(f"Произошла ошибка при определении наилучшего результата анализа целевой переменной: {e}")


def find_most_correlated_categorical_feature(data_df):
    """
    Анализ V-значений для определения наиболее коррелированных категориальных признаков в данных.
    Этот метод использует анализ V-значений для определения пары категориальных признаков с наибольшей корреляцией.
    1. Сначала функция выбирает небольшую случайную выборку из данных для эффективности вычислений.
    2. Затем она проходит по всем возможным парам категориальных признаков.
    3. Для каждой пары признаков функция вычисляет таблицу сопряженности и значение хи-квадрат.
    4. На основе значения хи-квадрат функция вычисляет V-значение.
    5. Находится пара признаков с наибольшим V-значением, которая указывает на наибольшую взаимосвязь между ними.
    :param data_df: DataFrame, набор данных для анализа
    :return: tuple, пара наиболее коррелированных категориальных признаков и их V-значение
    """
    data_sample = data_df.sample(frac=0.1, random_state=42)

    max_v_value = 0
    most_correlated_feature = None
    categorical_features = data_sample.select_dtypes(include=['object']).columns

    if 'y' in categorical_features:
        print("Target:", 'y', 1)
        return 'y', 1
    else:
        for feature1 in categorical_features:
            for feature2 in categorical_features:
                if feature1 != feature2:
                    contingency_table = pd.crosstab(data_sample[feature1], data_sample[feature2])
                    chi2_value, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    v_value = np.sqrt(chi2_value / (n * min(contingency_table.shape) - 1))
                    if v_value > max_v_value:
                        max_v_value = v_value
                        most_correlated_feature = (feature1, feature2)
        print("V-value analysis:", most_correlated_feature, max_v_value)
        return most_correlated_feature, max_v_value


def get_data_df(engine):
    """
    Получение данных из базы данных.
    Этот метод выполняет запрос к базе данных и возвращает данные из таблицы данных.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :return: DataFrame, набор данных из таблицы данных
    """
    data_query = f"SELECT * FROM {DATA_TABLE_NAME}"
    return pd.read_sql(data_query, engine)


def get_pca_data_df(engine):
    """
    Получение наиболее важных признаков из базы данных.
    Этот метод выполняет запрос к базе данных и возвращает данные из таблицы PCA_DATA_TABLE_NAME,
    содержащей наиболее важные признаки.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :return: DataFrame, набор данных из таблицы PCA_DATA_TABLE_NAME
    """
    data_query = f"SELECT * FROM {PCA_DATA_TABLE_NAME}"
    return pd.read_sql(data_query, engine)


def set_target_feature(engine):
    """
    Установка целевого признака в базе данных.
    Этот метод определяет предпочтительный признак для использования в качестве целевого и устанавливает его
    в метаданных базы данных.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    data_df = get_data_df(engine)
    preferred_feature = compare_results(data_df)
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text(
                    f"UPDATE {METADATA_TABLE_NAME} SET is_target = True WHERE column_name = '{preferred_feature}'"))
                trans.commit()
            except:
                trans.rollback()
                raise
        print(f"Целевая переменная обновлена: {preferred_feature} установлена в True")
    except Exception as e:
        print("Ошибка при обновлении базы данных:", e)


def get_target_column(engine):
    """
    Получение информации о целевой переменной из базы данных.
    Этот метод выполняет запрос к базе данных и возвращает информацию о целевой переменной,
    включая ее имя и тип данных.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :return: tuple, содержащий имя и тип данных целевой переменной
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT column_name, dtype FROM {METADATA_TABLE_NAME} WHERE is_target = true"))
            target_info = result.fetchone()
            if target_info:
                target_column, target_type = target_info
                return target_column, target_type
            else:
                print("Целевая переменная не найдена в базе данных.")
                return None, None
    except Exception as e:
        print("Ошибка при получении целевой переменной из базы данных:", e)
        return None, None


def get_data_column(engine, target_column):
    """
    Получение данных по столбцу из базы данных.
    Этот метод выполняет запрос к базе данных для выбора данных из указанного столбца таблицы данных.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :param target_column: str, имя столбца данных, который нужно выбрать
    :return: list, значения из указанного столбца
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT {target_column} FROM {DATA_TABLE_NAME}"))
            data_column = [row[0] for row in result.fetchall()]
            return data_column
    except Exception as e:
        print("Ошибка при получении данных из базы данных:", e)
        return None


def determine_task_type(engine):
    """
    Определение типа задачи машинного обучения на основе данных.
    Этот метод анализирует данные в базе данных и определяет тип задачи машинного обучения: классификация или регрессия.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    target_column, target_type = get_target_column(engine)
    target_data = get_data_column(engine, target_column)
    data_df = get_data_df(engine)
    if target_column:
        if target_type == 'object':
            print('classification')
            process_classification_data(engine, data_df)
        elif target_type in ['int64', 'float64']:
            # если целевая переменная числовая, проводим статистический тест для определения распределения
            p_value = stats.normaltest(target_data)[1]
            if p_value < 0.05:
                # если p-value меньше уровня значимости, то распределение не является нормальным, значит это задача
                # регрессии
                print('regression')
                process_regression_data(engine)
            else:
                print('classification')
                process_classification_data(engine, data_df)
        else:
            print('Задача не относится ни к регрессии, ни к классификации')
            return None


def process_classification_data(engine, data_df):
    """
    Обработка данных для задачи классификации.
    Этот метод выполняет необходимую обработку данных для решения задачи классификации,
    включая кодирование категориальных признаков и нормализацию числовых признаков.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :param data_df: DataFrame, набор данных для обработки
    """
    categorical_columns = data_df.select_dtypes(include=['object']).columns
    if 'id' in categorical_columns:
        categorical_columns = categorical_columns.drop('id')

    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data_df[col] = label_encoders[col].fit_transform(data_df[col])

    scaler = StandardScaler()
    columns_to_scale = [col for col in data_df.columns if col != 'id']
    data_df[columns_to_scale] = scaler.fit_transform(data_df[columns_to_scale])

    update_database_with_processed_data(engine, data_df, categorical_columns)


def process_regression_data(engine):
    """
    Обработка данных для задачи регрессии.
    Этот метод выполняет необходимую обработку данных для решения задачи регрессии,
    включая нормализацию числовых признаков.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    data_df = get_data_df(engine)
    numerical_features = data_df.select_dtypes(include=['int64', 'float64']).columns

    if 'id' in numerical_features:
        numerical_features = numerical_features.drop('id')

    scaler = StandardScaler()
    data_df_scaled = scaler.fit_transform(data_df[numerical_features])
    data_df[numerical_features] = data_df_scaled

    data_df = data_df.drop(columns=data_df.select_dtypes(include=['object']).columns, errors='ignore')

    update_database_with_processed_data(engine, data_df, numerical_features)


def update_database_with_processed_data(engine, data_df, columns_to_update):
    """
    Обновление базы данных обработанными данными.
    Этот метод обновляет базу данных с обработанными данными, выполняя обновление каждой строки данных
    с новыми значениями из DataFrame.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :param data_df: DataFrame, содержащий обработанные данные
    :param columns_to_update: list, список столбцов, которые нужно обновить
    """
    with engine.connect() as conn:
        try:
            count = 0
            for index, row in data_df.iterrows():
                trans = conn.begin()
                set_pairs = [f"{column} = :{column}" for column in columns_to_update]
                set_clause = ', '.join(set_pairs)
                update_query = text(f"UPDATE {DATA_TABLE_NAME} SET {set_clause} WHERE id = :index;")
                values_dict = {column: row[column] for column in columns_to_update}
                values_dict['index'] = row['id']
                conn.execute(update_query, values_dict)
                trans.commit()
                count += 1
            print("Все данные успешно обновлены в базе данных.")
            print(count)
        except Exception as e:
            print(f"Произошла ошибка при обновлении данных: {str(e)}")


def generate_features(engine):
    """
    Генерация признаков и обновление базы данных.
    Этот метод генерирует новые признаки на основе текущих данных, используя полиномиальные признаки и алгоритм
    Йео-Джонсона.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    data_df = get_data_df(engine)

    numerical_features = data_df.select_dtypes(include=['int64', 'float64']).columns
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data_df[numerical_features])
    poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
    poly_data_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    transformer = PowerTransformer(method='yeo-johnson')
    transformed_data = transformer.fit_transform(poly_data_df)
    transformed_data_df = pd.DataFrame(transformed_data, columns=poly_data_df.columns)

    update_database_with_generated_features(engine, transformed_data_df)


def update_database_with_generated_features(engine, generated_data_df):
    """
    Обновление базы данных сгенерированными признаками.
    Этот метод обновляет базу данных сгенерированными признаками, объединяя их с текущими данными и добавляя
    обновленные данные в базу данных.

    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :param generated_data_df: DataFrame, содержащий сгенерированные признаки
    """
    with engine.connect() as conn:
        try:
            trans = conn.begin()
            current_data_df = pd.read_sql_table(DATA_TABLE_NAME, conn)
            updated_data_df = pd.concat([current_data_df, generated_data_df], axis=1)
            updated_data_df.to_sql(name=DATA_TABLE_NAME, con=conn, if_exists='replace', index=False)
            trans.commit()
            print("Сгенерированные признаки успешно добавлены в базу данных.")
        except Exception as e:
            trans.rollback()
            print(f"Произошла ошибка при добавлении сгенерированных признаков в базу данных: {str(e)}")


def apply_pca(engine):
    """
    Применение метода главных компонент (PCA) и обновление базы данных.
    Этот метод применяет анализ PCA к данным, обновляет базу данных с данными PCA и запускает процесс обучения модели.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    data_df = get_data_df(engine)
    data_df = data_df.drop(columns=data_df.select_dtypes(include=['object']).columns, errors='ignore')

    pca = PCA(n_components=15)
    pca_result = pca.fit_transform(data_df)

    pca_columns = [f"pca_{i + 1}" for i in range(15)]
    pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
    print(pca_df)
    update_database_with_pca(engine, pca_df)

    return pca_df


def update_database_with_pca(engine, pca_data_df):
    """
    Обновление базы данных с данными PCA.
    Этот метод добавляет данные PCA в базу данных, объединяя их с текущими данными и добавляя обновленные данные
    в базу данных.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    :param pca_data_df: DataFrame, содержащий данные PCA
    """
    with (engine.connect() as conn):
        try:
            trans = conn.begin()
            target_column = get_target_column(engine)[0]
            target_data = pd.DataFrame(get_data_column(engine, target_column), columns=[target_column])
            pca_data_df.to_sql(name=PCA_DATA_TABLE_NAME, con=conn, if_exists='replace', index=False)

            current_data_df = pd.read_sql_table(PCA_DATA_TABLE_NAME, conn)
            updated_data_df = pd.concat([current_data_df, target_data], axis=1)
            updated_data_df.to_sql(name=PCA_DATA_TABLE_NAME, con=conn, if_exists='replace', index=False)
            trans.commit()
            print("Данные PCA успешно добавлены в базу данных.")
        except Exception as e:
            trans.rollback()
            print(f"Произошла ошибка при добавлении данных PCA в базу данных: {str(e)}")


def train_automl_model(engine):
    """
    Обучение модели AutoML.
    Этот метод использует данные PCA для обучения модели AutoML и вывода информации о лучшей модели.
    :param engine: SQLAlchemy engine, используемый для соединения с базой данных
    """
    data_df = get_pca_data_df(engine)
    h2o.init()
    target_column = get_target_column(engine)[0]
    h2o_df = h2o.H2OFrame(data_df)
    x_columns = h2o_df.drop(target_column)
    train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
    aml = H2OAutoML(max_models=20, seed=42)
    aml.train(x=x_columns.columns, y=target_column, training_frame=train)
    print(aml.leaderboard, aml.leader)


def main():
    engine = create_engine(
        f"postgresql://{config.pg_user}:{config.pg_password}@{config.pg_host}:{config.pg_port}/"
        f"{config.pg_database}")

    load_csv_to_postgres(engine)

    set_target_feature(engine)

    determine_task_type(engine)

    generate_features(engine)

    apply_pca(engine)

    train_automl_model(engine)


if __name__ == "__main__":
    main()
