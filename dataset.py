import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Загрузка датасета из csv-файла
def load_data(file_path):
    """
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    return pd.read_csv(file_path)


# Информация об отсутствующих данных 
def data_miss(df):
    """
    :param df: DataFrame с загруженными данными
    :return: общая информация об отсутствующих записях
    """
    total_data = df.isna().sum().sort_values(ascending=False)
    percentage_data = ((df.isna().sum()*100)/(df.shape[0])).sort_values(ascending=False)
    table_data = pd.concat([total_data, percentage_data], axis=1, keys=['Total missing values', 'Total Percentage'])
    return print(table_data)


# Преобразование вещественных данных в числовые
def data_encode(df):
    """
    :param df: датасет, содержащий вещественные значения
    :return: датасет после преобразования вещественных данных в числовые
    """
    encoder = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])
    return df


def param_sort(df, target_param=''):
    """
    :param df: исследуемый датасет
    :param target_param: название целевого параметра в датасете
    :return: X - набор признаков, y - целевой параметр
    """
    X = df.drop(target_param, axis=1)
    y = df[target_param]
    return X, y

