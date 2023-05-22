import numpy as np
import streamlit as st
import pandas as pd
from dython.nominal import associations
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def load_data():
    """
    Загрузка данных
    """
    return pd.read_csv("../datasets/Hotel Reservations.csv")


def remove_outliers(df, label):
    """
    Очистка выбросов в заданной колонке
    """
    q1 = df[label].quantile(0.25)
    q3 = df[label].quantile(0.75)
    iqr = (q3 - q1)
    fil = (df[label] >= q1 - 1.5 * iqr) & (df[label] <= q3 + 1.5 * iqr)
    return df.loc[fil]


@st.cache_data
def preprocess_data(data_in):
    """
    Обработка данных, возвращаем матрицу объекты-признаки X и столбец целевого признака y
    """
    data_out = data_in.copy()

    # Удаление колонок
    cols_to_drop = [
        "Booking_ID",
        "arrival_year",
        "arrival_month"
    ]
    data_out.drop(cols_to_drop, axis=1, inplace=True)

    # Категориальные колонки
    cat_features = []
    for col in data_out.columns:
        dt = str(data_out[col].dtype)
        if dt == "object":
            cat_features.append(col)

    # Обработка выбросов
    col_to_remove_outliers_in = [
        "avg_price_per_room",
        "lead_time"
    ]
    for col in col_to_remove_outliers_in:
        data_out = remove_outliers(data_out, col)

    # Числовые колонки для масштабирования

    # Кодирование признаков
    data_out["booking_status"] = LabelEncoder().fit_transform(data_out["booking_status"])
    cat_features.remove("booking_status")
    data_out = pd.get_dummies(data=data_out, columns=cat_features)

    # this line for test purpose only
    data_out = data_out[::30]

    # Результат работы функции
    return data_out, cat_features, data_out.drop("booking_status", axis=1).values, data_out["booking_status"].values


def main():
    st.sidebar.header('Метод опорных векторов')
    data, nominal_features, data_X, data_y = preprocess_data(load_data())
    svc_kernel = st.sidebar.radio('Ядро:', ('sigmoid', 'rbf', 'poly'))
    poly_degree_slider = st.sidebar.slider(
        'Степень для ядра "poly":',
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )
    num_folds_slider = st.sidebar.slider(
        'Количество фолдов при кросс-валидации":',
        min_value=2,
        max_value=20,
        value=2,
        step=1
    )
    if st.checkbox('Показать корреляционную матрицу'):
        res_corr = associations(
            dataset=data,
            nominal_columns=nominal_features,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            title='Матрица корреляции по всем признакам',
            clustering=True,
            figsize=(28, 28),
            plot=False,
            compute_only=True,
            mark_columns=True,
        )
        corr_matrix = res_corr['corr']
        fig, ax = plt.subplots(figsize=(32, 32))
        sns.heatmap(
            data=corr_matrix,
            square=True,
            center=0,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            annot=True,
            vmin=-1.0,
            vmax=1.0
        )
        st.pyplot(fig)

    # Количество строк и столбцов датасета
    num_rows = data.shape[0]
    num_cols = data.shape[1]
    st.write(f"Количество строк в датасете: {num_rows}")
    st.write(f"Количество столбцов в датасете: {num_cols}")

    # Гиперпараметры метода опорных векторов
    kernel = svc_kernel
    degree = int(poly_degree_slider)
    cv = int(num_folds_slider)

    # Обучение
    svc = SVC(kernel=kernel, degree=degree)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, random_state=9)
    svc.fit(X_train, y_train)

    # Оценка модели
    st.subheader('Оценка качества модели')
    y_pred_test_svc = svc.predict(X_test)
    y_pred_train_svc = svc.predict(X_train)
    st.subheader('На обучающей выборке:')
    st.text(classification_report(y_train, y_pred_train_svc))
    st.subheader('На тестовой выборке:')
    st.text(classification_report(y_test, y_pred_test_svc))
    scores_svc = cross_val_score(svc, data_X, data_y, cv=cv)
    st.write(f"Оценка accuracy с помощью {cv} фолдной кросс-валидации: {np.mean(scores_svc)}")

    if st.checkbox('Показать матрицу ошибок'):
        cm_svc = confusion_matrix(y_test, y_pred_test_svc, labels=svc.classes_)
        display_ = ConfusionMatrixDisplay(
            confusion_matrix=cm_svc
        )
        display_.plot()
        st.pyplot()


if __name__ == '__main__':
    main()

