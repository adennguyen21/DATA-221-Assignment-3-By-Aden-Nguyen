# Question 4 - Training a K-Nearest Neighbors classifier.
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def convert_csv_as_dataframe(kidney_file):
    # Converts the kidney csv as a dataframe.
    kidney_dataframe = pd.read_csv(kidney_file)

    return kidney_dataframe


def clean_dataframe(kidney_dataframe):
    # Copilot helped me with this function.
    kidney_dataframe.replace("?", pd.NA)

    numeric_columns = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    for column in numeric_columns:
        kidney_dataframe[column] = pd.to_numeric(kidney_dataframe[column], errors='coerce')

    kidney_dataframe[numeric_columns] = kidney_dataframe[numeric_columns].fillna(kidney_dataframe[numeric_columns].median())

    categorical_columns = kidney_dataframe.select_dtypes(include=["object", "string"]).columns
    kidney_dataframe[categorical_columns] = kidney_dataframe[categorical_columns].fillna(
        kidney_dataframe[categorical_columns].mode().iloc[0])

    kidney_dataframe_encoded = pd.get_dummies(kidney_dataframe, drop_first=True)

    return kidney_dataframe_encoded


def create_matrix_x(kidney_dataframe):
    kidney_matrix_x = kidney_dataframe.drop("classification_notckd", axis = 1)

    return kidney_matrix_x


def create_vector_y(kidney_dataframe):
    kidney_vector_y = kidney_dataframe["classification_notckd"]

    return kidney_vector_y


def split_dataset(matrix_x, vector_y):
    kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test = train_test_split(matrix_x, vector_y, test_size = 0.30, random_state = 42)


    return kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test


def define_k_nearest_neighbor_model():
    number_of_neighbors = 5
    k_nearest_neighbors_model = KNeighborsClassifier(n_neighbors = number_of_neighbors)

    return k_nearest_neighbors_model


def train_k_nearest_neighbor_model(knn_model, x_train, y_train):
    trained_knn_model = knn_model.fit(x_train, y_train)

    return trained_knn_model


def predict_labels_for_test_set(trained_knn_model, x_test):
    y_predicted_labels = trained_knn_model.predict(x_test)

    return y_predicted_labels


# =================================================================

def main():
    kidney_disease_csv_file = "kidney_disease.csv"

    kidney_disease_dataframe = convert_csv_as_dataframe(kidney_disease_csv_file)

    cleaned_kidney_dataframe = clean_dataframe(kidney_disease_dataframe)

    kidney_matrix_x = create_matrix_x(cleaned_kidney_dataframe)

    kidney_vector_y = create_vector_y(cleaned_kidney_dataframe)

    kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test = split_dataset(kidney_matrix_x, kidney_vector_y)

    knn_model = define_k_nearest_neighbor_model()

    trained_knn_model = train_k_nearest_neighbor_model(knn_model, kidney_x_train, kidney_y_train)

    kidney_y_predicted = predict_labels_for_test_set(trained_knn_model, kidney_x_test)

    # ===============================================

    print("Confusion matrix:")
    print(confusion_matrix(kidney_y_test, kidney_y_predicted))
    print()

    print(f"Accuracy: {accuracy_score(kidney_y_test, kidney_y_predicted)}")

    print(f"Precision: {precision_score(kidney_y_test, kidney_y_predicted)}")

    print(f"Recall: {recall_score(kidney_y_test, kidney_y_predicted)}")

    print(f"F1-score: {f1_score(kidney_y_test, kidney_y_predicted)}")


main()

# ==============================================================
# Questions: