# Question 3 - Creating training data and testing data.
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split


def convert_csv_as_dataframe(kidney_file):
    # Converts the kidney csv as a dataframe.
    kidney_dataframe = pd.read_csv(kidney_file)

    return kidney_dataframe


def create_matrix_x(kidney_dataframe):
    kidney_matrix_x = kidney_dataframe.drop(columns=["classification"])

    return kidney_matrix_x


def create_vector_y(kidney_dataframe):
    kidney_vector_y = kidney_dataframe["classification"]

    return kidney_vector_y


def split_dataset(matrix_x, vector_y):
    kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test = train_test_split(matrix_x, vector_y, test_size = 0.30, random_state = 42)

    return kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test

# ========================================================

def main():
    kidney_disease_csv_file = "Kidney_disease.csv"

    kidney_disease_dataframe = convert_csv_as_dataframe(kidney_disease_csv_file)

    kidney_matrix_x = create_matrix_x(kidney_disease_dataframe)

    kidney_vector_y = create_vector_y(kidney_disease_dataframe)

    split_dataset(kidney_matrix_x, kidney_vector_y)

main()
