# Question 3 - Creating training data and testing data.
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split


def convert_csv_as_dataframe(kidney_file):
    # Converts the kidney csv as a dataframe.
    kidney_dataframe = pd.read_csv(kidney_file)

    return kidney_dataframe


def create_matrix_x(kidney_dataframe):
    # Creates a matrix that contains all the columns except the "classification" column.
    kidney_matrix_x = kidney_dataframe.drop(columns=["classification"])

    return kidney_matrix_x


def create_vector_y(kidney_dataframe):
    # Creates a vector based on the column "classification".
    kidney_vector_y = kidney_dataframe["classification"]

    return kidney_vector_y


def split_dataset(matrix_x, vector_y):
    # Splits the dataset into training and testing data.
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

# =============================================================
# Explanation:

'''We should not train and test a model on the same data because the model could simply
memorize the training examples, and end up giving an unrealistically high performance
score (Overfitting). Testing on the same data used for training does not measure how 
well a model handles new data, as it is basically memorizing the training data. The
purpose of the testing set is to evaluate the model's performance after it has been
trained, with new data it hasn't see before. This helps to see how well the model will
preform in real-world situations, and also prevents overfitting as we're testing it 
with completely new data.'''