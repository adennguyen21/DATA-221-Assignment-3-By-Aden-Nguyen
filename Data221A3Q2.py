# Question 2 - Graphing ViolentCrimesPerPop.
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

def convert_csv_as_dataframe(crime_file):
    # Converts the crime csv as a dataframe.
    crime_dataframe = pd.read_csv(crime_file)

    return crime_dataframe


def convert_column_as_list(dataframe, column_name):
    column_list = list(dataframe[column_name])

    return column_list


def create_histogram(column_list):
    plt.hist(column_list, bins=30, color="purple", edgecolor="black")
    plt.title("Distribution of Violent Crimes Per Population")
    plt.xlabel("ViolentCrimesPerPop")
    plt.ylabel("Frequency")
    plt.show()

def create_boxplot(column_list):
    plt.boxplot(column_list)
    plt.title("Box Plot of Violent Crimes Per Population")
    plt.ylabel("ViolentCrimesPerPop")
    plt.xlabel("Distribution")
    plt.show()

# ===============================================================================

def main():
    crime_csv_file = "crime.csv"

    crime_dataframe = convert_csv_as_dataframe(crime_csv_file)

    violent_crime_per_pop_list = convert_column_as_list(crime_dataframe, "ViolentCrimesPerPop")

    create_histogram(violent_crime_per_pop_list)

    create_boxplot(violent_crime_per_pop_list)

main()

#====================================================
# Questions:
