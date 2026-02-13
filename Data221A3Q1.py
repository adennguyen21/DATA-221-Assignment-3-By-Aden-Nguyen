# Question 1 - Statistical measures based on ViolentCrimesPerPop.
# ===============================================================================

import pandas as pd
import numpy as np


def convert_csv_as_dataframe(crime_file):
    # Converts the crime csv as a dataframe.
    crime_dataframe = pd.read_csv(crime_file)

    return crime_dataframe


def convert_column_as_list(dataframe, column_name):
    column_list = list(dataframe[column_name])
    print(sorted(column_list))
    return column_list

def calculate_mean(column_list):
    column_mean = np.mean(column_list)

    return column_mean


def calculate_median(column_list):
    column_median = np.median(column_list)

    return column_median


def calculate_standard_deviation(column_list):
    column_standard_deviation = np.std(column_list)

    return column_standard_deviation


def calculate_minimum_value(column_list):
    column_minimum = np.min(column_list)

    return column_minimum

def calculate_maximum_value(column_list):
    column_maximum = np.max(column_list)

    return column_maximum

# ======================================================================

def main():
    crime_csv_file = "crime.csv"

    crime_dataframe = convert_csv_as_dataframe(crime_csv_file)

    violent_crime_per_pop_list = convert_column_as_list(crime_dataframe, "ViolentCrimesPerPop")

    print(f"Mean: {calculate_mean(violent_crime_per_pop_list)}")

    print(f"Median: {calculate_median(violent_crime_per_pop_list)}")

    print(f"Standard deviation: {calculate_standard_deviation(violent_crime_per_pop_list)}")

    print(f"Minimum value: {calculate_minimum_value(violent_crime_per_pop_list)}")

    print(f"Maximum value: {calculate_maximum_value(violent_crime_per_pop_list)}")

main()

# =============================================
# Questions:
