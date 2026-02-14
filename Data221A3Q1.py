# Question 1 - Statistical measures based on ViolentCrimesPerPop.
# ===============================================================================

import pandas as pd
import numpy as np


def convert_csv_as_dataframe(crime_file):
    # Converts the crime csv as a dataframe.
    crime_dataframe = pd.read_csv(crime_file)

    return crime_dataframe


def convert_column_as_list(dataframe, column_name):
    # Converts a certain column as a list of values.
    column_list = list(dataframe[column_name])

    return column_list


def calculate_mean(column_list):
    # Calculates the mean of the column that was turned into a list.
    column_mean = np.mean(column_list)

    return column_mean


def calculate_median(column_list):
    # Calculates the median of the column that was turned into a list.
    column_median = np.median(column_list)

    return column_median


def calculate_standard_deviation(column_list):
    # Calculates the standard deviation of the column that was turned into a list.
    column_standard_deviation = np.std(column_list)

    return column_standard_deviation


def find_minimum_value(column_list):
    # Finds the minimum value from the column that was turned into a list.
    column_minimum = np.min(column_list)

    return column_minimum


def find_maximum_value(column_list):
    # Finds the maximum value from the column that was turned into a list.
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

    print(f"Minimum value: {find_minimum_value(violent_crime_per_pop_list)}")

    print(f"Maximum value: {find_maximum_value(violent_crime_per_pop_list)}")

main()

# ==========================================================================
# Questions:

# 1. Compare the mean and median. Does the distribution look symmetric or skewed? Explain briefly.
#
#     The distribution looks skewed, as the mean is noticeably higher than the median. This means that there are some high
#     extreme values pulling the mean upward, causing the mean to be greater than the median.
#
#
# 2. If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why.
#
#     The mean would be more affected than the median, because the mean uses every value for its calculation, while the
#     median only depends on the middle value. This makes the median less affected by extreme values, as it doesn't use
#     extreme values in its calculation.