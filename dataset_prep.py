"""
Module to store the functions that will support an implementation of
the k-nearest neighbors model for predictions from a vector dataset
specifically by holding convenience functions for dataset loading,
cleanup, normalization, and standardization

functions will be built on the use of numpy arrays for convenience
"""

import csv
from collections import Counter
from operator import itemgetter

import numpy as np


def raw_dataset_csv_to_nested_list(filename):
    """
    Simple loading function using `csv.reader` to quickly import csv data

    Args:
        filename (str): string representing the file path from which the data
            will be loaded

    Returns:
        list of lists representing the data in the csv file
    """
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)
        return dataset


def cast_numbers_to_float(dataset):
    """
    Supply nested list, cast numbers to float

    semi-egregious use of `try` / `except`
    """
    new_dataset = []
    for row in dataset:
        new_row = []
        for item in row:
            try:
                new_row.append(float(item))
            except ValueError:
                new_row.append(item)
        new_dataset.append(new_row)
    return new_dataset


def inspect_types_in_dataset(dataset, column_index):
    """
    iterate over items in a selected column count types
    """
    column = [type(row[column_index]) for row in dataset]
    return Counter(column)


def pivot_categorical_feature_columns(dataset,
                                      column_index,
                                      header=True):
    """
    iterate over items in a selected column of categorical variables
    return a dataset where the categorical variables are columns (0, 1)

    using a convention where the order of the categorical variable labels
    will be in descending order of frequencies
    """
    # using the variable s as the beginning of the slice on the dataset
    s = 0
    if header:
        s = 1
    category_counts = Counter([row[column_index] for row in dataset[s:]])
    category_list = [(key, val) for key, val in category_counts.iteritems()]
    category_list.sort(key=itemgetter(1), reverse=True)
    num_categories = len(category_list)
    pivoted_data = [[row[0] for row in category_list]]
    identity_matrix = np.eye(num_categories)
    category_dict = {
        pivoted_data[0][i]: identity_matrix[i] for i in range(num_categories)
    }
    for row in dataset[s:]:
        vector = list(category_dict[row[column_index]])
        pivoted_data.append(vector)
    return pivoted_data


def norm_dataset(dataset, v_len):
    """normalize features in dataset

    Args:
        dataset (numpy.array): numpy array containing the dataset to be
            normalized
        v_len (int): the length of the feature vector each row; assumes that
            the features are ordered first in the rows and that the labels or
            true values are in the later values in the rows
    """
    n_range = dataset[:, :v_len].max(axis=0) - dataset[:, :v_len].min(axis=0)
    features_less_min = dataset[:, :v_len] - dataset[:, :v_len].min(axis=0)
    normed_features = features_less_min / n_range
    return np.concatenate((normed_features, dataset[:, v_len:]), axis=1)


def standardize_dataset(dataset, v_len):
    """standardize features in dataset

    Args:
        dataset (numpy.array): numpy array containing the dataset to be
            normalized
        v_len (int): the length of the feature vector each row; assumes that
            the features are ordered first in the rows and that the labels or
            true values are in the later values in the rows
    """
    features_less_mean = dataset[:, :v_len] - dataset[:, :v_len].mean(axis=0)
    standard_deviations = dataset[:, :v_len].std(axis=0)
    standardized_features = features_less_mean / standard_deviations
    return np.concatenate((standardized_features, dataset[:, v_len:]), axis=1)


def split_dataset(dataset, split_count, split_array=None):
    """split a dataset into two groups

    Args:
        dataset (numpy.array): numpy array containing the dataset to be split
        split_count (int): the number of rows from the dataset in one group
        array (numpy.array): an array of ones and zeros to be used to split
            the dataset; a convenience for testing

    Returns:
        dataset1, dataset2: two numpy.array objects containing each of the
            split datasets
    """
    dataset_length = dataset.shape[0]
    if split_count > dataset_length:
        raise ValueError("split_count value can't be more row number")
    try:
        split_array.any()
    except AttributeError:
        split_array = np.concatenate(
            np.ones(split_count),
            np.zeros(dataset_length - split_count),
            0
        )
        np.random.shuffle(split_array)
    split_indices = np.where(split_array)
    dataset1 = dataset[split_indices]
    other_indices = np.where(split_array == 0)
    dataset2 = dataset[other_indices]
    return dataset1, dataset2
