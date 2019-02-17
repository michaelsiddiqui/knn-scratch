"""
Module to store the functions that will build on the knn_base module code
to enable a 'from scratch' implementation of the k-nearest neighbors model for
regression (predicting continuous variables) from a dataset
"""

from math import fsum

from knn_base import find_k_neighbors


def predict_from_knn_average(query,
                             dataset,
                             k,
                             vector_length,
                             category_index=-1
                             ):
    """find knn and then predict outcome as mean of neighbors' values

    Args:
        query (list): vector for which neighbors will be found
        dataset (list): collection of vectors from which neighbors will
            be found
        k (int): number of neighbors to be returned
        vector_length (int): equals the length of the vector to be used in
            distance measurement; assumes numbers are stored in the first
            `vector_length` units of the vector, predicted or descriptive
            content are assumed to be in the remaining portion of the vector
            if applicable
        category_index (int): optional argument to select a specific element
            by index as the to be predicted value (objective) in the dataset
            rows; default is index in position one beyond length of the
            query vector

    Returns:
        A float value for the simple average of objective value for
        the k nearest neighbors
    """
    if category_index < 0:
        category_index = vector_length
    neighbors = find_k_neighbors(query, dataset, k, vector_length)
    predicted_outcome = fsum([row[category_index] for row in neighbors]) / k
    return predicted_outcome


def predict_knn_regression_for_set(query_dataset,
                                   model_dataset,
                                   k,
                                   vector_length,
                                   category_index=-1,
                                   algorithm=predict_from_knn_average):
    """run knn regression function to predict all values in a dataset

    Args:
        query (list): vector for which neighbors will be found
        dataset (list): collection of vectors from which neighbors will
            be found
        k (int): number of neighbors to be returned
        vector_length (int): equals the length of the vector to be used in
            distance measurement; assumes numbers are stored in the first
            `vector_length` units of the vector, predicted or descriptive
            content are assumed to be in the remaining portion of the vector
            if applicable
        category_index (int): optional argument to select a specific element
            by index as the to be predicted value (objective) in the dataset
            rows; default is index in position one beyond length of the
            query vector
        algorithm (function): optional argument to select a specific function
            to implement to calculate each individual value, assumes that
            function takes as arguments the same values; defaults to
            `predict_from_knn_average` function

    Returns:
        A list of floats representing the predicted values from the algorithm
        for each of the rows of the query_dataset
    """
    if category_index < 0:
        category_index = vector_length
    all_predictions = []
    for row in query_dataset:
        prediction = algorithm(
            row,
            model_dataset,
            k,
            vector_length,
            category_index=category_index
        )
        all_predictions.append(prediction)
    return all_predictions
