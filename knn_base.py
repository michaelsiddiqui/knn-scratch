"""
Module to store the functions that will make up a 'from scratch'
implementation of the k-nearest neighbors model for predictions
from a vector dataset
"""

from collections import Counter
from math import pow
from math import sqrt
from math import fsum
from operator import itemgetter


def euclidean_distance(vector1, vector2, length):
    """
    Compute distance between two vector of arbitrary length
    containing an arbitrary number of numerical values

    Args:
        vector1 (list): first vector
        vector2 (list): second vector
        length (int): length of vectors

    Returns:
        float representing the euclidean distance between the two vectors
    """
    distance = fsum([
        pow((vector1[i] - vector2[i]), 2) for i in range(length)
    ])
    return sqrt(distance)


def find_k_neighbors(query,
                     dataset,
                     k,
                     vector_length,
                     return_distances=False):
    """
    Return k vectors from dataset nearest to the query vector

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

    Returns:
        list of k vectors found to be nearest neighbors
    """
    distances = []
    for row in dataset:
        eu_dist = euclidean_distance(query, row, vector_length)
        distances.append((row, eu_dist))
    distances.sort(key=itemgetter(1))
    neighbors = [row[0] for row in distances[:k]]
    if return_distances:
        return neighbors, [row[1] for row in distances[:k]]
    else:
        return neighbors


def calc_category_frequency(dataset, vector_length, category_index=-1):
    """
    Return frequencies of the category variables in dataset

    Args:
        dataset (list): collection of vectors from which neighbors will
            be found
        vector_length (int): equals the length of the vector to be used in
            distance measurement; assumes numbers are stored in the first
            `vector_length` units of the vector, predicted or descriptive
            content are assumed to be in the remaining portion of the vector
            if applicable
        category_index (int): optional argument to select a specific element
            by index as the predicted category in the dataset rows;
            default is index in position one beyond length of query vector

    Returns:
        Counter object with frequencies of the category values in dataset
    """
    if category_index < 0:
        category_index = vector_length
    frequencies = Counter(
        [row[category_index] for row in dataset]
    )
    return frequencies


def predict_category_from_knn(query,
                              dataset,
                              k,
                              vector_length,
                              category_index=-1):
    """
    Return a category for query vector by knn algorithm applied to dataset

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
            by index as the predicted category in the dataset rows;
            default is index in position one beyond length of query vector

    Returns:
        nested list with knn "votes" by category sorted by descending votes
    """
    if category_index < 0:
        category_index = vector_length
    neighbors = find_k_neighbors(query, dataset, k, vector_length)
    votes = calc_category_frequency(neighbors,
                                    vector_length,
                                    category_index=category_index)
    predictions = [[key, value] for key, value in votes.iteritems()]
    predictions.sort(key=itemgetter(1))
    return predictions


def predict_knn_classes_from_set(test_set,
                                 training_set,
                                 k,
                                 vector_length,
                                 category_index=-1):
    """loop over the test set and generate predictions for each feature row

    Args:
        test_set (list): collection of vectors for which predictions are made
        dataset (list): collection of vectors from which neighbors will
            be found
        k (int): number of neighbors to be returned
        vector_length (int): equals the length of the vector to be used in
            distance measurement; assumes numbers are stored in the first
            `vector_length` units of the vector, predicted or descriptive
            content are assumed to be in the remaining portion of the vector
            if applicable
        category_index (int): optional argument to select a specific element
            by index as the predicted category in the dataset rows;
            default is index in position one beyond length of query vector
    """
    if category_index < 0:
        category_index = vector_length
    test_set_predictions = []
    for row in test_set:
        predictions = predict_category_from_knn(row,
                                                training_set,
                                                k,
                                                vector_length,
                                                category_index=category_index)
        prediction = predictions[0][0]
        test_set_predictions.append(prediction)
    return test_set_predictions
