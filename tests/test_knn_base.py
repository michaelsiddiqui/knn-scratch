"""
module to contain tests for code in knn_base module

to contain a test class per class or function in the knn_base module
"""

import unittest

from knn_base import euclidean_distance
from knn_base import find_k_neighbors
from knn_base import calc_category_frequency

FIXTURE_DATASET1 = [
    [-1, 0, 0, 'out'],
    [3, 3, 3, 'in'],
    [0, -1, 0, 'out'],
    [5, 5, 5, 'in'],
    [0, 0, -1, 'out'],
    [-1, -1, -1, 'out'],
    [7, 7, 7, 'in']
]


class TestEuclideanDistance(unittest.TestCase):
    """TestCase class containing unit tests for the euclidean_distance function

    Three tests:
    1. supply two identical vectors, returns 0
    2. supply different single number vectors, returns expected value
    3. supply different three number vectors, returns expected value
    """
    def test_identical_vector_arguments(self):
        """Test that two identical vectors return distance of zero"""
        expected = 0
        actual = euclidean_distance([2], [2], 1)
        self.assertEqual(expected, actual)

    def test_different_single_number_vectors(self):
        """Test that the distance between a [2] and [4] is 2"""
        expected = 2
        actual = euclidean_distance([2], [4], 1)
        self.assertEqual(expected, actual)

    def test_different_three_number_vectors(self):
        """Test that the distance between longer vectors is correct"""
        expected = 6
        vector1 = [1, 7, 2]
        vector2 = [-3, 3, 0]
        actual = euclidean_distance(vector1, vector2, 3)
        self.assertEqual(expected, actual)


class TestFindKNeighbors(unittest.TestCase):
    """TestCase class containing unit tests for the find_k_neighbors function

    Two tests:
    1. supply two row dataset and query, returns correct row
    2. supply larger dataset, query with k=3, returns expected value
    """
    def test_find_expected_single_neighbor(self):
        """supply a two row dataset and query, returns correct row"""
        dataset = [[0, 0, 0, 'wrong'], [7, 7, 7, 'correct']]
        query = [8, 8, 8]
        expected = [dataset[1]]
        actual = find_k_neighbors(query, dataset, 1, 3)
        self.assertEqual(expected, actual)

    def test_find_expected_three_neighbors(self):
        """supply a larger dataset, query with k equal to three
        """
        dataset = FIXTURE_DATASET1
        query = [6, 6, 6]
        expected = [dataset[3], dataset[6], dataset[1]]
        actual = find_k_neighbors(query, dataset, 3, 3)
        self.assertEqual(expected, actual)

    def test_find_expected_two_neighbors_with_distances(self):
        """query the fixture dataset and get expected distances
        """
        dataset = FIXTURE_DATASET1
        query = [2, 9, 5]
        expected_neighbors = [[5, 5, 5, 'in'], [7, 7, 7, 'in']]
        expected_distances = [5.0, 5.744562646538029]
        actual_result = find_k_neighbors(
            query,
            dataset,
            2,
            3,
            return_distances=True
        )
        self.assertEqual(expected_neighbors, actual_result[0])
        self.assertEqual(expected_distances, actual_result[1])


class TestCalcCategoryFrequency(unittest.TestCase):
    """TestCase class containing unit tests for calc_category_frequency func
    """
    def test_find_expected_frequent_category_from_dataset1(self):
        """supply fixture dataset1 get expected category count
        """
        dataset = FIXTURE_DATASET1
        query = [6, 6, 6]
        neighbors = find_k_neighbors(query, dataset, 3, 3)
        actual = calc_category_frequency(neighbors, 3)
        expected = {'in': 3}
        self.assertEqual(actual, expected)
