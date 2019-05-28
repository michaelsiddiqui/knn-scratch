"""
module to contain tests for code in knn_regression module

to contain a test class per class or function in the knn_regression module
"""

import unittest

from knn_regression import predict_from_knn_average
from knn_regression import predict_knn_regression_for_set
from knn_regression import linear_distance_weights
from knn_regression import gaussian
from knn_regression import gaussian_distance_weights
from knn_regression import predict_from_knn_linear_distance_weighted
from knn_regression import predict_from_knn_gaussian_distance_weighted

FIXTURE_DATASET1 = [
    [-1, 0, 0, 'out', 1.0],
    [3, 3, 3, 'in', 2.0],
    [0, -1, 0, 'out', 1.0],
    [5, 5, 5, 'in', 3.0],
    [0, 0, -1, 'out', 1.0],
    [-1, -1, -1, 'out', 1.0],
    [7, 7, 7, 'in', 4.0]
]


class TestPredictFromKnnAverage(unittest.TestCase):
    """TestCase class containing unit tests for predict_from_knn_average func
    """
    def test_simple_dataset_result(self):
        """supply fixture dataset1 get expected regression output
        """
        dataset = FIXTURE_DATASET1
        query = [6, 6, 6]
        actual = predict_from_knn_average(query,
                                          dataset,
                                          3,
                                          3,
                                          category_index=4)
        expected = 3.0
        self.assertEqual(actual, expected)

    def test_exact_match_with_1NN(self):
        """if I give an identical vector as one in the set and choose k=1
        I should get the exact value as that identical row
        """
        dataset = FIXTURE_DATASET1
        query = [5, 5, 5]
        actual = predict_from_knn_average(query,
                                          dataset,
                                          1,
                                          3,
                                          category_index=4)
        expected = 3.0
        self.assertEqual(actual, expected)


class Test(unittest.TestCase):
    """TestCase class containing tests for predict_knn_regression_for_set
    """
    def test_simple_query_set_and_model_dataset_result(self):
        """query a few rows against fixture dataset1 get expected precitions
        """
        dataset = FIXTURE_DATASET1
        query_set = [[6, 6, 6], [4, 4, 4], [-1, -1, -1]]
        expected = [3.5, 2.5, 1.0]
        actual = predict_knn_regression_for_set(
            query_set,
            dataset,
            2,
            3,
            category_index=4
        )
        self.assertEqual(actual, expected)


class TestLinearDistanceWeights(unittest.TestCase):
    """TestCase class containing tests for linear_distance_weights
    """
    def test_expected_output_for_two_distances(self):
        """I expect the weighting to be proportional to relative distances
        and normalized to a total value of one
        """
        # I expect that first weight to be four times the second weight
        distances = [1, 4]
        expected = [0.8, 0.2]
        actual = linear_distance_weights(distances)
        self.assertEqual(actual, expected)

    def test_expected_output_for_three_distances(self):
        distances = [0, 2, 3]
        expected = [0.5, 0.3, 0.2]
        actual = linear_distance_weights(distances)
        self.assertEqual(actual, expected)

    def test_all_same_distances_returns_simple_average_weights(self):
        """If all the distances are the same, the weightings should be equal
        """
        distances = [1, 1, 1, 1]
        expected = [0.25, 0.25, 0.25, 0.25]
        actual = linear_distance_weights(distances)
        self.assertEqual(actual, expected)


class TestGaussian(unittest.TestCase):
    """TestCase class containing tests for gaussian
    """
    def test_expected_output_for_zero(self):
        """gaussian pdf weight for value=0
        """
        expected = 0.3989422804014327
        actual = gaussian(0)
        self.assertEqual(actual, expected)

    def test_expected_output_for_one(self):
        """gaussian pdf weight for value=1
        """
        expected = 0.24197072451914337
        actual = gaussian(1)
        self.assertEqual(actual, expected)

    def test_expected_output_with_complex_args(self):
        """gaussian pdf weight for (7, 5, 5)
        """
        expected = 0.073654028060664664
        actual = gaussian(7, 5, 5)
        self.assertEqual(actual, expected)

    def test_expected_output_with_complex_args_including_float(self):
        """gaussian pdf weight for (98.0, 100, 12.0)
        """
        expected = 0.032786643008494994
        actual = gaussian(98.0, 100, 12.0)
        self.assertEqual(actual, expected)


class TestGaussianDistanceWeights(unittest.TestCase):
    """TestCase class containing tests for gaussian_distance_weights
    """
    def test_all_same_distances_returns_simple_average_weights(self):
        """If all the distances are the same, the weightings should be equal
        """
        distances = [1, 1, 1, 1]
        expected = [0.25, 0.25, 0.25, 0.25]
        actual = gaussian_distance_weights(distances)
        self.assertEqual(actual, expected)

    def test_expected_output_for_two_distances(self):
        distances = [0, 1]
        expected = [0.6224593312018546, 0.37754066879814546]
        actual = gaussian_distance_weights(distances)
        self.assertEqual(actual, expected)


class TestPredictFromKnnLinearDistanceWeighted(unittest.TestCase):
    """tests for predict_from_knn_linear_distance_weighted function
    """
    def test_exact_match_with_1NN(self):
        """if I give an identical vector as one in the set and choose k=1
        I should get the exact value as that identical row
        """
        dataset = FIXTURE_DATASET1
        query = [5, 5, 5]
        actual = predict_from_knn_linear_distance_weighted(query,
                                                           dataset,
                                                           1,
                                                           3,
                                                           category_index=4)
        expected = 3.0
        self.assertEqual(actual, expected)

    def test_simple_example_with_two_neighbors(self):
        """use single item vectors to make arithmetic simple
        """
        dataset = [
            [1, 5],  # distance = 1, weight = 0.8, contribution = 4.0
            [6, 10],  # distance = 4, weight = 0.2, contribution = 2.0
            [-10, 700]
        ]
        query = [2]
        actual = predict_from_knn_linear_distance_weighted(query,
                                                           dataset,
                                                           2,
                                                           1
                                                           )
        expected = 6.0
        self.assertEqual(actual, expected)

    def test_example_with_three_neighbors(self):
        """use single item vectors to make arithmetic simple
        """
        dataset = [
            [90, 40, 800],
            [6, 2, 2],  # distance = 2, weight = 0.3, contribution = 0.6
            [6, 0, 6],  # distance = 0, weight = 0.5, contribution = 3.0
            [-10, -60, 700],
            [3, 0, 2]  # distance = 3, weight = 0.2, contribution = 0.4
        ]
        query = [6, 0]
        actual = predict_from_knn_linear_distance_weighted(query,
                                                           dataset,
                                                           3,
                                                           2
                                                           )
        expected = 4.0
        self.assertEqual(actual, expected)


class TestPredictFromKnnGaussianDistanceWeighted(unittest.TestCase):
    """tests for predict_from_knn_gaussian_distance_weighted function
    """
    def test_exact_match_with_1NN(self):
        """if I give an identical vector as one in the set and choose k=1
        I should get the exact value as that identical row
        """
        dataset = FIXTURE_DATASET1
        query = [5, 5, 5]
        actual = predict_from_knn_gaussian_distance_weighted(query,
                                                             dataset,
                                                             1,
                                                             3,
                                                             category_index=4)
        expected = 3.0
        self.assertEqual(actual, expected)

    def test_simple_two_neighbor_case(self):
        """distances 0 and 1, to weight by [0.6224593312018546,
        0.37754066879814546]
        """
        dataset = [
            [0, 1],
            [1, 2],
            [3, 7]
        ]
        query = [0]
        expected = 0.6224593312018546 + 2*0.37754066879814546
        actual = predict_from_knn_gaussian_distance_weighted(query,
                                                             dataset,
                                                             2,
                                                             1
                                                             )
        self.assertEqual(actual, expected)
