"""
module to contain tests for code in knn_regression module

to contain a test class per class or function in the knn_regression module
"""

import unittest

from knn_regression import predict_from_knn_average

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
