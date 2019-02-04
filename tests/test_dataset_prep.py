"""
module to contain tests for code in dataset_prep module

to contain a test class per class or function in the dataset_prep module
"""

import unittest

import numpy as np

from dataset_prep import raw_dataset_csv_to_nested_list
from dataset_prep import cast_numbers_to_float
from dataset_prep import inspect_types_in_dataset
from dataset_prep import pivot_categorical_feature_columns
from dataset_prep import norm_dataset

FILENAME_FIXTURE1 = 'data/iris_data.csv'
FILENAME_FIXTURE2 = 'data/usedcars.csv'
NESTED_LIST = [
    ['string', 'mike', '2.2'],
    ['3.141', 'foo', '40'],
    ['bar', '7.6', 'buzz']
]


class TestRawDatasetCsvToNestedList(unittest.TestCase):
    """unit tests for the raw_dataset_csv_to_nested_list function
    """
    def test_load_fixture_file_match_first_two_rows(self):
        """Load the fixture data file `iris_data.csv` match first two rows
        """
        expected = [
            ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
            ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa']
        ]
        dataset = raw_dataset_csv_to_nested_list(FILENAME_FIXTURE1)
        actual = dataset[:2]
        self.assertEqual(expected, actual)


class TestCastNumbersToFloat(unittest.TestCase):
    """unit tests for the cast_numbers_to_float function
    """
    def test_mix_of_number_and_non_number_strings_in_nested_list(self):
        """TODO: improve docstrings
        """
        expected = [
            ['string', 'mike', 2.2],
            [3.141, 'foo', 40.0],
            ['bar', 7.6, 'buzz']
        ]
        actual = cast_numbers_to_float(NESTED_LIST)
        self.assertEqual(expected, actual)


class TestInspectTypesInDataset(unittest.TestCase):
    """unit tests for inspect_types_in_dataset function
    """
    def test_all_strings(self):
        """TODO: improve docstrings
        """
        expected = {str: 3}
        actual = inspect_types_in_dataset(NESTED_LIST, 0)
        self.assertEqual(expected, actual)

    def test_mix_of_strings_and_floats(self):
        """TODO: improve docstrings
        """
        expected = {float: 2, str: 1}
        mixed_input = cast_numbers_to_float(NESTED_LIST)
        actual = inspect_types_in_dataset(mixed_input, 2)
        self.assertEqual(expected, actual)


class TestPivotCategoricalFeatureColumens(unittest.TestCase):
    """unit tests for pivot_categorical_feature_columns function
    """
    def test_from_used_car_csv_data_models(self):
        """Using the 'model' categorical feature data from 'usedcar.csv' file
        to test expected behavior of the function
        """
        expected = [
            ['SE', 'SES', 'SEL'],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        dataset = raw_dataset_csv_to_nested_list(FILENAME_FIXTURE2)
        actual = pivot_categorical_feature_columns(dataset[44:53],
                                                   1,
                                                   header=False)
        self.assertEqual(expected, actual)


class Test(unittest.TestCase):
    """unit tests for norm_dataset function
    """
    def test_expected_output_with_simple_array(self):
        """Simple array of values to be normalized returns expected output
        """
        input_dataset = np.array([
            [4.0, 8.0, 10.0, 97.0],
            [2.0, 4.0, 0.0, 100.0],
            [4.0, 6.0, 3.3, 90.0],
            [4.0, 0.0, 2.7, 95.4]
        ])
        expected = [
            [1.0, 1.0, 1.0, 0.7],
            [0.0, 0.5, 0.0, 1.0],
            [1.0, 0.75, 0.33, 0.0],
            [1.0, 0.0, 0.27, 0.54]
        ]
        output = norm_dataset(input_dataset, 4)
        actual = []
        for row in output:
            new_row = []
            for item in row:
                new_row.append(round(item, 2))
            actual.append(new_row)
        test_bool = expected == actual
        self.assertTrue(test_bool)
