from django.test import TestCase

from apps.ml.Linear_Regression.linearRegression import LinReg

from apps.ml.registry import MLRegistry

import inspect


class MLTests(TestCase):
    def test_redlin_algorithm(self):
        input_data = {'fixed acidity': 7.4,
                      'volatile acidity': 0.7,
                      'citric acid': 0.0,
                      'chlorides': 0.076,
                      'free sulfur dioxide': 11.0,
                      'total sulfur dioxide': 34.0,
                      'density': 0.9978,
                      'pH': 3.51,
                      'sulphates': 0.56,
                      'alcohol': 9.4,
                      }
        my_alg = LinReg()
        response = my_alg.compute_prediction_red(input_data)
        # print('\n\n', response['message'], '\n\n')
        self.assertEqual('OK', response['status'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = LinReg()
        algorithm_name = "Linear Regression"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "karthik"
        algorithm_description = "Linear Regression with simple pre- and post-processing"
        algorithm_code = inspect.getsource(LinReg)

        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)

        self.assertEqual(len(registry.endpoints), 1)
