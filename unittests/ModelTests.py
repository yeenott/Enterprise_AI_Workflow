#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
from model.model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(os.path.join("data","cs-train"),test=True)
        saved_model = os.path.join("models","sl-netherlands-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        all_data, all_models = model_load()
        model = all_models['netherlands']
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## ensure that a list can be passed        
        result = model_predict('netherlands','2018','08','01',test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred >= 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
