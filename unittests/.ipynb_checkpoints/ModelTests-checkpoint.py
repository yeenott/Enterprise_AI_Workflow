


import unittest

## import model specific functions and variables
from model import *

import warnings
with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)

class ModelTest(unittest.TestCase):

    def test_01_train(self):
  
        ## train the model
        model_train(os.path.join("data","cs-train"),test=True)
        saved_model = os.path.join("models","sl-united_kingdom-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
  
        ## train the model
        all_data, all_models = model_load()
        model = all_models['united_kingdom']
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
   
        ## ensure that a list can be passed        
        result = model_predict('united_kingdom','2018','01','01',test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred >= 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
    
