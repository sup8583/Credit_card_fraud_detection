import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            return pred
                                    
        
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 distance_from_last_transaction:float,
                 ratio_to_median_purchase_price:float,
                 repeat_retailer:float,
                 used_chip:float,
                 used_pin_number:float,
                 online_order:float):
                 
        
        self.distance_from_last_transaction=distance_from_last_transaction
        self.ratio_to_median_purchase_price=ratio_to_median_purchase_price
        self.repeat_retailer=repeat_retailer
        self.used_chip=used_chip
        self.used_pin_number=used_pin_number
        self.online_order=online_order
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'distance_from_last_transaction':[self.distance_from_last_transaction],
                'ratio_to_median_purchase_price':[self.distance_from_last_transaction],
                'repeat_retailer':[self.repeat_retailer],
                'used_chip':[self.used_chip],
                'used_pin_number':[self.used_pin_number],
                'online_order':[self.online_order]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)