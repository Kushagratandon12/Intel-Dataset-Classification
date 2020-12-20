from sklearn.pipeline import Pipeline
from processing import load_data as ld
from processing import traindata as td
from processing import predict
from config import config
import os
import logging

pipe = Pipeline(
            [
                ('CreateDataset', ld.CreateDataset(config.TRAIN_DATA,config.TEST_DATA,log_path=config.LOG_PATH)
                    ),
                #('TrainDataset',td.TrainDataset(config.TRAIN_DATA,config.TEST_DATA,log_path=config.LOG_PATH)
                #   ),
                ('Inference',predict.Inference(config.MODEL_PATH,config.PRED_DATA,log_path=config.LOG_PATH))

            ]
        )

