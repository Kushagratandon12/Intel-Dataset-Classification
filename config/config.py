import pathlib
import os

# data
SYSTEM_PATH = os.getcwd()
DATA_URL = 'kaggle datasets download -d puneet6060/intel-image-classification'
DATA_NAME = 'intel_images'
DATA_PATH = '/media/samartht/eb7cc819-496c-4412-85c7-dbf08a6edd2a/dataset'
IMAGE_DATA = os.path.join(DATA_PATH,'intel_images')
TRAIN_DATA = os.path.join(IMAGE_DATA,'seg_train/seg_train')
TEST_DATA = os.path.join(IMAGE_DATA,'seg_test/seg_test')
PRED_DATA = os.path.join(IMAGE_DATA,'seg_pred/seg_pred')
LOG_PATH = os.path.join(SYSTEM_PATH,'Logs')
CLASS_NAMES  = list(['forest' ,'sea' ,'buildings' ,'glacier' ,'street', 'mountain'])


# Image Parameters
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Model Hyperparameters 
MODEL_VERSION = 1
LOSS = 'categorical_crossentropy'
BASE_MODEL='InceptionV3'
METRICS = 'accuracy'
EPOCHS = 50
INIT_LR = 1e-3
BATCH_SIZE = 128
MODEL_NAME = BASE_MODEL + '_model_weights.h5'
MODEL_PATH = os.path.join(SYSTEM_PATH,os.path.join('model',str(MODEL_VERSION)))

# Server Details
Model = 'intel_model'
HOST ='localhost'
RESTAPI_PORT = 8501