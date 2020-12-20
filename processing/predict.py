import time
import os
import glob
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2,InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from datetime import datetime
from config import config
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class Inference(BaseEstimator,TransformerMixin):
	def __init__(self,model_path,prediction_data,log_path='data',n_layers=150):
		self.model_path = model_path
		self.test_path = prediction_data
		self.n_layers = n_layers
		self.data_folder = log_path
		self.img_data = []
		self.predictions_data = []
		
		self.checkpoint_path = os.path.join(self.model_path,config.BASE_MODEL+"_training/cp.ckpt")

		self.transform()
		self.fit()
	
	def define_model(self):
		if config.BASE_MODEL =='ResNet50V2':
			# Pre-trained model with MobileNetV2
			base_model = ResNet50V2(input_shape=config.IMG_SHAPE,include_top=False,weights='imagenet')
			print("Number of layers in the base model: ", len(base_model.layers))
			head_model = base_model
			for layers in base_model.layers[:self.n_layers]:
			    layers.trainable=False
			head_model = head_model.output
			head_model = tf.keras.layers.GlobalMaxPooling2D()(head_model)
			head_model = tf.keras.layers.Flatten(name="Flatten")(head_model)
			head_model = tf.keras.layers.Dense(1024,activation='relu')(head_model)
			head_model = tf.keras.layers.Dropout(0.2)(head_model)
			prediction_layer = tf.keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax')(head_model)
			model = tf.keras.Model(inputs=base_model.input,outputs=prediction_layer)
		
		if config.BASE_MODEL =='InceptionV3':
			base_model = InceptionV3(input_shape=config.IMG_SHAPE,include_top=False,weights='imagenet')
			print("Number of layers in the base model: ", len(base_model.layers))
			head_model = base_model
			for layers in base_model.layers[:self.n_layers]:
			    layers.trainable=False
			head_model = head_model.output
			head_model = tf.keras.layers.GlobalMaxPooling2D()(head_model)
			head_model = tf.keras.layers.Flatten(name="Flatten")(head_model)
			head_model = tf.keras.layers.Dense(1024,activation='relu')(head_model)
			head_model = tf.keras.layers.Dropout(0.5)(head_model)
			prediction_layer = tf.keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax')(head_model)
			model = tf.keras.Model(inputs=base_model.input,outputs=prediction_layer)
		return model
	
	def transform(self):
		print("Reading test data for prediction .. \n")
		self.pred_data = glob.glob(self.test_path+'/*.jpg')
		for filename in self.pred_data:
			img = image.load_img(filename,target_size=(config.IMG_SHAPE))
			img = image.img_to_array(img)
			img = img/255.0
			self.img_data.append(img)
		self.img_data = np.asarray(self.img_data)
		print("Prediction data generated {} images".format(self.img_data.shape[0]))

	def fit(self):
		print("Reading Model checkpoints from",self.checkpoint_path)
		model = self.define_model()
		model.load_weights(self.checkpoint_path)
		model.save(os.path.join(config.MODEL_PATH,'saved_model/model'))
		start = time.time()
		for img in self.img_data:
			img = np.expand_dims(img,axis=0)

			predictions = np.argmax(model.predict(img))
			self.predictions_data.append(config.CLASS_NAMES[predictions])
		end = time.time()
		print("\nPredictions took {:.2f}s for {} images".format(end-start,len(self.pred_data)))
		print("Average time for predictions {:.2f}seconds/image".format((end-start)/len(self.pred_data)))
		print("saving model at {}".format(config.MODEL_PATH))
		return None
