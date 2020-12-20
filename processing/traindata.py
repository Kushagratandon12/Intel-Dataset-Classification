import time
import os
from tensorflow import __version__ 
from tensorflow.summary import create_file_writer
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2,InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from config import config
print("Tensorflow version:",__version__)
warnings.filterwarnings('ignore')

class TrainDataset(BaseEstimator,TransformerMixin):
	def __init__(self,trainpath,testpath,log_path='data',n_layers=150):
		self.train_path = trainpath
		self.test_path = testpath
		self.n_layers = n_layers
		self.data_folder = log_path
		self.callbacks_list = []
		# Define the basic TensorBoard callback.
		logdir = os.path.join(config.LOG_PATH,"logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
		
		file_writer_cm = create_file_writer(logdir + '/cm')
		tensorboard_callback = TensorBoard(log_dir=logdir)		

		checkpoint_path = config.BASE_MODEL+"_training/cp.ckpt"
		cp_callback = ModelCheckpoint(filepath=os.path.join(config.MODEL_PATH,checkpoint_path),
                                      save_weights_only=True,monitor='val_loss', verbose=1,
                                      save_best_only=True)
		
		early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=1, mode='auto')
		
		self.callbacks_list = [cp_callback,tensorboard_callback]
		self.transform()
		self.fit()
	
	def define_model(self):
		if config.BASE_MODEL =='ResNet50V2':
			# Pre-trained model with MobileNetV2
			base_model = ResNet50V2(input_shape=config.IMG_SHAPE,include_top=False,weights='imagenet')
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
		# This is one of the ways to create training data batch and label batch 
		image_generator = image.ImageDataGenerator(rescale=1./255,
		                                #rotation_range=0.2,
		                                #zoom_range=[2,5],
		                                brightness_range=[2,5],
		                                width_shift_range=0.2,
		                                height_shift_range=0.2,
		                                #horizontal_flip=False,
		                                validation_split=0.25
		                                )
		self.train_data_gen = image_generator.flow_from_directory(directory=self.train_path,
		                                                batch_size=config.BATCH_SIZE,
		                                                shuffle=True,
		                                                class_mode='categorical',
		                                                target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
		                                                classes = config.CLASS_NAMES
		                                                )
		self.validation_generator = image_generator.flow_from_directory(self.train_path,
		                                                  shuffle=True,subset='validation',
		                                                  class_mode='categorical',
		                                                  classes = config.CLASS_NAMES,
		                                                  target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
		                                                 )
		self.test_generator = image_generator.flow_from_directory(self.test_path,
		                                                  shuffle=True,
		                                                  class_mode='categorical',
		                                                  target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
		                                                 )
	def fit(self):
		opt = Adam(lr=0.01)
		model =self.define_model()
		#Compilation of the model
		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		# Fitting the model 
		H = model.fit(self.train_data_gen,
              epochs=config.EPOCHS,
              callbacks=self.callbacks_list,
              validation_data = self.validation_generator,
              )


		



