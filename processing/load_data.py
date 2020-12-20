import logging
import os
import sys
import glob
import numpy as np
import time 
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CreateDataset(BaseEstimator,TransformerMixin):
	def __init__(self,trainpath,testpath,log_path='data'):
		self.train_path = trainpath
		self.test_path = testpath
		self.train_data = []
		self.test_data = []
		
		self.data_folder = log_path
		self.transform()
		self.fit()
		
	def generate_imgdataframe(self,data):
		labels = []
		filenames = []
		image_shape = []
		for imgfile in data:
		    filename = os.path.basename(imgfile)
		    label = os.path.basename(os.path.split(imgfile)[0])
		    img = plt.imread(imgfile)
		    image_shape.append(img.shape)
		    labels.append(label)
		    filenames.append(filename)
		df = pd.DataFrame({'filename':filenames,'Class':labels,'shape':image_shape})
		return df


	def transform(self):
		if os.path.isdir(self.train_path):
			self.train_dataset = np.asarray(glob.glob(self.train_path+'/*/*.jpg'))
			logging.info("Training data :",self.train_dataset.shape)
			print("Training data :",self.train_dataset.shape[0])

		if os.path.isdir(self.test_path):
			self.test_dataset = np.asarray(glob.glob(self.test_path+'/*/*.jpg'))
			logging.info("Test data :",self.test_dataset.shape)
			print("Test data :",self.test_dataset.shape[0])
		return None 
		

	def fit(self):
		train_csv = os.path.join(self.data_folder,'train.csv')
		test_csv = os.path.join(self.data_folder,'test.csv')
		if not os.path.isfile(train_csv) and not os.path.isfile(test_csv):
			start = time.time()
			train_df = self.generate_imgdataframe(self.train_dataset)
			test_df = self.generate_imgdataframe(self.test_dataset)
			end = time.time()
			if not os.path.isdir(self.data_folder):
				try:
					os.makedirs(self.data_folder)
				except FileExistsError:
					sys.error("unable to create directory")		

			train_df.to_csv(train_csv)
			test_df.to_csv(test_csv)
			print('Data logs created in {:.2f}s'.format(end-start))
		else:
			print('Data logs already created - Skipping..')
		return None








if __name__ == '__main__':
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/tmp/pipeline.log',
                    filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)










