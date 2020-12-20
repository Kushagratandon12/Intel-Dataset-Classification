# importing the libraries
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from config import config
import numpy as np
import json
import glob
import requests

def get_rest_url(model_name, host='localhost', port='8501', verb='predict', version=None):
    """ generate the URL path"""
    url = "http://{host}:{port}/v1/models/{model_name}/".format(host=host, port=port, model_name=model_name)
    if version:
        url += 'versions/{version}'.format(version=version)
    url += ':{verb}'.format(verb=verb)
    return url

def decode_image(img_path):
    img = image.load_img(img_path,target_size=(config.IMG_SHAPE))
    # convert the compressed string to a 3D uint8 tensor
    img = image.img_to_array(img)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = img/255.0
    # expand image dimensions and convert it into numpy 
    img = np.expand_dims(img, axis=0)
    return img

def show_image(image,label=''):
    img = plt.imread(image)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.xlabel(label)
    plt.title('Predicted label:\n{}'.format(label), fontdict={'size': 16})
    plt.show()



def sendimg_data(img_vector,url):
    # create a json string to ask query to the depoyed model
    data = json.dumps({"signature_name": "serving_default",
                   "instances": img_vector.tolist()})
    # headers for the post request
    headers = {"content-type": "application/json"}

    # make the post request -  'http://localhost:8501/v1/models/my_model/versions/1:predict'
    json_response = requests.post(url,
                              data=data,
                              headers=headers)
    # get the predictions
    predictions = json.loads(json_response.text)

    idx = np.argmax(predictions['predictions'])

    return str(config.CLASS_NAMES[idx])

        

if __name__ == '__main__':
    pred_data = glob.glob(config.PRED_DATA+'/*.jpg')
    test_image = pred_data[np.random.randint(0,len(pred_data))]
    print("Opening {}".format(test_image))
    
    model_name = config.Model
    host = config.HOST
    version = str(config.MODEL_VERSION)
    restapi_port = config.RESTAPI_PORT
    
    url = get_rest_url(model_name,host=host,port=str(restapi_port),version=version)
    img_data = decode_image(test_image)
    
    class_name = sendimg_data(img_data,url)
    #print(class_name)
    show_image(test_image,class_name)
    # server : nohup tensorflow_model_server --rest_api_port=8501 --model_name=mnist_model --model_base_path=/media/samartht/eb7cc819-496c-4412-85c7-dbf08a6edd2a/Projects/tensorflow_serving/my_model >server.log 2>&1
