import os
import random
import cv2
import numpy as np
from mobilenetv2 import train_L2
import test_Predict_keras
from test_Predict_tflite import tflite_predition
import argparse
''' This project utilize to classify the automative spare parts images whether spare part is "Healthy" or "Defected".

Main logics is in the mobilenetv2 file, where you can further customize.
'''
def load_training_data(root_dir,classes_dir):
    healthy_samples= []
    defective_samples= []
    for cls in classes_dir:
        src = root_dir + cls # Folder to copy images from
        train_FileNames = os.listdir(src)
        print(cls)
        if cls=='/YE358311_defects':
            defective_samples=[src+'/'+ name for name in train_FileNames]
            print('Images in defective set are: ', len(defective_samples))
        else:
            healthy_samples = [src+'/'+ name for name in train_FileNames]
            print('Images in healthy set are: ', len(healthy_samples))
    return  defective_samples,healthy_samples

def data_preprocessing(image_path,image_width,image_height):
    """
    Pre-process the data- alligning, and resizing the orignal images
    """
    image=cv2.imread(image_path)
    image = np.array(image, dtype=np.uint8) 
    #Rotating if image is vertical
    if image.shape[1]<image.shape[0]:
        image = np.rot90(image)
    image=cv2.resize(image, (image_width,image_height), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(image_path,image)
    return image
def training_data_preparation(defective_samples,healthy_samples,image_width,image_height):
    #Label the dataset- Healthy and Defective
    y=[]
    x=[]
    for image in defective_samples:
        x.append(data_preprocessing(image,image_width,image_height))
        y.append(1)
    print("defective_samples is ",len(x),len(defective_samples))
    for image in healthy_samples:
        x.append(data_preprocessing(image,image_width,image_height))
        y.append(0)
    print("healthy_samples is ",len(x))
    # Shuffle the data samples randomlly and the order of samples and label is remaince same 
    data = list(zip(x, y))
    random.shuffle(data)
    x_rand, y_rand = zip(*data)
    #list to array
    x_train = np.array(x_rand)
    y_train = np.array(y_rand)
    x_data_size=x_train.shape
    y_data_size=y_train.shape
    print("Shape of training dataset is",x_data_size,y_data_size)
    return x_train,y_train

def main():
    ###****** This is the main function of Defect-detection-in-automative-parts project
    ap = argparse.ArgumentParser()
    ap.add_argument("--opMode", "-mode", default='Test',
        help="opertation mode test or trained")
    ap.add_argument("--Data_Directory", "-datadir", default='dataset/YE358311_Fender_apron',
        help="Path to test data")
    ap.add_argument("--classes_dir", "-classes_dir", default=['/YE358311_defects', '/YE358311_Healthy'],
        help="Different data class")
    ap.add_argument("--classes", "-no_classes", default=2,
        help="number of classes")
    ap.add_argument("--epoch", "-no_iterations", default=5,
        help="number of epoch for learning model")
    ap.add_argument("--alpha", "-alpha", default=0.5,
        help="alpha for learning model")
    ap.add_argument("--batch", "-batchsize", default=8,
        help="number of images in one batch")
    args = vars(ap.parse_args())
    
    trained_models = 'Trained_model'
    Keras_model='MobileNet_model_keras.h5'
    Tf_model = 'model.pb'
    tflite_model='converted_model.tflite'

    image_width = 224 
    image_height = 224
    chanel=3
    input_shape = (image_width, image_height, chanel)
    data_dir =  args["Data_Directory"]
    classes_dir = args['classes_dir']
    no_of_classes= args['classes']
    epoch = args['epoch']
    alpha = args['alpha']
    batch = args['batch']
    opMode =  args["opMode"]
    ###***************Train deep learnig model****************************##
    if opMode == 'Train':
        training_data_dir=data_dir+'/train'
        train_defective_samples,train_healthy_samples=load_training_data(training_data_dir,classes_dir)
        x_train,y_train=training_data_preparation(train_defective_samples,train_healthy_samples,image_width,image_height)

        print("L2-SoftmaxLoss training...")
        train_L2(x_train, y_train, no_of_classes,input_shape, epoch, alpha,batch, True, True, trained_models, Keras_model,  Tf_model, tflite_model)
    ###***************Test the model performance on test Dataset (unseen samples)****************************##
    elif opMode == 'Test':
        test_data_dir=data_dir+'/test'
        test_defective_samples,test_healthy_samples=load_training_data(test_data_dir,classes_dir)
        x_test,y_test=training_data_preparation(test_defective_samples,test_healthy_samples,image_width,image_height)
        
        ###***************Classify images using Keras model****************************##
        # test_Predict_keras.predict_images_labels(x_test,y_test)

        ###***************Classify images using TFLite model****************************##
        tflite_predition(x_test,y_test,trained_models,tflite_model)

if __name__== "__main__":
  main()
