import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model
from sklearn import metrics
from performance_matrics  import compute_stats, compute_stats,make_classes,showimage_label

def tflite_predition(x_test,y_test,trained_models,tflite_model):
    #### Predict the automative spare parts images weather spare part is "Healthy" or "Defective" using TFLite model
    
    ###****************** Upload the TFLite model*************************************
    model_path=trained_models+"/"+tflite_model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    input_data = np.array(x_test, dtype=np.float32)
    print(input_data.shape)

    interpreter.resize_tensor_input(input_details[0]['index'],[len(input_data), 224, 224, 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    ###*********************run the TFLite model inference************************************* 
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred = make_classes(output_data) 
    ###******************** compute the various performance matrices ****************************
    print("True Labels",y_test)
    print("Predicted Labels",np.array(y_pred))
    compute_stats(y_test, y_pred)
    showimage_label(x_test,y_test, y_pred)
