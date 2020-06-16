import cv2
import numpy as np
from keras.models import load_model
from sklearn import metrics
from performance_matrics  import compute_stats, compute_stats,make_classes,showimage_label

def predict_images_labels(x_test,y_test):
    #### Predict the automative spare parts images weather spare part is "Healthy" or "Defective using keras model"
    model = load_model('Trained_model/MobileNet_model_keras.h5')
    y_pred =model.predict(x_test)
    y_pred = make_classes(y_pred) 
    print("True Labels",y_test)
    print("Predicted Labels",np.array(y_pred))
    compute_stats(y_test, y_pred)
    showimage_label(x_test,y_test, y_pred)

