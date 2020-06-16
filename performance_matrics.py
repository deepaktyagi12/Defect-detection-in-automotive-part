import cv2
import numpy as np
from keras.models import load_model
from sklearn import metrics
def predict_images_labels(x_test,y_test):
    model = load_model('Trained_model/MobileNet_model_keras.h5')
    y_pred =model.predict(x_test)
    y_pred = make_classes(y_pred) 
    print("True Labels",y_test)
    print("Predicted Labels",np.array(y_pred))
    compute_stats(y_test, y_pred)
    showimage_label(x_test,y_test, y_pred)

def showimage_label(x_test,y_test, y_pred):
   for i in range(x_test.shape[0]):
        if y_test[i]==1:
            t_lab="Defected"
        else:
            t_lab="Healthy"
        if y_pred[i]==1:
            pre_lab="Defected"
        else:
            pre_lab="Healthy"
        image_name= "True Label:--"+t_lab+"   Predictive Label:--"+pre_lab
        print(image_name)
        cv2.namedWindow(image_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_name,720,720)
        cv2.imshow(image_name,x_test[i])
        cv2.waitKey(0)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

def compute_stats(y_test, y_pred):

    accuracy =metrics.accuracy_score(y_test, y_pred)
    print('Toatal Accuracy: %f' % accuracy)
    
    matrix = metrics.confusion_matrix(y_test, y_pred)   
    print('Confusion matrix:', matrix)

    precision = metrics.precision_score(y_test, y_pred)
    print('Precision: %f' % precision)

    recall = metrics.recall_score(y_test, y_pred)
    print('Recall: %f' % recall)

    f1 = metrics.f1_score(y_test, y_pred)
    print('F1 score: %f' % f1)

def make_classes(y_t_pred):
    y_pred=[]
    for i in y_t_pred:
        if i[0] <= 0.5:
            i[0]=1
            # i[1]=0
            y_pred.append(1)
        elif i[0] > 0.5:
            i[1]=1
            # i[0]=0
            y_pred.append(0)
    return y_pred
