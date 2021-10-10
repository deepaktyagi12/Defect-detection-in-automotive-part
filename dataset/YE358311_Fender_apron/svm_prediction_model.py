from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
from numpy import argmax
import pickle
from sklearn.calibration import CalibratedClassifierCV 
from sklearn import svm
import joblib

def svm_model(features, labels, data, class1,svm_out_file_name):
    
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.4, random_state=5)
    model = svm.LinearSVC(max_iter=5000)
    
    mrf=CalibratedClassifierCV(model)
    mrf.fit(X_train, y_train)
    
    filename = 'results/'+svm_out_file_name
    f=open(filename+".csv", 'w', newline='')
    
    modeil_file='data/'+svm_out_file_name
    with open(modeil_file+'.pkl','wb') as fm:
        pickle.dump(mrf,fm)
    sav_filename=modeil_file+'.joblib'
    joblib.dump(mrf, sav_filename, compress=3)
        
    print("SVM run successfully")
    yp_pred = mrf.predict_proba(X_test)
    y_pred = mrf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test,y_pred)
    print('Acuracy: %f' %accuracy)
    matrix=metrics.confusion_matrix(y_test,y_pred)
    print('Confusion matrix:', matrix)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, yp_pred[:,1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ix = argmax(tpr-fpr)
    best_thresh = thresholds[ix]
   
    y_final=lambda x: 1 if x>best_thresh else 0
    y_4=lambda x: 1 if x>0.4 else 0
    y_3=lambda x: 1 if x>0.3 else 0
    y_25=lambda x: 1 if x>0.25 else 0
    y_pred25=[]
    y_pred3=[]
    y_pred4=[]
    y_best=[]
    for i in range(len(y_pred)): 
        y_pred25.append(y_25(yp_pred[i,1]))
        y_pred3.append(y_3(yp_pred[i,1]))
        y_pred4.append(y_4(yp_pred[i,1]))
        y_best.append(y_final(yp_pred[i,1]))

    accuracy4=metrics.accuracy_score(y_test,y_pred4)
    print('Acuracy 0.40: %f' %accuracy4)
    matrix4=metrics.confusion_matrix(y_test,y_pred4)
    print('Confusion matrix 0.40:', matrix4)
    accuracy3=metrics.accuracy_score(y_test,y_pred3)
    print('Acuracy 0.3: %f' %accuracy3)
    matrix3=metrics.confusion_matrix(y_test,y_pred3)
    print('Confusion matrix 0.3:', matrix3)
    accuracy25=metrics.accuracy_score(y_test,y_pred25)
    print('Acuracy 0.25: %f' %accuracy25)
    matrix25=metrics.confusion_matrix(y_test,y_pred25)
    print('Confusion matrix 0.25:', matrix25)

    accuracybest=metrics.accuracy_score(y_test,y_best)
    print("AUC",roc_auc)
    print('Best Threshold=%f' % (best_thresh))
    print('Acuracy best: %f' %accuracybest)
    matrixbest=metrics.confusion_matrix(y_test,y_best)
    print('Confusion matrix best:', matrixbest)

    writer = csv.writer(f,delimiter=',')

    row=['Threshold','Accuracy', 'True positive', 'False  Positive', 'False  Negative', 'True Negative']
    writer.writerow(row)
    row1=['Roc', roc_auc,'thresh',best_thresh]
    writer.writerow(row1)
    row2=['best_thresh', accuracybest,matrixbest[0,0],matrixbest[0,1],matrixbest[1,0],matrixbest[1,1]]
    writer.writerow(row2)
    row3=['thresh_0.5', accuracy,matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1]]
    writer.writerow(row3)
    row4=['thresh_0.4', accuracy4,matrix4[0,0],matrix4[0,1],matrix4[1,0],matrix4[1,1]]
    writer.writerow(row4)
    row5=['thresh_0.3', accuracy3,matrix3[0,0],matrix3[0,1],matrix3[1,0],matrix3[1,1]]
    writer.writerow(row5)
    row6=['thresh_0.25', accuracy25,matrix25[0,0],matrix25[0,1],matrix25[1,0],matrix25[1,1]]
    writer.writerow(row6)
    f.close()   