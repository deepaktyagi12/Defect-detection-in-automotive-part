from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
from numpy import argmax
import xgboost as xgb   

def xgb_model(features, labels, data, class1,xgb_out_file_name):
    
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.4, random_state=5)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    param = {'max_depth': 5, 'eta': 0.05, 'objective': 'binary:logistic'}
    # param['nthread'] = 4
    param['eval_metric'] = 'auc'
    print("XGB run successfully")
    
    num_round = 5000
    bst = xgb.train(param, dtrain, num_round, evallist)   
    
    modeil_file = 'data/'+xgb_out_file_name
    model_name=modeil_file+'.model'
    bst.save_model(model_name)
    
   
    filename = 'results/'+xgb_out_file_name
    f=open(filename+".csv", 'w', newline='')
           
    y_pred = bst.predict(dtest)
  
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ix = argmax(tpr-fpr)
    best_thresh = thresholds[ix]
    print("AUC",roc_auc)
    print('Best Threshold=%f' % (best_thresh))

    y_final=lambda x: 1 if x>best_thresh else 0
    y_5=lambda x: 1 if x>0.5 else 0
    y_4=lambda x: 1 if x>0.4 else 0
    y_3=lambda x: 1 if x>0.3 else 0
    y_25=lambda x: 1 if x>0.25 else 0
    y_pred25=[]
    y_pred3=[]
    y_pred4=[]
    y_pred5=[]
    y_best=[]
    for i in range(len(y_pred)): 
        y_pred25.append(y_25(y_pred[i]))
        y_pred3.append(y_3(y_pred[i]))
        y_pred4.append(y_4(y_pred[i]))
        y_pred5.append(y_5(y_pred[i]))
        y_best.append(y_final(y_pred[i]))

    accuracy=metrics.accuracy_score(y_test,y_pred5)
    print('Acuracy 0.50: %f' %accuracy)
    matrix=metrics.confusion_matrix(y_test,y_pred5)
    print('Confusion matrix 0.40:', matrix)
    
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
    print('Acuracy best: %f' %accuracybest)
    matrixbest=metrics.confusion_matrix(y_test,y_best)
    print('Confusion matrix best:', matrixbest)


    writer = csv.writer(f,delimiter=',')
    # for i in range(len(y_pred)):    
    #     row=[yp_pred[i,0],yp_pred[i,1],y_test.iloc[i],y_pred[i],y_pred4[i],y_pred3[i],y_pred25[i],y_best[i]]
    #     writer.writerow(row)
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