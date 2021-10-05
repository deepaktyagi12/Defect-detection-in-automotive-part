import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocessing
import pickle
from Feature_relevance_model import feature_importance
from Rf_prediction_model import RF_model
from svm_prediction_model import svm_model
from xgb_prediction_model import xgb_model
from statistics import mean

def read_data():
    data = pd.read_csv(r'data/TrainingData_V1.csv')
 
    data['order_date']=pd.to_datetime(data['order_date'])
    data['delivery_date']=pd.to_datetime(data['delivery_date'])
    data['user_dob']=pd.to_datetime(data['user_dob'])
    data['user_reg_date']=pd.to_datetime(data['user_reg_date'])
    
    data["order_day"]=data["order_date"].dt.day
    data["order_month"]=data["order_date"].dt.month
    data["order_year"]=data["order_date"].dt.year
    
    data["delivery_day"]=data["delivery_date"].dt.day
    data["delivery_month"]=data["delivery_date"].dt.month
    data["delivery_year"]=data["delivery_date"].dt.year   
   
    data["user_dob_day"]=data["user_dob"].dt.day
    data["user_dob_month"]=data["user_dob"].dt.month
    data["user_dob_year"]=data["user_dob"].dt.year  

    data["user_reg_day"]=data["user_reg_date"].dt.day
    data["user_reg_month"]=data["user_reg_date"].dt.month
    data["user_reg_year"]=data["user_reg_date"].dt.year
     
    data['user_age']     =(data['order_date']-data['user_dob']).astype('timedelta64[D]')
    data['delivery_time']=(data['delivery_date']-data['order_date']).astype('timedelta64[D]')
    data['user_old']     =(data['order_date']-data['user_reg_date']).astype('timedelta64[D]')
 
    data=data.drop(['order_date','delivery_date','user_dob','user_reg_date'], axis = 1)
        
    # data = data[data['user_age'] >= 0]
    # data = data[data['delivery_time'] >= 0]
    # data = data[data['user_old'] >= 0]
   ##one hot encoder
    # data=pd.get_dummies(data, columns=['item_size', 'item_color','user_title'])
    data["item_color"] = data["item_color"].astype('category').cat.codes
    data["item_size"] =  data["item_size"].astype('category').cat.codes
    data["user_title"] = data["user_title"].astype('category').cat.codes
    data=data.fillna(0)
    # data.to_csv('data/file1.csv')
    Y_lab=data.user_decision
    data=data.drop(['user_decision'],axis=1)
    
    sz=data.shape[0]
    X=data.iloc[0:sz]   
    Y=Y_lab.iloc[0:sz]
    
    sc=StandardScaler()
    sc.fit(X)
    features=sc.fit_transform(X)
    labels=Y
    class_samples=labels.value_counts()
    class1=class_samples[0]/class_samples[1]
    np.nan_to_num(features,copy=True, nan=0.0, posinf=None, neginf=None)   
    # np.savetxt(r'data/data_encode.csv',features,delimiter=',')    
    with open(r'data/data_encode.pkl','wb') as ft:
        pickle.dump(sc,ft)

    features.shape

    print("input",features.shape,"output", len(labels), class_samples)
    return features, labels, X, class1


def main():
    features, labels, df, class1= read_data()
    
    feature_selection=True
    if feature_selection:    
        print("PM running ")
        features,selected_features=feature_importance(features, labels, df)
    else:
        data = pd.read_csv(r'data/PM_rf_importance.csv')
        for column in data.columns:
            rel_score = data[column].tolist() 
        rel_mean=mean(rel_score)
        # print(rel_mean)
        selected_features= [idx for idx, val in enumerate(rel_score) if val > 0]
        features=features[:,selected_features]
        print(selected_features,features.shape)
     
    print("xgb successfully trained")
    xgb_out_file_name='final_pm_xgb'
    xgb_model(features, labels, df, class1, xgb_out_file_name)
    
    print("SVM running ")
    svm_out_file_name='final_pm_svm'
    svm_model(features, labels, df, class1, svm_out_file_name)
      
    
    print("RF running ")
    rf_out_file_name='final_pm_rf'
    rf=RF_model(features, labels, df, class1,rf_out_file_name)
if __name__ == '__main__':

    # Pyinstaller fix
    multiprocessing.freeze_support()

    main()