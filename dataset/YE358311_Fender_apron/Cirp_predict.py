import pandas as pd
import numpy as np
import multiprocessing
import pickle
from Rf_prediction_model import RF_model
import xgboost as xgb 
from Cirp_main import read_data 

def read_testdata(test_file_path):
    
    data=pd.read_excel(test_file_path)
    
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
    # print(data["user_reg_day"] , data['user_old'])
 
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
    cl=pickle.load(open(r'data/data_encode.pkl','rb'))
    features = cl.transform(data)
    np.nan_to_num(features,copy=True, nan=0.0, posinf=None, neginf=None) 

    print(features.shape)
    
    return features, data['order_item_id']
def svc_predict(svc2,X_test,order_id):
###save results
    pred_svcp=svc2.predict(X_test)
    result=pd.DataFrame()
    result["order_item_id"]=order_id
    result["user_decision"]=pred_svcp
    file_name = 'results/svm_predict_results.csv'
    result.to_csv(file_name,index=False)
    
def xgb_predict(xgb_model,X_test,order_id):
###save results
    dtest = xgb.DMatrix(X_test)
    ypred = xgb_model.predict(dtest)
    print(len(ypred))
    y_5=lambda x: 1 if x>0.5 else 0
    y_pred=[]
    for i in range(len(ypred)): 
        y_pred.append(y_5(ypred[i]))
    
    result=pd.DataFrame()
    result["order_item_id"]=order_id
    result["user_decision"]=y_pred
    file_name = 'results/xgb_predict_results.csv'
    result.to_csv(file_name,index=False)  
def rf_predict(pm_rf,X_test,order_id):
###save results
    pred_rf=pm_rf.predict(X_test)
    result=pd.DataFrame()
    result["order_item_id"]=order_id
    result["user_decision"]=pred_rf
    file_name = 'results/Rf_predict_results.csv'
    result.to_csv(file_name,index=False)
    
def main():
    features,order_id=read_testdata('data/TestingData_For_Candidate.xlsx')
    data = pd.read_csv(r'data/PM_rf_importance_v_01.csv')
    for column in data.columns:
        rel_score = data[column].tolist() 
    # rel_mean=mean(rel_score)
    # print(rel_mean)
    selected_features= [idx for idx, val in enumerate(rel_score) if val > -0.00001]
    features=features[:,selected_features]
    print(selected_features,features.shape)
     
    print("SVM prediction model is running ")
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model('data/final_pm_xgb.model')  # load data
    xgb_predict(bst,features,order_id)
   
    print("SVM prediction model is running ")
    svc2=pickle.load(open(r'data/final_pm_svm.pkl','rb'))
    svc_predict(svc2, features,order_id)
    
    print("RF  prediction model is running ")
    features_train, labels, df, class1= read_data()
    features_train=features_train[:,selected_features]
    rf_out_file_name='final_pm_rf'
    pm_rf=RF_model(features_train, labels, df, class1,rf_out_file_name)
    rf_predict(pm_rf, features,order_id)
if __name__ == '__main__':

    # Pyinstaller fix
    multiprocessing.freeze_support()

    main()