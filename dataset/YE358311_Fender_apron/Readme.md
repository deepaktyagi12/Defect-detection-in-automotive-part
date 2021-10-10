Problem Statement:
____________________________________________________________________________________________________________________________________________________________________________________________________________
Customers send back a substantial part of the products that they purchase online. Return shipping is expensive for online platforms and return orders are said to reach 50% for 
certain industries and products. Nevertheless, free or inexpensive return shipping has become a customer expectation and de-facto standard in the fierce online competition on clothing,
but shops have indirect ways to influence customer purchase behaviour. For purchases where return seems likely, a shop could, for example, restrict payment options or display additional 
marketing communication.
Data Set:
TrainingData_v1 - To be used for training and testing
TestingData_For_Candidate - To be used for predicting the output (this will be compared against the results that we have retained)
_____________________________________________________________________________________________________________________________________________________________________________________________________________
 
In this assignment, machine learning-based approaches for predicting user decision using XGBoost, Random Forest and Support vetctor machine  are implemented,
and Permutation importance technique using Random Forest is used for selecting the optimal feature subset.

_____________________________________________________________________________________________________________________________________________________________________________________________________________
1. Pre-requisites:
    Python 3.8
    pip install sklearn
    pip install xgboost
    pip install numpy

2. The Project is distributed with the following structure.
	I.   Data:   		consists of training and testing dataset and artifacts.
	II.  Results:    	Predicted results are saved in this directory (csv and pdf file).
	III. Python script:	"Cirp_main.py"    is the main script of this project which is used to prepare the dataset and fit the models.
				"Cirp_predict.py" is used to predict the user decision on test dataset and saved into results directory (xgb_predict_results, rf_predict_results,svm_predict_results). 
				"Feature_relevance_model.py" relevent score and feature selection function is written in this srcipt. 
				"xgb_prediction_model.py" XGboost prediction model training and evalutaion code is written in this srcipt. 
				"Rf_prediction_model.py" random forest prediction model training and evalutaion code is written in this srcipt. 
				"svm_prediction_model.py" SVM model is implemented in this srcipt. 
    
3. Run:
	For Training:
		python Cirp_main.py

	For Prediction:
		python Cirp_predict.py

4. Results:
       
AUC (%)		Random forest 	XGBoost	 SVM 
With PM		73.18		74.29	67.03
Without PM 	66.54		68.85	63.42
