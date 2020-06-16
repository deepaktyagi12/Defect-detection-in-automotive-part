There are various mathematical or deterministic models that can be used to the defect in the automotive spare parts images. But, in the recent past, deep neural network models have outperformed the other models. This document describes how the deep learning model can be utilized to classify the automotive spare parts images whether the spare part is "Healthy" or "Defected".
____________________________________________________________________________
1. Prerequisites:
    Python 3.7
    Scipy
    Numpy
    opencv-python
    Scikit-image
    numpy
    Tensorflow 1.14 with NVIDIA GPU or CPU

2. The Project is distributed with the following structure.
    1. "dataset":          contains the python code through which you can execute this project.
    2. "Trained_model":    trained models are saved in this directory (.h5, .pb, .tflite models).
    3. "Python code file": contains the python code through which you can execute this project.
            I. "prepare_data.py": This python code utilizes to prepare the dataset into "Training" and "testing" sets.
            II. "main.py": This is the main code file of this project. Run this file to train model and test model performance.

    
3. Run:
I). download dataset "YE358311_Fender_apron" and copy into the "dataset" directory:
                    https://drive.google.com/file/d/1k57jP_oy4c9VDZmlgqCvfErzVTzPeA_M/view?usp=sharing

II). Divide the dataset into "Training" and "testing" sets. The dataset contains two classes: "Healthy" and "Defected".
        Run "python prepare_data.py"

III). To train a deep learning model, run the following command:
            Run "python main.py --opMode 'Train'"
   Output: The trained models will be saved in the "Trained_model" folder.


IV). To Evaluate the performance of trained  model on "Test dataset", run the following command:
							Run  "python main.py --opMode 'Test'". 

V). Output would be like this:
        
        
        True Labels [1 0 1 0 0 0 1 0 0 0 1 1 1 0 0 0 1 1 1 1 0 0 1 0 1 0]
	Predicted Labels [1 0 1 0 0 0 1 0 0 0 1 1 1 0 0 0 1 1 1 1 0 0 1 0 1 0]
	Toatal Accuracy: 1.000000
	Confusion matrix: [[14  0]
	 		  [ 0 12]]
	Precision: 1.000000
	Recall: 1.000000
	F1 score: 1.000000
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy
	True Label:--Defected  Predictive Label:--Defected
	True Label:--Healthy   Predictive Label:--Healthy

