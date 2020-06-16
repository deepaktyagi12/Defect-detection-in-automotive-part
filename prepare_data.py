import os
import numpy as np
import shutil
import random
import argparse
'''
This pyhton code utilize to prepare the dataset into "Training" and "testing" set.
The dataset contains two classes: "Healthy" and "Defected".
'''
def prepare_taining_testing_data(root_dir,classes_dir,test_ratio):
    
    for cls in classes_dir:
        training_dir=root_dir +'/train' + cls
        testing_dir=root_dir +'/test' + cls
        if os.path.exists(training_dir):
            print("Date Directory is already exist")
        else:
            os.makedirs(training_dir)
        if os.path.exists(testing_dir):
            print("Date Directory is already exist")
        else:
            os.makedirs(testing_dir)

        src = root_dir + cls # Folder to copy images from
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*(1-test_ratio))])
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

        print('Total images in dataset is: ', len(allFileNames))
        print('Images in Training set is: ', len(train_FileNames))
        print('Images in Testing set is: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir +'/train' + cls)

        for name in test_FileNames:
            shutil.copy(name, root_dir +'/test' + cls)
def main():
    '''
    There are three parameters
    1. Data set path-- default='dataset/YE358311_Fender_apron'
    2. Different classes in the dataset-- default=['/YE358311_defects', '/YE358311_Healthy'],
    3.Divide data into training and testing set "--test_data_ratio"and "-test_data_ratio", default=0.10,
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--Data_Directory", "-datadir", default='dataset/YE358311_Fender_apron',
        help="Path to test data")
    ap.add_argument("--classes_dir", "-classes_dir", default=['/YE358311_defects', '/YE358311_Healthy'],
        help="Different data class")
    ap.add_argument("--test_data_ratio", "-test_data_ratio", default=0.10,
        help="Divide data into training and testing set")
    args = vars(ap.parse_args())
   
    data_dir =  args["Data_Directory"]
    classes_dir = args['classes_dir']
    test_ratio = args['test_data_ratio']
    prepare_taining_testing_data(data_dir,classes_dir,test_ratio)

if __name__== "__main__":
  main()
