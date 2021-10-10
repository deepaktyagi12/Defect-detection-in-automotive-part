from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

def feature_importance(features, labels, data):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.4, random_state=5)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 3)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    rf_best_param=rf_random.best_params_
    rf_model=rf_random.best_estimator_
    
    y_pred = rf_model.predict(X_test)
    accuracy=metrics.accuracy_score(y_test,y_pred)
    print('Acuracy: %f' %accuracy,rf_best_param)
    rf_importance=rf_model.feature_importances_
    rf_feature_list= rf_model.feature_importances_.argsort()
    feture_impt=[rf_importance,rf_feature_list]
    np.savetxt('data/rf_importance_v_01.csv', feture_impt, delimiter=',')
        
    plt.barh(rf_feature_list, rf_importance[rf_feature_list])
    plt.xlabel("Random Forest Feature Importance")
    plt.savefig('results/RF_feature_chart_v_01.pdf')
    
    
    perm = permutation_importance(rf_model, X_test, y_test)
    perm_importance=perm.importances_mean
    pm_feature_list = perm_importance.argsort()
    
    # print("pm feature importance", perm_importance)
    # print("pm feature list",pm_feature_list)

    plt.barh(pm_feature_list, perm_importance)
    plt.xlabel("Permutation Feature Importance")
    plt.savefig('results/PM_RF_feature_chart_v_01.pdf')
    np.savetxt('data/PM_rf_importance_v_01.csv', perm_importance, delimiter=',')
    
    relevence_mean=np.mean(perm_importance)
    selected_features=[]
    print('relevence_mean',relevence_mean,selected_features)
    for i in pm_feature_list :
        if perm_importance[i]>0:
            selected_features.append(i)
    print(selected_features) 
    
    features=features[:,selected_features]
    print(features.size)
    
    return features, selected_features

