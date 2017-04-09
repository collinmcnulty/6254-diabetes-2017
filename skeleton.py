import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

def clean_separate(datafile): 
    # Read in datafile, sort based on patient number, remove duplicate patient entries 
    raw_data = pd.read_csv(datafile, index_col=1, header=0, na_values='?', low_memory=False)
    raw_data = raw_data.sort_index()
    raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
    raw_data = pd.get_dummies(raw_data) # switch to one-hot encoding
    
    # Split into training, verification and test sets
    fraction = {"training": 0.7, "verification": .05}
    target_column = 'readmitted_NO'
    related_columns = ['readmitted_<30', 'readmitted_>30', 'encounter_id', 
                       'payer_code_UN', 'payer_code_WC', 'payer_code_MC', 'payer_code_PO', 
                       'payer_code_OG', 'payer_code_MD', 'payer_code_HM', 'payer_code_DM', 
                       'payer_code_CP', 'payer_code_CM', 'payer_code_CH', 'payer_code_BC',
                       'weight_>200','weight_[0-25)', 'weight_[100-125)', 'weight_[125-150)',
                       'weight_[150-175)', 'weight_[175-200)', 'weight_[25-50)',
                       'weight_[50-75)', 'weight_[75-100)', 'payer_code_FR', 'payer_code_MP',
                       'payer_code_OT', 'payer_code_SI', 'payer_code_SP',]
    
    cutoffs = []
    cutoffs.append(int(len(raw_data) * fraction["training"]))
    cutoffs.append(cutoffs[-1] + int(len(raw_data) * fraction["verification"]))
    cutoffs.append(len(raw_data))
    
    training = raw_data[0:cutoffs[0]]
    verification = raw_data[cutoffs[0]:cutoffs[1]]
    testing = raw_data[cutoffs[1]:cutoffs[2]]
    
    def separate_target(df, target_name):
        train = df.drop(target_name, axis=1)
        target = df[target_name]
        return train, target
    
    def remove_related_targets(df):
        df_out = df.copy()
        for col in related_columns: 
            df_out = df_out.drop(col, axis=1)
        return df_out
    
    training = remove_related_targets(training)
    testing = remove_related_targets(testing)
    verification = remove_related_targets(verification)
    
    scaler = sklearn.preprocessing.StandardScaler()
    training.update(scaler.fit_transform(training.as_matrix()))
    verification.update(scaler.transform(verification.as_matrix()))
    testing.update(scaler.transform(testing.as_matrix()))
    
    training_x, training_y = separate_target(training, target_column)
    testing_x, testing_y = separate_target(testing, target_column)
    verification_x, verification_y = separate_target(verification, target_column)
    return training_x, training_y, testing_x, testing_y, verification_x, verification_y 

def ml_fit(training_x, training_y, verification_x, verification_y): 
    # Multi-variable logistic regression 
    log_C = np.arange(1e-3, 1e-1, 1e-3)
    score_vals=[]
    for C in log_C: 
        log_model = sklearn.linear_model.LogisticRegression(solver='newton-cg',C=C, max_iter=100)
        log_model = log_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = log_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
        
    logr_optimal_C = log_C[np.argmax(score_vals)]
    logr_optimal_score = np.max(score_vals)
    print "Logistic Regression C = %.4f score = %.4f" %(logr_optimal_C, logr_optimal_score)
    
    # Least Squares Regression 
    lsq_model = sklearn.linear_model.LinearRegression()
    lsq_model = lsq_model.fit(training_x.as_matrix(), training_y.as_matrix())
    lsq_predicted_y = np.round(lsq_model.predict(verification_x.as_matrix()))
    curr_score = score(lsq_predicted_y, verification_y.as_matrix())
    print "Least Squares Regression score = %.10f" %(curr_score)
    
    # Ridge Regression 
    ridge_alpha = np.arange(1e-9, 1e-5, 1e-6)
    score_vals = []
    for alpha in ridge_alpha: 
        ridge_model = sklearn.linear_model.Ridge(alpha=alpha)
        ridge_model = ridge_model.fit(training_x.as_matrix(), training_y.as_matrix())
        ridge_predicted_y = np.round(ridge_model.predict(verification_x.as_matrix()))
        curr_score = score(ridge_predicted_y, verification_y.as_matrix())
        score_vals.append(curr_score)
        
    ridge_optimal_alpha = ridge_alpha[np.argmax(score_vals)]
    ridge_optimal_score = np.max(score_vals)
    print "Ridge Regression alpha = %.10f score = %.10f" %(ridge_optimal_alpha, ridge_optimal_score)
    
    # LASSO 
    lasso_alpha = np.arange(0.0001, 0.1, 0.001)
    score_vals = []
    for alpha in lasso_alpha: 
        lasso_model = sklearn.linear_model.Lasso(alpha=alpha)
        lasso_model = lasso_model.fit(training_x.as_matrix(), training_y.as_matrix())
        lasso_predicted_y = np.round(lasso_model.predict(verification_x.as_matrix()))
        curr_score = score(lasso_predicted_y, verification_y.as_matrix())
        score_vals.append(curr_score)
    
    lasso_optimal_alpha = lasso_alpha[np.argmax(score_vals)]
    lasso_optimal_score = np.max(score_vals)
    print "LASSO alpha = %.4f score = %.10f" %(lasso_optimal_alpha, lasso_optimal_score)
    
    # Knn
    knn_k = np.arange(7, 20, 1)
    score_vals = []
    for k in knn_k: 
        knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        knn_model = knn_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = knn_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
    
    knn_optimal_k = knn_k[np.argmax(score_vals)]
    knn_optimal_score = np.max(score_vals)
    print "Knn k = %d score = %.10f" %(knn_optimal_k, knn_optimal_score)
      
    #SVC
    svc_c = np.arange(-500 , -5, 50)
    svc_gamma = np.arange(1e-9, 1e-4, 1e-5)
    score_vals = []
    for c in svc_c: 
        svc_model = sklearn.svm.SVC(C=0.001,kernel='poly',degree=1,coef0=1.0)
        svc_model = svc_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = svc_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
    
    svc_optimal_c = svc_c[np.argmax(score_vals)]
    svc_optimal_gamma = svc_gamma[np.argmax(score_vals)]
    svc_optimal_score = np.max(score_vals)
    print "SVC c = %.3f gamma = %.7f score = %.10f" %(svc_optimal_c, svc_optimal_gamma, svc_optimal_score)
    
    return log_model, lsq_model, ridge_model, lasso_model, knn_model, svc_model
    
def ml_predict(testing_x, testing_y, log_model, lsq_model, ridge_model, lasso_model, knn_model, svc_model): 

    # Logistic Regression
    log_predicted_y = log_model.predict(testing_x.as_matrix())
    log_score = log_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, "bo", label='Actual Y')
    plt.plot(testing_x, log_predicted_y, "rs", label='Predicted Y')
    plt.title('Logistic Regression')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    plt.show()
    
    # Least Squares Regression 
    lsq_predicted_y = np.round(lsq_model.predict(testing_x.as_matrix()))
    lsq_score = score(lsq_predicted_y,testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, 'bo', testing_x, lsq_predicted_y, 'rs')
    plt.title('Least Squares Regression')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    plt.show()
    
    # Ridge Regression 
    ridge_predicted_y = np.round(ridge_model.predict(testing_x.as_matrix()))
    ridge_score = score(ridge_predicted_y, testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, 'bo', testing_x, ridge_predicted_y, 'rs')
    plt.title('Ridge Regression')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    plt.show()
    
    # Lasso 
    lasso_predicted_y = np.round(lasso_model.predict(testing_x.as_matrix()))
    lasso_score = score(lasso_predicted_y, testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, 'bo', testing_x, lasso_predicted_y, 'rs')
    plt.title('LASSO Regression')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    plt.show()

    # K-nn
    knn_predicted_y = knn_model.predict(testing_x.as_matrix())
    knn_score = knn_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, 'bo', testing_x, knn_predicted_y, 'rs')
    plt.title('K-nn')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    plt.show()
    
    # SVC
    svc_predicted_y = svc_model.predict(testing_x.as_matrix())
    svc_score = svc_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    plt.figure()
    plt.plot(testing_x, testing_y, 'bo', testing_x, svc_predicted_y, 'rs')
    plt.title('SVC')
    plt.xlabel('Feature')
    plt.ylabel('Predicted and Actual Result')
    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    plt.show()
    
    print "Logistic Regression score = %.10f" % log_score
    print "Least Squares Regression score = %.10f" % lsq_score
    print "Ridge Regression score = %.10f" % ridge_score 
    print "LASSO Regression score = %.10f" % lasso_score
    print "K-nn score = %.10f" % knn_score
    print "SVC score = %.10f" % svc_score
    
    def find_best_classifier():
        classifiers = ([log_model, lsq_model, ridge_model, lasso_model, knn_model, svc_model])
        classifier_names = (["logistic regression", "least squares regression", "ridge regression", "LASSO", "K-nn", "SVC"])
        scores = ([log_score, lsq_score, ridge_score, lasso_score, knn_score, svc_score])
        best_classifier = classifiers[np.argmax(scores)]
        print classifier_names[np.argmax(scores)]
        return best_classifier
    return find_best_classifier()
    
def random_forest_feature_importance(X, Y):
    def plot_feature_importance(importances, testing_x, testing_y):
    # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(testing_x.shape[1]), importances, color="r", align="center")
        plt.xticks(range(testing_x.shape[1]))
        plt.xlim([-1, testing_x.shape[1]])
        plt.show()
    rf = sklearn.ensemble.RandomForestRegressor()
    rf.fit(X, Y)
    importances = rf.feature_importances_
    plot_feature_importance(importances, X, Y)
    indices = np.argsort(importances)
    num_features_keeping = 10
    ordered_importances = importances[indices]
    least_important = ordered_importances[:-num_features_keeping]
    least_important_indices = indices[:-num_features_keeping]
    best_indices = np.argsort(importances)[::-1]
    return least_important_indices, least_important, best_indices

def score(predicted_y, true_y): 
    correct = np.sum(predicted_y == true_y)
    return correct / float(len(predicted_y))
    
# Read in the datafile and clean it
np.random.seed(6254)
datafile = 'dataset_diabetes/diabetic_data.csv'
training_x, training_y, testing_x, testing_y, verification_x, verification_y = clean_separate(datafile)

# Extract the features of most importance
indices, importances, best_indices = random_forest_feature_importance(training_x, training_y)
feature_list = list(training_x)
best_features = np.array(feature_list)[best_indices[0:10]]
print best_features

# Remove unimportant features 
testing_x = testing_x.drop(list(np.array(feature_list)[indices]), axis=1)
verification_x = verification_x.drop(list(np.array(feature_list)[indices]), axis=1)
training_x = training_x.drop(list(np.array(feature_list)[indices]), axis=1)

# do machine learning
log_model, lsq_model, ridge_model, lasso_model, knn_model, svc_model = ml_fit(training_x, training_y, verification_x, verification_y)
best_classifier = ml_predict(testing_x, testing_y, log_model, lsq_model, ridge_model, lasso_model ,knn_model, svc_model)

# Determine success of random classifier 
random_y = np.random.randint(0, 2, size=len(testing_y))
random_classifier_score = score(random_y, testing_y)
print "Random Classifier score = %.10f" % random_classifier_score

# TODO:  Determine the ratio of admittance in testing, verification, and training groups 
#        Randomly assigning predictions to the test group and compare to our best score 
#        
#        create plots/visuals of results

