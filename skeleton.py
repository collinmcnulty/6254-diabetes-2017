import pandas as pd
import numpy as np
import sklearn.ensemble
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression

def clean_separate(datafile): 
    # Read in datafile, sort based on patient number, remove duplicate patient entries 
    raw_data = pd.read_csv(datafile, index_col=1, header=0, na_values='?', low_memory=False)
    raw_data = raw_data.sort_index()
    raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
    raw_data = pd.get_dummies(raw_data) # switch to one-hot encoding
    
    # Split into training, verification and test sets
    fraction = {"training": 0.7, "verification": .05}
    target_column = 'readmitted_NO'
    related_columns = ['readmitted_<30', 'readmitted_>30']
    
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
    
    training_x, training_y = separate_target(training, target_column)
    testing_x, testing_y = separate_target(testing, target_column)
    verification_x, verification_y = separate_target(verification, target_column)
    return training_x, training_y, testing_x, testing_y, verification_x, verification_y 

def ml_fit(training_x, training_y, verification_x, verification_y): 
    # Multi-variable logistic regression 
    log_C = np.arange(0.01, 1.0, 0.1)
    score_vals=[]
    for C in log_C: 
        log_model = sklearn.linear_model.LogisticRegression(solver='newton-cg',C=C)
        log_model = log_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = log_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
        print log_model.n_iter_
        print curr_score
        
    logr_optimal_C = log_C[np.argmax(score_vals)]
    logr_optimal_score = np.max(score_vals)
    print "Logistic Regression C = %.2f score = %.4f" %(logr_optimal_C, logr_optimal_score)
    
    # Least Squares Regression 
    lsq_model = sklearn.linear_model.LinearRegression()
    lsq_model = lsq_model.fit(training_x.as_matrix(), training_y.as_matrix())
    score = lsq_model.score(verification_x.as_matrix(), verification_y.as_matrix())
    print "Least Squares Regression score = %.4f" %(score)
    
    # Ridge Regression 
    ridge_alpha = np.arange(0.01, 10, 0.2)
    score_vals = []
    for alpha in ridge_alpha: 
        ridge_model = sklearn.linear_model.Ridge(alpha=alpha)
        ridge_model = ridge_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = ridge_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
        print curr_score
        
    ridge_optimal_alpha = ridge_alpha[np.argmax(score_vals)]
    ridge_optimal_score = np.max(score_vals)
    print "Ridge Regression alpha = %.2f score = %.10f" %(ridge_optimal_alpha, ridge_optimal_score)
    
    # LASSO 
    lasso_alpha = np.arange(0.0001, 0.1, 0.001)
    score_vals = []
    for alpha in lasso_alpha: 
        lasso_model = sklearn.linear_model.Lasso(alpha=alpha)
        lasso_model = lasso_model.fit(training_x.as_matrix(), training_y.as_matrix())
        curr_score = lasso_model.score(verification_x.as_matrix(), verification_y.as_matrix())
        score_vals.append(curr_score)
        print curr_score
    
    lasso_optimal_alpha = lasso_alpha[np.argmax(score_vals)]
    lasso_optimal_score = np.max(score_vals)
    print "LASSO alpha = %.2f score = %.10f" %(lasso_optimal_alpha, lasso_optimal_score)
    
    return log_model, lsq_model, ridge_model, lasso_model
    
def ml_predict(testing_x, testing_y, log_model, lsq_model, ridge_model, lasso_model): 
    # Logistic Regression
    log_predicted_y = log_model.predict(testing_x.as_matrix())
    log_score = log_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    
    # Least Squares Regression 
    lsq_predicted_y = lsq_model.predict(testing_x.as_matrix())
    lsq_score = lsq_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    
    # Ridge Regression 
    ridge_predicted_y = ridge_model.predict(testing_x.as_matrix())
    ridge_score = ridge_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    
    # Lasso 
    lasso_predicted_y = lasso_model.predict(testing_x.as_matrix())
    lasso_score = lasso_model.score(testing_x.as_matrix(), testing_y.as_matrix())
    
    print "Logistic Regression score = %.10f" % log_score
    print "Least Squares Regression score = %.10f" % lsq_score
    print "Ridge Regression score = %.10f" % ridge_score 
    print "LASSO Regression score = %.10f" % lasso_score

datafile = 'dataset_diabetes/diabetic_data.csv'
training_x, training_y, testing_x, testing_y, verification_x, verification_y = clean_separate(datafile)
# do machine learning
log_model, lsq_model, ridge_model, lasso_model = ml_fit(training_x, training_y, verification_x, verification_y)
ml_predict(testing_x, testing_y, log_model, lsq_model, ridge_model, lasso_model)

# TODO: Clean up input data, tune parameters of models, create plots/visuals of results
# write out results and plot 
    

