import pandas as pd
import numpy as np
import sklearn.ensemble

# Import data. Changed '?' to NaN
datafile = 'dataset_diabetes/diabetic_data.csv'
#datafile = 'test.csv'
raw_data = pd.read_csv(datafile, index_col=0, na_values='?')
raw_data = pd.get_dummies(raw_data) # switch to one-hot encoding

# encounter_id is the index. The following two lines are equivalent:
# a = raw_data.iloc[[2]]
# b = raw_data.loc[[64410]]
# To access a specific value, use this syntax. note that 148530 is an index label
# b = raw_data['race'][148530]

# Split into training, verification and test sets
fraction = {"training": 0.7, "verification": .05}
target_column = 'readmitted_NO'

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

training_x, training_y = separate_target(training, target_column)
testing_x, testing_y = separate_target(testing, target_column)
verification_x, verification_y = separate_target(training, target_column)



# Write ML stuff here
# rf = sklearn.ensemble.RandomForestClassifier()
# rf.fit(training_x, training_y)
