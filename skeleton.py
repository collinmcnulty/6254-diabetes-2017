import pandas as pd
import numpy as np
import sklearn as sk

# Import data. Changed '?' to NaN
datafile = 'dataset_diabetes/diabetic_data.csv'
#datafile = 'test.csv'
raw_data = pd.read_csv(datafile, index_col=0, na_values='?')

# encounter_id is the index. The following two lines are equivalent:
# a = raw_data.iloc[[2]]
# b = raw_data.loc[[64410]]
# To access a specific value, use this syntax. note that 148530 is an index label
# b = raw_data['race'][148530]

# Split into training, verification and test sets
fraction = {"training": 0.7, "verification": .05}

cutoffs = []
cutoffs.append(int(len(raw_data) * fraction["training"]))
cutoffs.append(cutoffs[-1] + int(len(raw_data) * fraction["verification"]))
cutoffs.append(len(raw_data))

training = raw_data[0:cutoffs[0]]
verification = raw_data[cutoffs[0]:cutoffs[1]]
testing = raw_data[cutoffs[1]:cutoffs[2]]

a=pd.get_dummies(raw_data)
1