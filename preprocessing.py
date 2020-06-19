import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0으로 설정하면 모든 메시지를 보여줍니다.(디폴트 상태)

# 1로 설정하면 INFO 메시지를 숨깁니다.

# 2로 설정하면 INFO, WARNINGS 메시지를 숨깁니다.

# 3으로 설정하면 INFO, WARNINGS, ERROR 메시지를 숨깁니다.
#titanic.. test
'''
1.titanic data load from local disc

train data :  C:\MLTEST\handson-ml-master\datasets\titanic\train.csv
test  data :  C:\MLTEST\handson-ml-master\datasets\titanic\test.csv

'''

TRAIN_DATA_FILE =  "C:\\MLTEST\\handson-ml-master\\datasets\\titanic\\train.csv"
TEST_DATA_FILE =  "C:\\MLTEST\\handson-ml-master\\datasets\\titanic\\test.csv"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
#import tensorflow.feature_column as fc



train_data_org = pd.read_csv(TRAIN_DATA_FILE)
test_data_org = pd.read_csv(TEST_DATA_FILE)

#train_data.groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('% survive')


#Age의 Null값 변환.

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

preprocess_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, ["Age"]),
        ("cat_pipeline", cat_pipeline, ["Cabin","Embarked"])
    ])

train_data_age_cabin_nn = pd.DataFrame( preprocess_pipeline.fit_transform(train_data_org) )
test_data_age_cabin_nn = pd.DataFrame( preprocess_pipeline.fit_transform(test_data_org) )


train_data_nn = train_data_org
train_data_nn['Age'] = train_data_age_cabin_nn[0].astype(int)
train_data_nn['Cabin'] = train_data_age_cabin_nn[1]
train_data_nn['Embarked'] = train_data_age_cabin_nn[2]
test_data_nn = test_data_org
test_data_nn['Age'] = test_data_age_cabin_nn[0].astype(int)
test_data_nn['Cabin'] = test_data_age_cabin_nn[1]
test_data_nn['Embarked'] = test_data_age_cabin_nn[2]

list_ticket_onec=[]
for i in train_data_org['Ticket'][:].to_numpy():
    if i[0] >='0' and i[0] <='9':
        list_ticket_onec += [ '0' ]
    else:
        list_ticket_onec += [ i[0] ]


dic_ticket_onec={'Onech':list_ticket_onec}
DF_ticket_onec = pd.DataFrame( dic_ticket_onec)
train_data_nn['Ticket_OneCh']=DF_ticket_onec['Onech']


for i in train_data_nn.columns:
    print(i, train_data_nn[i].isnull().values.any())

# train_data_nn['Age'].isnull().values.any()

from sklearn.model_selection import train_test_split

train_data, varify_data = train_test_split(train_data_nn, test_size=0.2,random_state=42)




