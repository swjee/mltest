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

from titanic.load import load , MakeTopWord,GetTopWord,IsHave2Name,Address,OtherName


train_data_org,test_data_org = load()

list_address = ['mr', 'miss', 'mrs', 'master']
list_address_col = Address( train_data_org['Name'],list_address)
train_data_org['Address'] = pd.Series(list_address_col)
list_address_col = Address( test_data_org['Name'],list_address)
test_data_org['Address'] = pd.Series(list_address_col)

list_address_col = OtherName( train_data_org['Name'])
train_data_org['OtherName'] = pd.Series(list_address_col)
list_address_col = OtherName( test_data_org['Name'])
test_data_org['OtherName'] = pd.Series(list_address_col)


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

#train_data, varify_data = train_test_split(train_data_nn, test_size=0.2,random_state=42)
train_data, varify_data = train_test_split(train_data_nn, test_size=0.2)

from tensorflow import feature_column
from tensorflow.keras import layers


# 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Survived')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
# 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
def df_to_dataset_tst(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
#  labels = dataframe.pop('Survived')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

'''
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())
batch_size = 5 # 예제를 위해 작은 배치 크기를 사용합니다.
train_ds = df_to_dataset(train_data, shuffle=False, batch_size=batch_size)
varify_ds = df_to_dataset(varify_data, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset_tst(test_data_nn, shuffle=False, batch_size=batch_size)

example_batch = next(iter(train_ds))[0]
#example_batch = train_ds.as_dataset()
'''

age = feature_column.numeric_column("Age")

age_buckets = feature_column.bucketized_column(age, boundaries=[10,  50,  100])

Fare = feature_column.numeric_column("Fare")
Fare_buckets = feature_column.bucketized_column(Fare, boundaries=[50,500,1000])


sex_l = feature_column.categorical_column_with_vocabulary_list('Sex', ['male','female'])
sex_one_hot = feature_column.indicator_column(sex_l)


Cabin_l = feature_column.categorical_column_with_vocabulary_list('Cabin', train_data_nn['Cabin'].unique())
Cabin_one_hot = feature_column.indicator_column(Cabin_l)

Embarked_l = feature_column.categorical_column_with_vocabulary_list('Embarked', train_data_nn['Embarked'].unique())
Embarked_one_hot = feature_column.indicator_column(Embarked_l)

pclass_l = feature_column.categorical_column_with_vocabulary_list('Pclass', train_data_org['Pclass'].unique())
pclass_one_hot = feature_column.indicator_column(pclass_l)



#parch = feature_column.numeric_column("Parch")
feature_columns = []
feature_columns.append(age_buckets)
feature_columns.append(sex_one_hot)
feature_columns.append(pclass_one_hot)
#feature_columns.append(parch)
#feature_columns.append(Cabin_one_hot)
feature_columns.append(Embarked_one_hot)
feature_columns.append(Fare_buckets)
#feature_columns.append(feature_column.indicator_column(SibSp_x_Parch))
'''
for feature_name in ['SibSp','Parch']:
  vocabulary = train_data_nn[feature_name].unique()
  feature_columns.append(feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)))
'''

SibSp = feature_column.numeric_column("SibSp")
SibSp_buckets = feature_column.bucketized_column(SibSp, boundaries=[0,10])
feature_columns.append( SibSp_buckets )

Parch = feature_column.numeric_column("Parch")
Parch_buckets = feature_column.bucketized_column(Parch, boundaries=[0,10])
feature_columns.append( Parch_buckets)

Ticket_OneCh_l = feature_column.categorical_column_with_vocabulary_list('Ticket_OneCh', train_data_nn['Ticket_OneCh'].unique())
Ticket_OneCh_one_hot = feature_column.indicator_column(Ticket_OneCh_l)
feature_columns.append( Ticket_OneCh_one_hot)


Address_col = feature_column.categorical_column_with_vocabulary_list('Address',list_address)
Address_col_one_hot= feature_column.indicator_column(Address_col)
feature_columns.append( Address_col_one_hot)

OtherName_col = feature_column.categorical_column_with_vocabulary_list('OtherName',[1,0])
OtherName_col_one_hot= feature_column.indicator_column(OtherName_col)
feature_columns.append( OtherName_col_one_hot)

'''
for col in train_data_org.columns:
    print(col,'------------------')
    if col=='PassengerId' or col=='Name':
        continue
    else:
#        print( train_data_org[col].unique() )
'''

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#batch_size =140
#train_ds = df_to_dataset(train_data, shuffle=False ,batch_size=batch_size)
#varify_ds = df_to_dataset(varify_data, shuffle=False, batch_size=batch_size)
#test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

list_acc=[]
#train_data, varify_data = train_test_split(train_data_nn, test_size=0.2,random_state=42)
for i in range(1):

    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    batch_size =30
    train_ds = df_to_dataset(train_data, shuffle= True, batch_size=batch_size)
    varify_ds = df_to_dataset(varify_data, shuffle=False, batch_size=batch_size)
    #test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    model.fit(train_ds,validation_data=varify_ds,epochs=40,verbose=1)
    loss, accuracy = model.evaluate(varify_ds)
    print("정확도", accuracy)
    list_acc += [ accuracy ]


for i in list_acc:
    print(i)

print('mean is : ', np.mean(  np.array(list_acc) )  )

varify_data.count()

pr = model.predict(varify_ds)


a=[]
for i in iter(varify_ds):
    b= i[1].numpy() #label
    print ( i[1].numpy() )
    for j in b:
         a += [j]


vr_nparray=np.array(a)


pr_nparray=(pr.ravel()>0.5).astype(int)

cnt = (vr_nparray==pr_nparray).sum()

n,=pr_nparray.shape

print( cnt/n )


