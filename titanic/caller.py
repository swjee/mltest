#import load as ld
from load import load , MakeTopWord,GetTopWord,IsHave2Name,Address,OtherName

train_data_org,test_data_org = load()

print( train_data_org.shape)
print( train_data_org.shape)

print(train_data_org.columns)


print ( train_data_org.head() )

dic_li = MakeTopWord(train_data_org['Name'])
print(dic_li)
list_address = GetTopWord(dic_li,4)
print( list_address)

print( IsHave2Name('asdf cdef(hello world)'))
print( IsHave2Name('asdf cdef'))
# add new field 'Address' , 'OtherName'

print(Address(['mr Kim','asfd','master Lee'],list_address))


print( OtherName( ['Kim','Kim(aaa)','Lee','Lee(asdf)']))

#org data --> add 2 column Address , OtherName

import pandas as pd

list_address_col = Address( train_data_org['Name'],list_address)
train_data_org['Address'] = pd.Series(list_address_col)
list_address_col = Address( test_data_org['Name'],list_address)
test_data_org['Address'] = pd.Series(list_address_col)

list_address_col = OtherName( train_data_org['Name'])
train_data_org['OtherName'] = pd.Series(list_address_col)
list_address_col = OtherName( test_data_org['Name'])
test_data_org['OtherName'] = pd.Series(list_address_col)

print(train_data_org)

train_data_org.describe()

