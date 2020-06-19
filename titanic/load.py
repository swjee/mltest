#load titanic csv file to dataframe...

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

def load():
    train_data_org = pd.read_csv(TRAIN_DATA_FILE)
    test_data_org = pd.read_csv(TEST_DATA_FILE)
    print('id of load ft',id(train_data_org))
    print('inside load ft',train_data_org.shape)
    return train_data_org,test_data_org

import re
def MakeTopWord(list_data):
    dic_word_list={}

    pattern_txt = '\\w+'
    pattern_word = re.compile(pattern_txt, re.I)

    for str_data in list_data:
        '''first to lower case.'''
        list_oneline = pattern_word.findall(str_data.lower())
        # print(list_oneline)

        for word in list_oneline:
            if word in [ 'mr','miss','mrs','master']:
                if word in dic_word_list:
                    dic_word_list[word] += 1
                else:
                    dic_word_list[word] = 1

    return dic_word_list


TOP_WORD_COUNT = 20
def GetTopWord(dic_word,top_word_cnt=20):
    word_series = pd.Series(list(dic_word.keys()))
    count_series = pd.Series(list(dic_word.values()))

    pd_word_list = pd.DataFrame({'word': word_series, 'count': count_series})
    A = pd_word_list.sort_values(by=['count'], ascending=False)

    chk_word = list(A['word'][0:top_word_cnt].to_numpy())
    return chk_word


def IsHave2Name(str_name):
    pattern_txt = '\(.*\)'
    pattern_word = re.compile(pattern_txt, re.I)

    list_oneline = pattern_word.findall(str_name.lower())
    if len(list_oneline) > 0:
        return True
    else:
        return False

def Address(list_data,chk_word_list):
#if not in nothing assign
    list_result=[]
    for name in list_data:
        pattern_txt = '\\w+'
        pattern_word = re.compile(pattern_txt, re.I)
        list_oneline = pattern_word.findall(name.lower())

        bFound=False
        for word in list_oneline:
            if chk_word_list.count(word) > 0:
                bFound=True
                list_result += [word]
                break
        if bFound == False :
            list_result += ['nothing']

    return list_result

def OtherName( list_data ):
    list_result=[]
    for name in list_data:
        list_result += [IsHave2Name(name)]

    return list_result
