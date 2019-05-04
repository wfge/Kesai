# -*- coding: utf-8 -*-
# @Time    : 2019/5/4 20:44
# @Author  : SU
# @Email   : gewanfeng@whu.edu.cn
# @File    : preapre_data.py
# @Software: PyCharm
import  pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV



train=pd.read_csv('./train.csv',engine='python',encoding='utf-8')
test=pd.read_csv('./test.csv',engine='python',encoding='utf-8')
# print(train.head())

format_label=lambda x:1 if x=='Positive' else 0

label=train["label"].map(format_label)
# print(label.head())

train_data=[]

for i in range(len(train["review"])):
        train_data.append(train["review"][0])

test_data=[]
for i in range(len(test["review"])):
    test_data.append((test["review"][0]))


tfidf=TFIDF(min_df=2,max_features=None,strip_accents='unicode',analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1,3),
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1,
            # stop_words='english'
            )

data_all=train_data+test_data
len_train=len(train_data)

tfidf.fit(data_all)
data_all=tfidf.transform(data_all)

train_x=data_all[0:len_train]
test_x=data_all[len_train:]

# print(train_x[0:10])
# print(test_x[0:10])
grid_values={'C':[30]}
model_LR=GridSearchCV(LR(penalty='L2',dual=True,random_state=0),grid_values=0,tol=0.0001)



