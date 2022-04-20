# Data Manipulation 
import numpy as np
import pandas as pd
import math
# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Machine learning 
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model, neighbors, svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, cross_val_score
from vecstack import stacking
from xgboost import XGBClassifier 

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')

#사용자 함수 ---------------------------------------------------------------
def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)

def get_category(Age):
    cat=''
    if Age <= -1 : cat = 'Unknown'
    elif Age <=5 : cat = 'Baby'
    elif Age <=12 : cat = 'Child'
    elif Age <=18 : cat = 'Teenager'
    elif Age <=25 : cat = 'Student'
    elif Age <=35 : cat = 'Young Adult'
    elif Age <=60 : cat = 'Adult'
    else : cat = 'Ederly'
    
    return cat
#-------------------------------------------------------------------------

# Target 데이터, Feature 데이터 구분
data_df = pd.read_csv('titanic.csv')
x_data = data_df.copy()

# 호칭 열 추가
name_title = pd.DataFrame(i.split(',')[1].split(' ')[1] for i in x_data['Name'])
name_title

for title in name_title.iterrows():
    if title[1][0] not in ['Mr.','Mrs.','Miss.','Master.','Dr.']:
        name_title.iloc[title[0]]='Others'

x_data['NameTitle'] =  name_title

#각각의 X_data 열과 생존률(Survived)의 관계
data_df_v=data_df[['SibSp','Parch','Pclass','Sex','Embarked','Survived']].astype(object)
plot_bivariate_bar(data_df_v, hue='Survived', cols=3, width=20, height=12, hspace=0.4, wspace=0.5)

# Pcalss별 생존자수 그래프
data_df["Survived(humanized)"]=data_df["Survived"].replace(0, "Perish").replace(1,"Survived")
data_df["Pclass(humanized)"]=data_df["Pclass"].replace(1,"First Class").replace(2, "Second Class").replace(3, "Third Class")
sns.countplot(data=data_df, x="Pclass(humanized)", hue="Survived(humanized)")

# Embarked 별 생존자수 그래프
sns.countplot(data=data_df, x="Embarked", hue="Survived(humanized)")

# Sex 별 생존자수 그래프
sns.countplot(data=data_df, x="Sex", hue="Survived(humanized)")

# Family 별 생존자수 그래프
data_df['FamilyNum'] = data_df['SibSp']+data_df['Parch']
sns.countplot(data=data_df, x="FamilyNum", hue="Survived(humanized)")

# Name 호칭 별 생존자수 그래프
# 호칭 열 추가
name_title = pd.DataFrame(i.split(',')[1].split(' ')[1] for i in data_df['Name'])
name_title

for title in name_title.iterrows():
    if title[1][0] not in ['Mr.','Mrs.','Miss.','Master.','Dr.']:
        name_title.iloc[title[0]]='Others'

data_df['NameTitle'] =  name_title

sns.countplot(data=data_df, x="NameTitle", hue="Survived(humanized)")

# Age 별 생존자수 그래프
# 타이타닉에 탑승한 전체적인 age 비율입니다. 20-40대가 가장 많음을 알 수 있습니다.
fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(data_df['Age'], bins=25, ax=ax)
plt.show()

# 나이 대 별로 그룹화 하여 분류한 후 그래프를 그려보면  어린 여자 아이일수록 생존률이 높음을 알 수 있습니다.
plt.figure(figsize=(10,6))
group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Ederly']
data_df['Age_cat']=data_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=data_df, order=group_names)

# Fare 별 생존자수 그래프
# 전체 Fare 분포도 입니다. 3rd 클라스 승객이 가장 많은 것을 확인했듯이, 요금이 적을수록 승객의 분포도가 높았습니다.
fig, ax = plt.subplots(figsize=(10,6))
sns.distplot(data_df['Fare'], bins=10, ax=ax)
plt.show()

# Cabin별 생존자수 그래프
# 시각화만을 위한 전처리 
cabin_visual_df = pd.read_csv('titanic.csv')
cabin_visual_df['Cabin'] = cabin_visual_df['Cabin'].str[:1]
cabin_visual_df['Cabin'] = cabin_visual_df['Cabin'].fillna('Na')
# Cabin의 생존별 그래프
Cabin1 = cabin_visual_df[cabin_visual_df['Cabin']=='A']['Survived'].value_counts()
Cabin2 = cabin_visual_df[cabin_visual_df['Cabin']=='B']['Survived'].value_counts()
Cabin3 = cabin_visual_df[cabin_visual_df['Cabin']=='C']['Survived'].value_counts()
Cabin4 = cabin_visual_df[cabin_visual_df['Cabin']=='D']['Survived'].value_counts()
Cabin5 = cabin_visual_df[cabin_visual_df['Cabin']=='E']['Survived'].value_counts()
Cabin6 = cabin_visual_df[cabin_visual_df['Cabin']=='F']['Survived'].value_counts()
Cabin7 = cabin_visual_df[cabin_visual_df['Cabin']=='G']['Survived'].value_counts()
Cabin8 = cabin_visual_df[cabin_visual_df['Cabin']=='T']['Survived'].value_counts()
Cabin9 = cabin_visual_df[cabin_visual_df['Cabin']=='N']['Survived'].value_counts()
df = pd.DataFrame([Cabin1, Cabin2, Cabin3,Cabin4,Cabin5,Cabin6,Cabin7,Cabin8,Cabin9])
df.index = ['A', 'B', 'C','D','E','F','G','T','Na']
df.plot(kind='bar', stacked=True, figsize=(10, 5))

# 남녀 나이대별 생존자, 사망자 비교
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
women = data_df[data_df['Sex']=='female']
men = data_df[data_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")
ax.legend()
ax.set_title('Male')