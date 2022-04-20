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

# 사용자 함수---------------------------------------------------------
# 결측치 평균으로 채워 넣는 함수
def fillna_with_mean(df, col_name):
    
    na_rows = df[df[col_name].isnull()]
    not_na_rows = df[df[col_name].notnull()]
    
    value_fill = not_na_rows[col_name].mean()
    
    df[col_name] = df[col_name].fillna(value_fill)
    
    return df

# 결측치 비중이 높은 cat으로 채워 넣는 함수
def fillna_with_mode(df, col_name):
    
    na_rows = df[df[col_name].isnull()]
    not_na_rows = df[df[col_name].notnull()]
    
    value_fill = not_na_rows[col_name].value_counts().index[0]
    
    df[col_name] = df[col_name].fillna(value_fill)
    
    return df

# pipe line 적용 함수 - input과 output은 train데이터와 test 데이터
def pipe_processing(train_df, test_df):
    num_features = []
    cat_features = []
    
    for col in train_df.columns:
        if train_df[col].dtypes == object:
            cat_features.append(col)
        else:
            num_features.append(col)
    
    # Pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(categories='auto', handle_unknown='ignore') 

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)])

    preprocessor_pipe = Pipeline(steps=[('preprocessor', preprocessor)]) # preprocessing-only
    preprocessor_pipe.fit(train_df)

    train_df_transformed = preprocessor_pipe.transform(train_df)
    test_df_transformed = preprocessor_pipe.transform(test_df)
    
    return train_df_transformed, test_df_transformed

# 결과보기 함수
def model_report(model, x_train, x_test, y_train, y_test):
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=model.predict_proba(x_test)[:,1]) # real y & predicted y (based on "Sepal width")
    roc_auc = auc(fpr, tpr) # AUC 면적의 값 (수치)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("ROC curve")
    plt.show()

    predictions = model.predict(x_test)

    print("Accuracy on Training set: {:.3f}".format(model.score(x_train, y_train)))
    print("Accuracy on Test set: {:.3f}".format(model.score(x_test, y_test)))
    
    return
# --------------------------------------------------------------------
    
# ML을 통한 결측치 채우기 - 모델은 GradientBoostingRegressor    
def fillna_with_ML(df, col_name):
      
    not_na_rows = df[df[col_name].notnull()]
    train_x = not_na_rows.drop([col_name],axis=1)
    train_y = not_na_rows[col_name]
    
    na_rows = df[df[col_name].isnull()]
    test_x = na_rows.drop([col_name],axis=1)
     
    train_x_transformed, test_x_transformed = pipe_processing(train_x ,test_x)
    
    model = GradientBoostingRegressor(n_estimators=200, random_state=0)
    model.fit(train_x_transformed, train_y)
    test_y = model.predict(test_x_transformed)
    
    na_rows[col_name] = test_y
    
    df = pd.concat([not_na_rows,na_rows],axis=0).sort_index()
    
    return df    

data_df = pd.read_csv('titanic.csv')

missingno.bar(data_df, sort='ascending', figsize = (30,5))

# Target 데이터, Feature 데이터 구분
data_df = pd.read_csv('titanic.csv')
y_data = data_df[['Survived']]

del data_df['Survived']
x_data = data_df.copy()

# 가족 사이즈 열 추가
x_data['FamilyNum'] =  x_data['SibSp']+x_data['Parch']

# 호칭 열 추가 - Age 값을 예측함에 있어서 필요함. 
name_title = pd.DataFrame(i.split(',')[1].split(' ')[1] for i in x_data['Name'])
name_title


for title in name_title.iterrows():
    if title[1][0] not in ['Mr.','Mrs.','Miss.','Master.','Dr.']:
        name_title.iloc[title[0]]='Others'

x_data['NameTitle'] =  name_title

# PassengerId, Cabin, Ticket, Name 열 날리기
x_data= x_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

# Pclass 열 object type으로 바꾸기
x_data = x_data.astype({'Pclass':object})

# Fare 열 log값 취하기
x_data['Fare'] = np.log1p(x_data['Fare'])

# Embarked 열에 대해서는 최빈 값으로 채우기
x_data = fillna_with_mode(x_data,'Embarked')

# Age 열에 대해서는 결측치 없는 열로 학습한 모델로 예측한 결과를 기입
x_data = fillna_with_ML(x_data,'Age')

""" 분석 """
""" 여러 모델에 대해서 학습 및 정확도 확인 """
#분석 설정값
test_size = 0.3
random_state = 0
num_folds=10
str_kf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = random_state)


# KNN - 단일 시행
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

model = neighbors.KNeighborsClassifier(5) # K = 5
model.fit(x_train_transformed, y_train)

model_report(model, x_train_transformed, x_test_transformed, y_train, y_test)


# SVM
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

model = svm.SVC() 
model.fit(x_train_transformed, y_train)

y_pred = model.predict(x_test_transformed)  # 예측 라벨
print(accuracy_score(y_pred, y_test)) # 정확도 측정 및 기록


# RandomForestClassifier - 단일 시행
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)
model = RandomForestClassifier(random_state = random_state, n_jobs = -1, n_estimators = 100, max_depth = 3)
model.fit(x_train_transformed, y_train)

model_report(model, x_train_transformed, x_test_transformed, y_train, y_test)


# GradientBoostingClassifier - 단일 시행
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)
model = GradientBoostingClassifier(n_estimators=300, random_state=random_state)
model.fit(x_train_transformed, y_train)

model_report(model, x_train_transformed, x_test_transformed, y_train, y_test)


# XGBClassifier - 단일 시행
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)
model = XGBClassifier(n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 2, random_state=random_state)
model.fit(x_train_transformed, y_train)

model_report(model, x_train_transformed, x_test_transformed, y_train, y_test)


# GradientBoostingClassifier- K-fold
accuracy_history = []

for train_index, test_index in str_kf.split(x_data, y_data):
    X_train, X_test = x_data.loc[train_index], x_data.loc[test_index]
    y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]

    x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

    model = GradientBoostingClassifier(n_estimators=300, random_state=random_state)
    model.fit(x_train_transformed, y_train) # <- x_train_transformed (not x_train)
    
    y_pred = model.predict(x_test_transformed) # 예측 라벨
    accuracy_history.append(accuracy_score(y_pred, y_test)) # 정확도 측정 및 기록

print("각 분할의 정확도 기록 :", accuracy_history)
print("평균 정확도 :", np.mean(accuracy_history))   


# XGBClassifier- K-fold
accuracy_history = []

for train_index, test_index in str_kf.split(x_data, y_data):
    X_train, X_test = x_data.loc[train_index], x_data.loc[test_index]
    y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]

    x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

    model = XGBClassifier(n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 2, random_state=random_state, eval_metric='mlogloss')
    model.fit(x_train_transformed, y_train) # <- x_train_transformed (not x_train)
    
    y_pred = model.predict(x_test_transformed) # 예측 라벨
    accuracy_history.append(accuracy_score(y_pred, y_test)) # 정확도 측정 및 기록

print("각 분할의 정확도 기록 :", accuracy_history)
print("평균 정확도 :", np.mean(accuracy_history))   


# Train 데이터에 대해서 하이퍼 파라미터를 조절해가면서 K-fold 교차 검증을 진행 -> 가장 정확도가 높은 모델에 대해서 Test 
n_estimators = np.linspace(100,500,5,dtype = int)
learning_rate = np.logspace(-4, -1, 4)
max_depth = np.linspace(2,5,4, dtype = int)

best_accuracy = 0
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)

X_train = X_train.reset_index().drop(['index'],axis=1)
y_train = y_train.reset_index().drop(['index'],axis=1)

for n in n_estimators:
    for l in learning_rate:
        for m in max_depth:
            accuracy_history = []
            for train_index, val_index in str_kf.split(X_train, y_train):
                x_train_CV, x_val_CV = X_train.loc[train_index], X_train.loc[val_index]
                y_train_CV, y_val_CV = y_train.loc[train_index], y_train.loc[val_index]

                x_train_transformed , x_val_transformed = pipe_processing(x_train_CV, x_val_CV)

                model = XGBClassifier(random_state=random_state,eval_metric='mlogloss')
                model.fit(x_train_transformed, y_train_CV) # <- x_train_transformed (not x_train)

                y_pred = model.predict(x_val_transformed) # 예측 라벨
                accuracy_history.append(accuracy_score(y_pred, y_val_CV)) # 정확도 측정 및 기록

            if best_accuracy < np.mean(accuracy_history):
                best_n = n
                best_l = l
                best_m = m
                best_accuracy = np.mean(accuracy_history)

print("n_estimators: ",best_n,"learning_rate: ",best_l, "max_depth: ", best_m )
print("Best Mean Accuracy :", best_accuracy)   


# XGBClassifier - 단일 시행
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)
model = XGBClassifier(n_jobs = -1, learning_rate = 0.0001, n_estimators = 100, max_depth = 2, random_state=random_state)
model.fit(x_train_transformed, y_train)

model_report(model, x_train_transformed, x_test_transformed, y_train, y_test)

n_estimators = np.linspace(100,500,5,dtype = int)
learning_rate = np.logspace(-4, -1, 4)
max_depth = np.linspace(2,5,4, dtype = int)

best_accuracy = 0

for n in n_estimators:
    for l in learning_rate:
        for m in max_depth:
            accuracy_history = []
            for train_index, test_index in str_kf.split(x_data, y_data):
                X_train, X_test = x_data.loc[train_index], x_data.loc[test_index]
                y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]

                x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

                model =  XGBClassifier(n_jobs = -1, learning_rate = l, n_estimators = n, max_depth = m, random_state=random_state, eval_metric='mlogloss')
                model.fit(x_train_transformed, y_train) # <- x_train_transformed (not x_train)

                y_pred = model.predict(x_test_transformed) # 예측 라벨
                accuracy_history.append(accuracy_score(y_pred, y_test)) # 정확도 측정 및 기록

            if best_accuracy < float(np.mean(accuracy_history)):
                best_n = n
                best_l = l
                best_m = m
                best_accuracy = float(np.mean(accuracy_history))

print("n_estimators: ",best_n,"learning_rate: ",best_l, "max_depth: ", best_m )
print("Best Mean Accuracy :", best_accuracy)   


# Stacking  - GradientBoostingClassifier, RandomForestClassifier, XGBClassifier
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

models = [ 
GradientBoostingClassifier(n_estimators=500, random_state=0), 
RandomForestClassifier(random_state = random_state, n_jobs = -1, n_estimators = 500, max_depth = 3), 
XGBClassifier(random_state = random_state, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 2, eval_metric='mlogloss')] 

S_train, S_test = stacking(models, 
                       x_train_transformed, y_train, x_test_transformed, 
                       regression = False, 
                       metric = accuracy_score, 
                       n_folds = 10, stratified = True, shuffle = True, 
                       random_state = 0, verbose = 0)


model = XGBClassifier(random_state = random_state, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 2, eval_metric='mlogloss')

model = model.fit(S_train, y_train)  # <- x_train_transformed (not x_train)


model_report(model,S_train, S_test, y_train, y_test)


# Stacking  - GradientBoostingClassifier, KNeighborsClassifier, XGBClassifier, SVM, RandomForest
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
x_train_transformed , x_test_transformed = pipe_processing(X_train, X_test)

models = [ 
GradientBoostingClassifier(n_estimators=500, random_state=0), 
neighbors.KNeighborsClassifier(5),
XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 500, max_depth = 3, eval_metric='mlogloss')] 

S_train, S_test = stacking(models, 
                       x_train_transformed, y_train, x_test_transformed, 
                       regression = False, 
                       metric = accuracy_score, 
                       n_folds = 10, stratified = True, shuffle = True, 
                       random_state = 0, verbose = 0)


model = XGBClassifier(random_state = random_state, n_jobs = -1, learning_rate = 0.1, n_estimators = 100, max_depth = 3, eval_metric='mlogloss')

model = model.fit(S_train, y_train)  # <- x_train_transformed (not x_train)

model_report(model,S_train, S_test, y_train, y_test)