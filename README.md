# kaggle_titanic
titanic 생존자 예측

## 1. 시각화
- 남녀 나이대별 생존자, 사망자 비교
<p align="center"><img src="./images/survived_malefemale.png" width="80%" height="80%"></p>
나이에 따라 생존률의 차이가 있음을 확인할 수 있다. 따라서 결측치를 채워 사용하기로 했다.   
<br><br>

- Pclass별 생존자수 
<p align="center"><img src="./images/Pclass.png" width="50%" height="50%"></p>

- Embarked 별 생존자수
<p align="center"><img src="./images/Embarked.png" width="50%" height="50%"></p>

- Sex별 생존자수
<p align="center"><img src="./images/Sex_survived.png" width="50%" height="50%"></p>

- Family 별 생존자수
<p align="center"><img src="./images/Family.png" width="50%" height="50%"></p>

- Name 호칭 별 생존자수
<p align="center"><img src="./images/Name.png" width="50%" height="50%"></p>

- Age 별 생존자수
<p align="center"><img src="./images/Age.png" width="50%" height="50%"></p>

- Age 나이대별 생존자수
<p align="center"><img src="./images/Age_category.png" width="50%" height="50%"></p>

- Fare 별 생존자수
<p align="center"><img src="./images/Fare.png" width="50%" height="50%"></p>

- Cabin별 생존자수
<p align="center"><img src="./images/Cabin.png" width="50%" height="50%"></p>
<br><br>

## 2. 데이터 분석
1. SibSp 숫자에 따라 생존률이 차등이 있었다
2. Parch 숫자에 따라 생존률이 차등이 있었다
3. Pclass는 낮은 클라스 일수록 사망률이 높았다. (3rd > 2nd > 1st)
4.성별에 따라 생존률이 차등이 있었다.
5. 탑승장소에 따라 생존률 차등이 있었다.
6. 운임은 폭이 너무 커 log로 변환한 뒤 확인이 필요하다
<br><br>

## 3. 여러 ML모델에 대해 학습 및 정확도 확인
- KNN(KNeighborsClassifier)
<p align="center"><img src="./images/knn.png" width="50%" height="50%"></p>
Accuracy on Training set: 0.864
<br>  
Accuracy on Test set: 0.821  
<br><br>

- Random Forest Tree
<p align="center"><img src="./images/randomforest.png" width="50%" height="50%"></p>
Accuracy on Training set: 0.828
<br>
Accuracy on Test set: 0.806
<br><br>

- Gradient Boosting
<p align="center"><img src="./images/gradientboosting.png" width="50%" height="50%"></p>
Accuracy on Training set: 0.957
<br>  
Accuracy on Test set: 0.828
<br><br>

- XGBoost (Extreme Gradient Boosting)
<p align="center"><img src="./images/xgb.png" width="50%" height="50%"></p>
Accuracy on Training set: 0.894  
<br>
Accuracy on Test set: 0.828
<br><br>

- Stacking(GB, RandomForest, XGB)
<p align="center"><img src="./images/stacking(gb,rf,xgb).png" width="50%" height="50%"></p>
Accuracy on Training set: 0.830
<br>
Accuracy on Test set: 0.832
<br><br>

- Stacking(GB, KNN, XGB)
<p align="center"><img src="./images/stacking(gb,knn,xgb).png" width="50%" height="50%"></p>
Accuracy on Training set: 0.826
<br>   
Accuracy on Test set: 0.860