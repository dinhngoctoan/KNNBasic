from KNN import KNNModel
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv',encoding='utf-8-sig')

y = cancer['diagnosis']
X = cancer.drop(['diagnosis','Unnamed: 32','id'], axis =1)

X = X.to_numpy()
y = y.to_list()

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=42)

model = KNNModel(23)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
accuracy = accuracy_score(y_predict,y_test)
print(accuracy)