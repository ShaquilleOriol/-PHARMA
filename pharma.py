import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



trainData = pd.read_csv("train.csv")
trainData.head()


testData = pd.read_csv("test.csv")
testData.head()



labelencoder = LabelEncoder()
trainData["species"] = labelencoder.fit_transform(trainData["species"])
trainData.head(6)



X = trainData.iloc[:, 1:]
Y = trainData["species"]



train_x, test_x, train_y, test_y = train_test_split(
    X, Y, random_state=5, test_size=20)



randForestClass = RandomForestClassifier(n_estimators=100)
randForestClass .fit(train_x, train_y)

RFprediction = randForestClass.predict(test_x)
print("Radom Forest Accuracy Score : ",
      metrics.accuracy_score(RFprediction, test_y))



DTclass = DecisionTreeClassifier()
DTclass.fit(train_x, train_y)

DTprediction = DTclass.predict(test_x)
print("Decision Tree Accuracy Score : ",
      metrics.accuracy_score(DTprediction, test_y))



gaussianClass = GaussianNB()
gaussianClass.fit(train_x, train_y)

GNpredict = gaussianClass.predict(test_x)
print("Naive Bayes Accuracy Score : ",
      metrics.accuracy_score(GNpredict, test_y))


svmClass = LinearSVC()
svmClass.fit(train_x, train_y)

SVMprediction = svmClass.predict(test_x)
print("SVM Accuracy Score : ", metrics.accuracy_score(SVMprediction, test_y))



testPrediction = gaussianClass.predict(testData)


testData["species"] = labelencoder.inverse_transform(testPrediction)
testData.head()


testData.to_csv("Output.csv", index=True)

