
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

#Murat Dogan

#dataset
dataset = pd.read_csv("dataset.csv")

#visualization
dataset.groupby(["label"])["text"].count().plot(kind="bar")
plt.savefig("labelscount.pdf")
plt.title("How many fake an real names we have")
plt.close()

#train test
labels = dataset.label
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], labels,
                                                    test_size=0.3, random_state=10)


#tfidf
tfidf = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

#count
count = CountVectorizer(stop_words='english')
count_train = count.fit_transform(X_train)
count_test = count.transform(X_test)

#Naive Bayes classifier for Multinomial model with tfidf
clf = MultinomialNB()

clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score, "NB-TFIDF")
plot_confusion_matrix(clf, tfidf_test,y_test,cmap=plt.cm.Blues)
plt.title("NB-TFIDF")
plt.savefig("NB-TFIDF.pdf")
plt.close()

#Naive Bayes classifier for Multinomial model with count
clf = MultinomialNB()

clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score, "NB-COUNT")
plot_confusion_matrix(clf, count_test,y_test,cmap=plt.cm.Blues)
plt.title("NB-COUNT")
plt.savefig("NB-COUNT.pdf")
plt.close()

#Applying Passive Aggressive Classifier with TFIDF
clf = PassiveAggressiveClassifier()

clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score, "PassiveAggressive-TFIDF")
plot_confusion_matrix(clf, tfidf_test,y_test,cmap=plt.cm.Blues)
plt.title("PassiveAggressive-TFIDFF")
plt.savefig("PassiveAggressive-TFIDF.pdf")
plt.close()

#fake news words
feature_names = tfidf.get_feature_names()
datas = sorted(zip(clf.coef_[0], feature_names))[:30]
print("----------------------------------------------")
print("Best accuracy score algorithm used for visualization of result")
for coef, words in datas:
    print(coef, words,"FAKE")
print("----------------------------------------------")

#Applying Passive Aggressive Classifier with count
clf = PassiveAggressiveClassifier()

clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score, "PassiveAggressive-COUNT")
plot_confusion_matrix(clf, count_test,y_test,cmap=plt.cm.Blues)
plt.title("PassiveAggressive-COUNT")
plt.savefig("PassiveAggressive-COUNT.pdf")
plt.close()
