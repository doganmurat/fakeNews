import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#dataset
dataset = pd.read_csv("dataset.csv")
labels = dataset.label
#splitting train and test part
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], labels,
                                                    test_size=0.3, random_state=10)
#vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)
