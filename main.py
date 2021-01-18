import pandas as pd
from sklearn.model_selection import train_test_split

#reads dataset
df = pd.read_csv("dataset.csv")
labels = df.label

#split data test and train part
X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
