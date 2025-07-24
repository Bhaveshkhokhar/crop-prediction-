import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
df = pd.read_csv('Crop_recommendation.csv')
X=df.drop('label',axis=1)
y=df['label']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


rf = RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 5, n_estimators= 100)
rf.fit(X_train, y_train)
pickle.dump(rf,open('model.pkl','wb'))