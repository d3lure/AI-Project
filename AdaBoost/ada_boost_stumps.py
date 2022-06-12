import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data_path = '../data/'
base_name = 'dogs_cats'
width = 80
include = {'dog', 'cat'}

data = joblib.load(f'./{base_name}_{width}x{width}px.pkl')

labels = np.unique(data['label'])

X = np.array(data['data'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

n, nx, ny, rgb = X_train.shape
X_train = X_train.reshape((n, nx * ny * rgb))

n, nx, ny, rgb = X_test.shape
X_test = X_test.reshape((n, nx * ny * rgb))

n_estimators = 80

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)

ada = AdaBoostClassifier(
        base_estimator=dt_stump,
        n_estimators= n_estimators
    )

model = ada.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

values = {
    "X_train": X_train,
    "X_test": X_test,
    "y_pred": y_pred,
    "y_train": y_train,
    "y_test": y_test
}

pickle.dump(model, open('adb_model2.sav', 'wb'))
pickle.dump(ada, open('adb_clf_2.sav', 'wb'))

with open('values2.pickle', 'wb') as handle:
    pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
