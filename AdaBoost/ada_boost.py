
import numpy as np
import os
from random import randint

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread
from skimage.transform import resize, AffineTransform
from skimage.transform import rotate
from skimage.transform import warp
from collections import Counter



def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}x{height}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))  # [:,:,::-1]
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))  # [:,:,::-1]
                    im = rotate(im, randint(1, 20))
                    tf = AffineTransform(shear=-0.5)
                    im = warp(im, tf, order=1, preserve_range=True, mode='wrap')
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, pklname)


data_path = '../data/'
base_name = 'dogs_cats'
width = 80
include = {'dog', 'cat'}

resize_all(src=data_path, pklname=base_name, width=width, include=include)

data = joblib.load(f'./{base_name}_{width}x{width}px.pkl')
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
# print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

print(Counter(data['label']))

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])

X = np.array(data['data'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

n, nx, ny, rgb = X_train.shape
X_train = X_train.reshape((n, nx * ny * rgb))

n, nx, ny, rgb = X_test.shape
X_test = X_test.reshape((n, nx * ny * rgb))

print(X_train.shape)
print(X_test.shape)

print(np.array(y_train))
exit()
#
# # Load libraries
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import datasets
# # Import train_test_split function
# from sklearn.model_selection import train_test_split
# # Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
#
# from sklearn.svm import SVC
#
# # import scikit-learn metrics module for accuracy calculation
# from sklearn.metrics import accuracy_score
#
# # svc = SVC(probability=True, kernel='linear')
# n_estimators = 100
#
# clf = AdaBoostClassifier(n_estimators= n_estimators)
#
# model = clf.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# Instantiate the bagging classifier
#

# Create adaboost classifier object
# abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Train Adaboost Classifier
# model = abc.fit(X_train, y_train)

# Predict the response for test dataset
# y_pred = model.predict(X_test)
#
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print(abc.score(X_test, y_test))
# pickle.dump(model, open('adb_model.sav', 'wb'))
# pickle.dump(clf, open('adb_clf.sav', 'wb'))

