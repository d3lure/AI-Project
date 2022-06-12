import matplotlib.pyplot as plt
import os
import pathlib
import pickle
from sklearn.metrics import RocCurveDisplay, zero_one_loss
import seaborn as sns
import numpy as np
# provide dataset directory
data_dir = pathlib.Path(os.path.join('../data/'))

model = pickle.load(open('adb_model.sav', 'rb'))

values = pickle.load(open('values.pickle','rb'))

colors = sns.color_palette("colorblind")

def ROC():
    svc_disp = RocCurveDisplay.from_estimator(model, values['X_test'], values['y_test'])
    svc_disp.plot()
    plt.savefig("adaboost_roc.png")
    plt.show()


def ERR(n_estimators, model, values):
    err_test = np.zeros((n_estimators,))
    for i, y_pred in enumerate(model.staged_predict(values['X_test'])):
        err_test[i] = zero_one_loss(y_pred, values['y_test'])

    err_train = np.zeros((n_estimators,))

    for i, y_pred in enumerate(model.staged_predict(values['X_train'])):
        err_train[i] = zero_one_loss(y_pred, values['y_train'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        np.arange(n_estimators) + 1,
        err_test,
        label="AdaBoost Test Error",
        color=colors[0],
    )
    ax.plot(
        np.arange(n_estimators) + 1,
        err_train,
        label="AdaBoost Train Error",
        color=colors[1],
    )

    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel("Number of weak learners")
    ax.set_ylabel("error rate")

    leg = ax.legend(loc="upper right", fancybox=True)
    leg.get_frame().set_alpha(0.7)

    plt.savefig(f"adaboost_err_{n_estimators}.png")
    #plt.show()


# ROC()
ERR(80, pickle.load(open('adb_model80.sav', 'rb')), pickle.load(open('values80.pickle','rb')))
ERR(100, pickle.load(open('adb_model.sav', 'rb')), pickle.load(open('values.pickle','rb')))
ERR(128, pickle.load(open('adb_model128.sav', 'rb')), pickle.load(open('values128.pickle','rb')))