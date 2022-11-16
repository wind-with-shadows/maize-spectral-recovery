from functools import reduce
from operator import truediv
from tkinter import Y
import numpy as np
from numpy.core.fromnumeric import reshape
from pandas.core.accessor import register_dataframe_accessor
import scipy.io as sio
from scipy.sparse.sputils import matrix
from sklearn.decomposition import PCA
import spectral
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from  time import time
from sklearn.svm import SVC
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

y=sio.loadmat(r'C:\Users\DAI\Desktop\病害识别\数据\alllabel\211TRUTH.mat')['I']
y=np.array(y)
print(y.shape)
a=spectral.imshow(classes = y)
plt.show()
plt.pause(5000)