import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import time, sys, numpy as np
import h5py, pickle

k = 40
kNearest = KNeighborsClassifier(k)

hf_x_train = h5py.File('x_train.h5', 'r')
x_train = np.array(hf_x_train.get('name-of-dataset'))
hf_x_test = h5py.File('x_test.h5', 'r')
x_test = np.array(hf_x_test.get('name-of-dataset'))
hf_y_train = h5py.File('y_train.h5', 'r')
y_train = np.array(hf_y_train.get('name-of-dataset'))
hf_y_test = h5py.File('y_test.h5', 'r')
y_test = np.array(hf_y_test.get('name-of-dataset'))

st_train = time.perf_counter()
kNearest.fit(x_train, y_train)
ed_train = time.perf_counter()
right = 0
st_test = time.perf_counter()
for n in range(len(x_test)):
    if kNearest.predict([x_test[n]]) == y_test[n]:
        right+=1
ed_test = time.perf_counter()
print(right/len(x_test))
print("Train time:", ed_train-st_train, "\nTest time:", ed_test-st_test)
print("Train set size:", len(x_train), "\nTest set size:", len(x_test))

# Its important to use binary mode 
knnPickle = open('knnpickle_file', 'wb') 
      
# source, destination 
pickle.dump(kNearest, knnPickle)  

# close the file
knnPickle.close()