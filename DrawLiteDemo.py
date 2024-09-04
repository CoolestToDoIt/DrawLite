import pickle, h5py, numpy as np, matplotlib.pyplot as plt

hf_x_test = h5py.File('x_test.h5', 'r')
x_test = np.array(hf_x_test.get('name-of-dataset'))
hf_y_test = h5py.File('y_test.h5', 'r')
y_test = np.array(hf_y_test.get('name-of-dataset'))

print(len(y_test))
raw = int(input())

loaded_model = pickle.load(open('knnpickle_file', 'rb'))
face1 = x_test[raw].reshape(28,28)
plt.imshow(face1)
plt.show()

result = loaded_model.predict([x_test[raw]])
print(y_test[raw])
print(result)

"""
2 - hat
4 - flower
10 - bed
18 - cloud
21 - axe
25 - sun
26 - flower
27 - airplane
30 - pizza
"""