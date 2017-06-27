import numpy as np
from keras import losses
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=losses.binary_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(training_data, target_data, nb_epoch=200, verbose=2)

test_data = np.array([[1,1], [1,0]])
print(model.predict(test_data))
