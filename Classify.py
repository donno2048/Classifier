from tensorflow import convert_to_tensor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
def Classifier(images, labels, number_of_neurons = 100, method = "relu", optimizer = "Adam", runs = 1):
    model = Sequential()
    model.add(Flatten(input_shape = images.shape[1:]))
    model.add(Dense(number_of_neurons, activation = method))
    model.add(Dense(len(set(labels)), activation = "softmax"))
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    model.fit(images, labels, epochs = runs)
    return lambda image: model.predict(convert_to_tensor([image]))[0].argmax()