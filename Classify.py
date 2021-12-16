from tensorflow import convert_to_tensor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from jax.numpy import unique, argmax
class Classifier(Sequential):
    def __init__(self, images, labels, number_of_neurons = 100, method = "relu", optimizer = "Adam", runs = 1):
        super().__init__([
            Flatten(input_shape = images.shape[1:]),
            Dense(number_of_neurons, activation = method),
            Dense(unique(labels).size, activation = "softmax")
        ])
        self.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
        self.fit(images, labels, epochs = runs)
    def predict_label(self, image): return argmax(self.predict(convert_to_tensor([image]))[0])