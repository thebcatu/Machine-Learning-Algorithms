import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class NeuralNetworkModel:
    def __init__(self):
        self.data = load_digits()
        self.X = self.data.images.reshape(len(self.data.images), -1)  # Flatten images
        self.y = to_categorical(self.data.target)  # One-hot encoding
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(64,)),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')  
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=20, batch_size=32):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return accuracy

    def predict_sample(self, sample):
        sample_scaled = self.scaler.transform([sample])
        prediction = self.model.predict(sample_scaled)
        return np.argmax(prediction)

nn_model = NeuralNetworkModel()
nn_model.train(epochs=20)
accuracy = nn_model.evaluate()
print(f"Model Accuracy: {accuracy * 100:.2f}%")

sample_data = nn_model.X[0]
predicted_class = nn_model.predict_sample(sample_data)
print(f"Predicted digit for sample data: {predicted_class}")
