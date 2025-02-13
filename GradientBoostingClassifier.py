from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

class GradientBoostingModel:
    def __init__(self):
        self.data = load_breast_cancer()
        self.X = self.data.data
        self.y = self.data.target
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        return accuracy, report

    def predict_sample(self, sample):
        return self.model.predict([sample])

gb_model = GradientBoostingModel()
gb_model.train()
accuracy, report = gb_model.evaluate()
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

sample_data = gb_model.X_test[0]
predicted_class = gb_model.predict_sample(sample_data)
print(f"Predicted class for sample data: {predicted_class[0]}")
