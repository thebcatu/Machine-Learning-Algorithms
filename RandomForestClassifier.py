from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MachineLearningModel:
    def __init__(self):
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

ml_model = MachineLearningModel()
ml_model.train()
accuracy = ml_model.evaluate()
print(f"Model Accuracy: {accuracy * 100:.2f}%")
