from src.preprocess import load_data
from src.train import train_model
from src.evaluate import evaluate_model

X_train, X_test, y_train, y_test = load_data("data/kdd_test.csv")

model, history = train_model(X_train, X_test, y_train, y_test)

evaluate_model(model, X_test, y_test)
