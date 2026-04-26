from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test):
 

    os.makedirs("results", exist_ok=True)

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.clf()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("results/roc_curve.png")
    plt.clf()

    auc = roc_auc_score(y_test, y_prob)
    print("AUC:", auc)
