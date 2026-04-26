from .model import build_model
from sklearn.utils import class_weight
import numpy as np

def train_model(X_train, X_test, y_train, y_test):

    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Handle imbalance
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    model.save("ids_model.h5")

    return model, history