import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    # Clean label
    df["labels"] = df["labels"].astype(str).str.strip()
    df["labels"] = (df["labels"] != "normal").astype(int)

    # Drop missing
    df = df.dropna()

    # Encode categorical
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Split
    X = df.drop("labels", axis=1)
    y = df["labels"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Train-test split HERE (clean design)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)