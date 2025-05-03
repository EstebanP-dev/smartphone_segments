import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("smartphone_segments.csv")

le = LabelEncoder()
df["segment_label"] = le.fit_transform(df["segment"])

features = ["price_usd", "ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_size_in", "weight_g"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df["segment_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    results.append((name, acc, f1))
    print(f"{name}: Accuracy={acc:.3f}, F1-macro={f1:.3f}")

best_name, best_acc, best_f1 = max(results, key=lambda x: x[2])
best_model = models[best_name]
print(f"Mejor modelo: {best_name} (F1={best_f1:.3f})")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "encoder": le
    }, f)
