from pathlib import Path
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")

data = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "model.pkl")

print("Model saved successfully")
