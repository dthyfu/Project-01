"""Local, dataset-based diabetes risk estimate; this is not a diagnostic tool."""

import csv
import threading
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_prediction_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_cnn.keras"
TRANSFORM_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_cnn_transform.joblib"
FEATURES = (
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
)
CATEGORIES = {
    "gender": {"Female", "Male", "Other"},
    "smoking_history": {"never", "No Info", "current", "former", "not current", "ever"},
}
NUMERIC_RANGES = {
    "age": (1, 120), "bmi": (10, 100), "HbA1c_level": (2, 20),
    "blood_glucose_level": (20, 600),
}

_lock = threading.Lock()
_model = None
_metrics = None


class CNNRiskModel:
    def __init__(self, network, transform, np):
        self.network = network
        self.transform = transform
        self.np = np

    def predict_proba(self, frame):
        encoded = self.transform.transform(frame).astype("float32")
        positive = self.network.predict(
            encoded[..., self.np.newaxis], verbose=0,
        ).reshape(-1)
        return self.np.column_stack((1.0 - positive, positive))


class RiskInputError(ValueError):
    pass


def validate_input(values):
    if not isinstance(values, dict):
        raise RiskInputError("Hãy nhập đầy đủ các chỉ số trong biểu mẫu.")
    result = {}
    for field in FEATURES:
        value = values.get(field)
        if field in CATEGORIES:
            if value not in CATEGORIES[field]:
                raise RiskInputError("Một lựa chọn trong biểu mẫu không hợp lệ.")
            result[field] = value
        elif field in {"hypertension", "heart_disease"}:
            if str(value) not in {"0", "1"}:
                raise RiskInputError("Tăng huyết áp và bệnh tim phải chọn Có hoặc Không.")
            result[field] = int(value)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise RiskInputError("Hãy nhập đầy đủ các chỉ số dạng số.") from None
            low, high = NUMERIC_RANGES[field]
            if not low <= number <= high:
                raise RiskInputError(f"Giá trị {field} nằm ngoài phạm vi biểu mẫu.")
            result[field] = number
    return result


def predict_risk(values):
    """Classify the supplied profile into a dataset-based risk band."""
    clean = validate_input(values)
    model, _metrics = _get_model()
    import pandas as pd

    probability = float(model.predict_proba(pd.DataFrame([clean], columns=FEATURES))[0][1])
    flags = []
    if clean["HbA1c_level"] >= 6.5:
        flags.append(
            "HbA1c bạn nhập nằm từ ngưỡng xét nghiệm thường dùng để phát hiện đái tháo đường; "
            "cần nhân viên y tế đánh giá và xác nhận bằng xét nghiệm phù hợp."
        )
    if values.get("fasting_status") == "yes" and clean["blood_glucose_level"] >= 126:
        flags.append(
            "Đường huyết lúc đói bạn nhập nằm từ ngưỡng xét nghiệm thường dùng để phát hiện đái tháo đường; "
            "cần nhân viên y tế đánh giá và xác nhận bằng xét nghiệm phù hợp."
        )
    high_risk = bool(flags) or probability >= 0.5
    return {
        "risk_level": "high" if high_risk else "not_high",
        "risk_label": "Nguy cơ cao" if high_risk else "Không thuộc nhóm nguy cơ cao",
        "status": "experimental_estimate",
        "clinical_note": " ".join(flags),
        "message": (
            "Kết quả này không phải chẩn đoán. “Không thuộc nhóm nguy cơ cao” "
            "không có nghĩa là hoàn toàn không có nguy cơ. Hãy trao đổi với nhân viên y tế nếu "
            "lo lắng hoặc có kết quả xét nghiệm bất thường."
        ),
    }


def _get_model():
    global _model, _metrics
    if _model is not None:
        return _model, _metrics
    with _lock:
        if _model is not None:
            return _model, _metrics

        if MODEL_PATH.is_file() and TRANSFORM_PATH.is_file():
            import joblib
            import numpy as np
            import tensorflow as tf

            network = tf.keras.models.load_model(MODEL_PATH, compile=False)
            transform = joblib.load(TRANSFORM_PATH)
            _model = CNNRiskModel(network, transform, np)
            return _model, _metrics

        if not DATA_PATH.is_file():
            raise FileNotFoundError(f"Không tìm thấy dataset: {DATA_PATH.name}")

        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        import numpy as np
        import tensorflow as tf

        frame = pd.read_csv(DATA_PATH)
        required = set(FEATURES) | {"diabetes"}
        if not required.issubset(frame.columns):
            raise ValueError("Dataset thiếu các cột cần thiết để huấn luyện mô hình.")
        frame = frame[list(FEATURES) + ["diabetes"]].dropna()
        x = frame[list(FEATURES)]
        y = frame["diabetes"].astype(int)
        categorical = ["gender", "smoking_history"]
        numeric = [name for name in FEATURES if name not in categorical]
        transform = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y,
        )
        x_train = transform.fit_transform(x_train).astype("float32")
        x_test = transform.transform(x_test).astype("float32")
        tf.keras.utils.set_random_seed(42)
        network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x_train.shape[1], 1)),
            tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"),
            tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        network.fit(
            x_train[..., np.newaxis], y_train.to_numpy(dtype="float32"),
            epochs=4, batch_size=512, validation_split=0.1,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True,
            )],
            verbose=0,
        )
        scores = network.predict(x_test[..., np.newaxis], verbose=0).reshape(-1)

        candidate = CNNRiskModel(network, transform, np)
        _metrics = {
            "test_rows": int(len(y_test)),
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 3),
            "brier_score": round(float(brier_score_loss(y_test, scores)), 4),
        }
        import joblib
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        network.save(MODEL_PATH)
        joblib.dump(transform, TRANSFORM_PATH)
        _model = candidate
        return _model, _metrics
