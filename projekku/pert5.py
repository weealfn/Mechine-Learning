import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("processed_kelulusan.csv")
print("Dataset:", df.shape)
print(df.head())

# Pastikan kolom target ada
assert "Lulus" in df.columns, "Kolom target 'Lulus' tidak ditemukan!"

# =========================
# 2. Pisahkan fitur dan target
# =========================
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# =========================
# 3. Split data train / val / test
# =========================
# Pakai stratify kalau jumlah data per kelas cukup
if y.value_counts().min() >= 2:
    stratify_opt = y
else:
    print("⚠️ Warning: Data tidak seimbang, stratify dimatikan sementara")
    stratify_opt = None

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=stratify_opt, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp if stratify_opt is not None else None, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# =========================
# 4. Pipeline: Preprocessing + Model
# =========================
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

pipe_lr = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

# =========================
# 5. Train model
# =========================
pipe_lr.fit(X_train, y_train)

# =========================
# 6. Evaluasi di Validation
# =========================
y_val_pred = pipe_lr.predict(X_val)
print("\nEvaluasi di Validation Set:")
print("F1-macro:", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# =========================
# 7. Evaluasi di Test Set
# =========================
y_test_pred = pipe_lr.predict(X_test)
print("\nEvaluasi di Test Set:")
print("F1-macro:", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
