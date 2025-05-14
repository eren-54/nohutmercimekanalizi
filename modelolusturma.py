import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Veriyi oku
df = pd.read_csv(r"C:\Users\erenh\OneDrive\Desktop\soncsv\veri_nisan4390.csv")
X = df.drop('label', axis=1)
y = df['label']

# Normalizasyon (kaydedilecek)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
joblib.dump(minmax_scaler, "minmax_scaler.joblib")
pd.DataFrame(X_normalized, columns=X.columns).assign(label=y).to_csv("veriler_normalize.csv", index=False)

# Standartlaştırma (isteğe bağlı)
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
pd.DataFrame(X_std, columns=X.columns).assign(label=y).to_csv("veriler_standardize.csv", index=False)

# Eğitim ve test verisi bölme
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# GridSearchCV ile en iyi k
params = {'n_neighbors': range(1, 21)}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)

print("En iyi k:", grid.best_params_)
print("En iyi doğruluk:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Test doğruluk:", accuracy_score(y_test, y_pred))
print("Rapor:\n", classification_report(y_test, y_pred))

# Modeli kaydet
joblib.dump(best_model, "knn_modelim.joblib")

