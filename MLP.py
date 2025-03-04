import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


file_path = 'C:\\Users\\rezaa\\OneDrive\\Desktop\\Personal\\Uni\\4022\\Machine Leaning Basics\\Project\\02\\leaves.csv'
data = pd.read_csv(file_path)


labels = data.iloc[:, 0]
features = data.iloc[:, 2:]


imputer = SimpleImputer(strategy='most_frequent')
features_imputed = imputer.fit_transform(features)


mlp_config = {
    'hidden_layer_sizes': (150,),
    'activation': 'tanh',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 50,
    'learning_rate': 'invscaling',
    'learning_rate_init': 0.01,
    'max_iter': 400,
    'random_state': 42,
    'tol': 1e-4
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=12)),
    ('mlp', MLPClassifier(**mlp_config))
])


X_train, X_test, y_train, y_test = train_test_split(features_imputed, labels, test_size=0.3, random_state=42)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


cv_scores = cross_val_score(pipeline, features_imputed, labels, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

