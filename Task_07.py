import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA


print("Loading and preparing the dataset...")

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining SVM models...")

svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
linear_score = svm_linear.score(X_test_scaled, y_test)
print(f"Linear SVM accuracy: {linear_score:.4f}")


svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
rbf_score = svm_rbf.score(X_test_scaled, y_test)
print(f"RBF SVM accuracy: {rbf_score:.4f}")


print("\nVisualizing decision boundaries...")

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

svm_linear_2d = SVC(kernel='linear', random_state=42)
svm_linear_2d.fit(X_train_2d, y_train)

svm_rbf_2d = SVC(kernel='rbf', random_state=42)
svm_rbf_2d.fit(X_train_2d, y_train)


def plot_decision_boundary(model, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', alpha=0.9)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

plot_decision_boundary(svm_linear_2d, 'SVM with Linear Kernel Decision Boundary')
plot_decision_boundary(svm_rbf_2d, 'SVM with RBF Kernel Decision Boundary')


print("\nTuning hyperparameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


best_svm = grid_search.best_estimator_
best_score = best_svm.score(X_test_scaled, y_test)
print(f"Best SVM test accuracy: {best_score:.4f}")


print("\nEvaluating model performance...")
y_pred = best_svm.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nModel Comparison:")
print(f"Linear SVM accuracy: {linear_score:.4f}")
print(f"RBF SVM accuracy: {rbf_score:.4f}")
print(f"Tuned SVM accuracy: {best_score:.4f}")
