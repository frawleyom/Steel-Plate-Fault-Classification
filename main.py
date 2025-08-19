
import pandas as pd
from scripts.preprocessing import load_and_split_data
from scripts.feature_engineering import apply_pca, apply_lda
from scripts.kNN import train_knn
from scripts.naive_bayes import train_naive_bayes
from scripts.MLP import train_mlp
from scripts.evaluation import evaluate_model
from scripts.clustering import run_kmeans_clustering
from scripts.visualisation import (
    load_dataset, plot_class_distribution, plot_pca_scree,
    plot_correlation_heatmap, plot_pca_2d, plot_classification_report,
    plot_accuracy_bar_chart, plot_roc_curve
)

# Load dataset for initial visualizations
X, y = load_dataset()

# Generate initial visualizations
plot_class_distribution(y)
plot_pca_scree(X)
plot_correlation_heatmap(X)
plot_pca_2d(X, y)

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_split_data()

# Store accuracy scores for bar chart and best model details
accuracy_scores = {'k-NN': [], 'Naive Bayes': [], 'MLP': []}
best_models = {'k-NN': None, 'Naive Bayes': None, 'MLP': None}
best_datasets = {'k-NN': None, 'Naive Bayes': None, 'MLP': None}
dataset_types = ['Raw Data', 'PCA', 'LDA']

# --------------------------
# Evaluation without feature engineering
# --------------------------
print("Evaluating classifiers on raw data...")

# Train and evaluate k-NN classifier
knn_model = train_knn(X_train, y_train)
knn_metrics = evaluate_model(knn_model, X_test, y_test)
accuracy_scores['k-NN'].append(knn_metrics['accuracy'])
best_models['k-NN'], best_datasets['k-NN'] = knn_model, 'Raw Data'

# Train and evaluate Naive Bayes classifier
nb_model = train_naive_bayes(X_train, y_train)
nb_metrics = evaluate_model(nb_model, X_test, y_test)
accuracy_scores['Naive Bayes'].append(nb_metrics['accuracy'])
best_models['Naive Bayes'], best_datasets['Naive Bayes'] = nb_model, 'Raw Data'

# Train and evaluate MLP classifier
mlp_model = train_mlp(X_train, y_train)
mlp_metrics = evaluate_model(mlp_model, X_test, y_test)
accuracy_scores['MLP'].append(mlp_metrics['accuracy'])
best_models['MLP'], best_datasets['MLP'] = mlp_model, 'Raw Data'

# --------------------------
# Evaluation with PCA
# --------------------------
print("\nEvaluating classifiers with PCA-transformed data...")

# Apply PCA
X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=5)

# Train and evaluate k-NN with PCA
knn_model_pca = train_knn(X_train_pca, y_train)
knn_metrics_pca = evaluate_model(knn_model_pca, X_test_pca, y_test)
accuracy_scores['k-NN'].append(knn_metrics_pca['accuracy'])
if knn_metrics_pca['accuracy'] > knn_metrics['accuracy']:
    best_models['k-NN'], best_datasets['k-NN'] = knn_model_pca, 'PCA'

# Train and evaluate Naive Bayes with PCA
nb_model_pca = train_naive_bayes(X_train_pca, y_train)
nb_metrics_pca = evaluate_model(nb_model_pca, X_test_pca, y_test)
accuracy_scores['Naive Bayes'].append(nb_metrics_pca['accuracy'])
if nb_metrics_pca['accuracy'] > nb_metrics['accuracy']:
    best_models['Naive Bayes'], best_datasets['Naive Bayes'] = nb_model_pca, 'PCA'

# Train and evaluate MLP with PCA
mlp_model_pca = train_mlp(X_train_pca, y_train)
mlp_metrics_pca = evaluate_model(mlp_model_pca, X_test_pca, y_test)
accuracy_scores['MLP'].append(mlp_metrics_pca['accuracy'])
if mlp_metrics_pca['accuracy'] > mlp_metrics['accuracy']:
    best_models['MLP'], best_datasets['MLP'] = mlp_model_pca, 'PCA'

# --------------------------
# Evaluation with LDA
# --------------------------
print("\nEvaluating classifiers with LDA-transformed data...")

# Apply LDA
X_train_lda, X_test_lda, lda = apply_lda(X_train, X_test, y_train, n_components=2)

# Train and evaluate k-NN with LDA
knn_model_lda = train_knn(X_train_lda, y_train)
knn_metrics_lda = evaluate_model(knn_model_lda, X_test_lda, y_test)
accuracy_scores['k-NN'].append(knn_metrics_lda['accuracy'])
if knn_metrics_lda['accuracy'] > max(knn_metrics['accuracy'], knn_metrics_pca['accuracy']):
    best_models['k-NN'], best_datasets['k-NN'] = knn_model_lda, 'LDA'

# Train and evaluate Naive Bayes with LDA
nb_model_lda = train_naive_bayes(X_train_lda, y_train)
nb_metrics_lda = evaluate_model(nb_model_lda, X_test_lda, y_test)
accuracy_scores['Naive Bayes'].append(nb_metrics_lda['accuracy'])
if nb_metrics_lda['accuracy'] > max(nb_metrics['accuracy'], nb_metrics_pca['accuracy']):
    best_models['Naive Bayes'], best_datasets['Naive Bayes'] = nb_model_lda, 'LDA'

# Train and evaluate MLP with LDA
mlp_model_lda = train_mlp(X_train_lda, y_train)
mlp_metrics_lda = evaluate_model(mlp_model_lda, X_test_lda, y_test)
accuracy_scores['MLP'].append(mlp_metrics_lda['accuracy'])
if mlp_metrics_lda['accuracy'] > max(mlp_metrics['accuracy'], mlp_metrics_pca['accuracy']):
    best_models['MLP'], best_datasets['MLP'] = mlp_model_lda, 'LDA'

# --------------------------
# Save results to CSV
# --------------------------
results = pd.DataFrame({
    'Raw Data': [knn_metrics, nb_metrics, mlp_metrics],
    'PCA': [knn_metrics_pca, nb_metrics_pca, mlp_metrics_pca],
    'LDA': [knn_metrics_lda, nb_metrics_lda, mlp_metrics_lda]
}, index=['k-NN', 'Naive Bayes', 'MLP'])

results.to_csv('results/classifier_metrics.csv', index=True)
print("\nEvaluation results saved to 'results/classifier_metrics.csv'")

# --------------------------
# Plot visualizations for best models
# --------------------------
print("\nGenerating visualizations for best models...")
for classifier_name, model in best_models.items():
    dataset_type = best_datasets[classifier_name]
    
    # Select the appropriate test set based on the best dataset type
    if dataset_type == 'Raw Data':
        X_best_test = X_test
        X_best_train = X_train
    elif dataset_type == 'PCA':
        X_best_test = X_test_pca
        X_best_train = X_train_pca
    elif dataset_type == 'LDA':
        X_best_test = X_test_lda
        X_best_train = X_train_lda
    
    # Generate classification report and ROC curve for the best model
    plot_classification_report(y_test, model.predict(X_best_test), y.columns, f'results/visualizations/{classifier_name}_{dataset_type}_classification_report.png')
    plot_roc_curve(X_best_train, y_train, X_best_test, y_test, model, y.columns, f'results/visualizations/{classifier_name}_{dataset_type}_roc_curve.png')


# Plot accuracy bar chart
plot_accuracy_bar_chart(accuracy_scores, dataset_types, accuracy_scores.keys(), 'results/visualizations/accuracy_comparison.png')

# --------------------------
# K-Means Clustering
# --------------------------
print("\nRunning K-Means clustering...")

kmeans_metrics = run_kmeans_clustering(X, y)

# --------------------------
# Save results to CSV
# --------------------------
results = pd.DataFrame({
    'Raw Data': [knn_metrics, nb_metrics, mlp_metrics],
    'PCA': [knn_metrics_pca, nb_metrics_pca, mlp_metrics_pca],
    'LDA': [knn_metrics_lda, nb_metrics_lda, mlp_metrics_lda],
    'ARI': [None, None, None, kmeans_metrics['ARI']],
    'NMI': [None, None, None, kmeans_metrics['NMI']],
    'V-Measure': [None, None, None, kmeans_metrics['V-Measure']]
}, index=['k-NN', 'Naive Bayes', 'MLP', 'K-Means'])

results.to_csv('results/classifier_metrics.csv', index=True)
print("\nEvaluation results saved to 'results/classifier_metrics.csv'")



