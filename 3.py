import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


def load_data(file_path):
    """Load Excel data"""
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        print(f"Data shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def perform_pca_analysis(df):
    """Perform PCA analysis"""
    # Check data columns
    expected_columns = ['x1', 'x2', 'x3', 'x4', '类别']
    if not all(col in df.columns for col in expected_columns):
        print("Data columns do not match expected format. Please ensure columns are: x1, x2, x3, x4, 类别")
        return None

    # Prepare features and target
    X = df[['x1', 'x2', 'x3', 'x4']]
    y = df['类别']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Display class distribution
    print("\nClass distribution:")
    class_distribution = y.value_counts()
    print(class_distribution)

    # Perform PCA
    pca = PCA(n_components=min(4, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nPCA Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"PCA Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

    return pca, X_scaled, y, X_pca, scaler


def pca_classification_analysis(X_pca, y):
    """Perform classification using PCA components"""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train classifier on PCA components
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nClassification accuracy using PCA components: {accuracy:.4f}")

    # Display classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - PCA Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob


def visualize_pca_results(pca, X_pca, y, X_original):
    """Visualize PCA results"""
    # Create figure
    plt.figure(figsize=(18, 12))

    # Subplot 1: PCA projection (2D)
    plt.subplot(2, 3, 1)
    unique_classes = np.unique(y)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        if X_pca.shape[1] >= 2:
            plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1],
                        alpha=0.7, c=[colors[i]], label=f'Class {cls}', s=50)

    plt.title('PCA Projection (First 2 Components)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Explained variance
    plt.subplot(2, 3, 2)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.bar(range(1, len(explained_variance) + 1), explained_variance,
            alpha=0.7, label='Individual')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             'ro-', label='Cumulative')

    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True)

    # Subplot 3: PCA loadings (component coefficients)
    plt.subplot(2, 3, 3)
    feature_names = ['x1', 'x2', 'x3', 'x4']
    n_components = min(4, pca.components_.shape[0])

    for i in range(n_components):
        plt.bar(feature_names, pca.components_[i], alpha=0.7,
                label=f'PC{i + 1}')

    plt.title('PCA Component Loadings')
    plt.xlabel('Original Features')
    plt.ylabel('Loading Value')
    plt.legend()

    # Subplot 4: 3D PCA projection (if enough components)
    if X_pca.shape[1] >= 3:
        ax = plt.subplot(2, 3, 4, projection='3d')
        for i, cls in enumerate(unique_classes):
            ax.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], X_pca[y == cls, 2],
                       alpha=0.7, c=[colors[i]], label=f'Class {cls}', s=50)

        ax.set_title('PCA 3D Projection')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend()

    # Subplot 5: Correlation between original features and PC1
    plt.subplot(2, 3, 5)
    correlations = []
    for i, feature in enumerate(feature_names):
        correlation = np.corrcoef(X_original[:, i], X_pca[:, 0])[0, 1]
        correlations.append(correlation)

    plt.bar(feature_names, correlations, alpha=0.7)
    plt.title('Correlation with PC1')
    plt.xlabel('Original Features')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_feature_importance(pca, feature_names):
    """Analyze feature importance in PCA"""
    print(f"\n{'=' * 50}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'=' * 50}")

    n_components = pca.components_.shape[0]

    for i in range(n_components):
        print(f"\nPrincipal Component {i + 1}:")
        print(f"Explained variance: {pca.explained_variance_ratio_[i]:.4f}")

        # Get absolute loadings and sort them
        loadings = np.abs(pca.components_[i])
        sorted_indices = np.argsort(loadings)[::-1]

        for j in sorted_indices:
            print(f"  {feature_names[j]}: {pca.components_[i][j]:.4f}")


def pca_by_class_analysis(df):
    """Perform PCA analysis for each class separately"""
    unique_classes = df['类别'].unique()
    results = {}

    for class_label in unique_classes:
        print(f"\n{'=' * 50}")
        print(f"PCA ANALYSIS FOR CLASS: {class_label}")
        print(f"{'=' * 50}")

        # Create subset for this class
        class_data = df[df['类别'] == class_label]
        X_class = class_data[['x1', 'x2', 'x3', 'x4']]

        # Standardize and perform PCA
        scaler = StandardScaler()
        X_class_scaled = scaler.fit_transform(X_class)

        pca_class = PCA(n_components=min(4, X_class.shape[1]))
        X_class_pca = pca_class.fit_transform(X_class_scaled)

        print(f"Explained variance for class {class_label}: {pca_class.explained_variance_ratio_}")

        results[class_label] = {
            'pca': pca_class,
            'X_pca': X_class_pca,
            'scaler': scaler
        }

    return results


def main():
    # File path - replace with your Excel file path
    file_path = "LDA.xlsx"  # Change to your file path

    # Load data
    df = load_data(file_path)
    if df is None:
        return

    # Perform PCA analysis
    print(f"\n{'=' * 50}")
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print(f"{'=' * 50}")

    pca, X_scaled, y, X_pca, scaler = perform_pca_analysis(df)

    # Perform classification using PCA components
    clf, X_train, X_test, y_train, y_test, y_pred, y_prob = pca_classification_analysis(X_pca, y)

    # Visualize PCA results
    visualize_pca_results(pca, X_pca, y, X_scaled)

    # Analyze feature importance
    feature_names = ['x1', 'x2', 'x3', 'x4']
    analyze_feature_importance(pca, feature_names)

    # Perform PCA analysis for each class separately
    class_results = pca_by_class_analysis(df)

    # Display detailed results
    print(f"\n{'=' * 50}")
    print("PCA MODEL DETAILS")
    print(f"{'=' * 50}")

    print(f"Number of components: {pca.n_components_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Singular values: {pca.singular_values_}")

    # Show reconstruction error information
    print(f"\nReconstruction information:")
    print(f"Number of features: {pca.n_features_in_}")
    print(f"Number of samples: {pca.n_samples_}")


if __name__ == "__main__":
    main()