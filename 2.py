import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
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


def analyze_by_class(df):
    """Analyze data by class"""
    # Check data columns
    expected_columns = ['x1', 'x2', 'x3', 'x4', '类别']
    if not all(col in df.columns for col in expected_columns):
        print("Data columns do not match expected format. Please ensure columns are: x1, x2, x3, x4, 类别")
        return None

    # Get unique classes
    unique_classes = df['类别'].unique()
    print(f"\nFound {len(unique_classes)} unique classes: {unique_classes}")

    # Display class distribution
    print("\nClass distribution:")
    print(df['类别'].value_counts())

    return unique_classes


def perform_lda_for_each_class(df, unique_classes):
    """Perform LDA analysis for each class separately"""
    results = {}

    for class_label in unique_classes:
        print(f"\n{'=' * 50}")
        print(f"ANALYSIS FOR CLASS: {class_label}")
        print(f"{'=' * 50}")

        # Create binary classification: current class vs all others
        df_binary = df.copy()
        df_binary['target'] = (df_binary['类别'] == class_label).astype(int)

        # Prepare features and target
        X = df_binary[['x1', 'x2', 'x3', 'x4']]
        y = df_binary['target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Create and train LDA model
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        # Predict
        y_pred = lda.predict(X_test)
        y_prob = lda.predict_proba(X_test)[:, 1]  # Probability for class 1

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for class {class_label}: {accuracy:.4f}")

        # Store results
        results[class_label] = {
            'lda': lda,
            'X': X,
            'y': y,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy
        }

    return results


def perform_overall_lda(df):
    """Perform overall LDA analysis for all classes"""
    print(f"\n{'=' * 50}")
    print("OVERALL LDA ANALYSIS")
    print(f"{'=' * 50}")

    # Prepare features and target
    X = df[['x1', 'x2', 'x3', 'x4']]
    y = df['类别']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create and train LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict
    y_pred = lda.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Display classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=lda.classes_, yticklabels=lda.classes_)
    plt.title('Confusion Matrix - Overall Analysis')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return lda, X, y, X_test, y_test, y_pred


def visualize_class_analysis(results, df):
    """Visualize analysis for each class"""
    unique_classes = list(results.keys())
    n_classes = len(unique_classes)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot 1: Accuracy for each class
    accuracies = [results[cls]['accuracy'] for cls in unique_classes]
    axes[0].bar(unique_classes, accuracies)
    axes[0].set_title('Accuracy for Each Class (vs All Others)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.1)
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center')

    # Plot 2: Feature importance for each class
    feature_names = ['x1', 'x2', 'x3', 'x4']
    x_pos = np.arange(len(feature_names))
    width = 0.8 / n_classes

    for i, cls in enumerate(unique_classes):
        coef = results[cls]['lda'].coef_[0]  # Get coefficients
        axes[1].bar(x_pos + i * width, np.abs(coef), width, label=f'Class {cls}')

    axes[1].set_title('Feature Importance (Absolute Coefficients)')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Coefficient Value')
    axes[1].set_xticks(x_pos + width * (n_classes - 1) / 2)
    axes[1].set_xticklabels(feature_names)
    axes[1].legend()

    # Plot 3: Class distribution
    class_counts = df['类别'].value_counts()
    axes[2].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[2].set_title('Class Distribution')

    # Plot 4: LDA projection for overall analysis (if available)
    # This would require running the overall LDA first

    plt.tight_layout()
    plt.show()

    # Additional visualization: ROC curves for each class
    plt.figure(figsize=(12, 8))
    from sklearn.metrics import roc_curve, auc

    for cls in unique_classes:
        fpr, tpr, _ = roc_curve(results[cls]['y_test'], results[cls]['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curves for Each Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_lda_projection(lda, X, y, title):
    """Visualize LDA projection"""
    X_lda = lda.transform(X)

    plt.figure(figsize=(12, 8))
    unique_classes = np.unique(y)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        if X_lda.shape[1] >= 2:
            # 2D projection
            plt.scatter(X_lda[y == cls, 0], X_lda[y == cls, 1],
                        alpha=0.7, c=[colors[i]], label=f'Class {cls}', s=50)
        else:
            # 1D projection
            plt.scatter(X_lda[y == cls, 0], np.zeros_like(X_lda[y == cls, 0]),
                        alpha=0.7, c=[colors[i]], label=f'Class {cls}', s=50)

    plt.title(title)
    plt.xlabel('LDA Component 1')
    if X_lda.shape[1] >= 2:
        plt.ylabel('LDA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # File path - replace with your Excel file path
    file_path = "LDA.xlsx"  # Change to your file path

    # Load data
    df = load_data(file_path)
    if df is None:
        return

    # Analyze by class
    unique_classes = analyze_by_class(df)
    if unique_classes is None:
        return

    # Perform LDA for each class (binary classification: class vs all others)
    class_results = perform_lda_for_each_class(df, unique_classes)

    # Perform overall LDA analysis
    lda_overall, X_overall, y_overall, X_test_overall, y_test_overall, y_pred_overall = perform_overall_lda(df)

    # Visualize class analysis
    visualize_class_analysis(class_results, df)

    # Visualize overall LDA projection
    visualize_lda_projection(lda_overall, X_overall, y_overall, 'Overall LDA Projection')

    # Display detailed results for each class
    print(f"\n{'=' * 50}")
    print("DETAILED RESULTS FOR EACH CLASS")
    print(f"{'=' * 50}")

    for cls in unique_classes:
        result = class_results[cls]
        print(f"\nClass {cls}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Coefficients: {result['lda'].coef_[0]}")
        print(f"  Intercept: {result['lda'].intercept_[0]:.4f}")


if __name__ == "__main__":
    main()