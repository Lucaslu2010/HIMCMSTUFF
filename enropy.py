import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib

matplotlib.use('TkAgg')


def load_and_preprocess_data(csv_file_path):
    """
    Load and preprocess data from CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path, header=0)

        print("First 5 rows of data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())
        print(f"\nData shape: {df.shape}")

        # Check data types
        print("\nData types:")
        print(df.dtypes)

        # Select specific columns: 6,7,8,10,11,12,13,15,16 (0-indexed: 5,6,7,9,10,11,12,14,15)
        target_columns_indices = [5, 6, 7, 9, 10, 11, 12, 14, 15]

        # Get actual column names based on indices
        feature_columns = []
        actual_indices_used = []

        for idx in target_columns_indices:
            if idx < len(df.columns):
                feature_columns.append(df.columns[idx])
                actual_indices_used.append(idx)
            else:
                print(f"Warning: Column index {idx} is out of range")

        if not feature_columns:
            raise ValueError("No valid feature columns found")

        print(f"Selected columns (indices {actual_indices_used}): {feature_columns}")

        # Prepare feature data - convert to numeric and handle missing values
        X_data = []
        clean_feature_names = []

        for col in feature_columns:
            try:
                # Convert to numeric, coercing errors to NaN
                numeric_data = pd.to_numeric(df[col], errors='coerce')

                # Check for missing values
                missing_count = numeric_data.isna().sum()
                if missing_count > 0:
                    print(f"Warning: Column '{col}' has {missing_count} missing values. Filling with mean.")
                    numeric_data = numeric_data.fillna(numeric_data.mean())

                X_data.append(numeric_data.values)
                clean_feature_names.append(f"{col} (Col{df.columns.get_loc(col) + 1})")

            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                continue

        if not X_data:
            raise ValueError("No valid feature columns found after processing")

        X = np.column_stack(X_data)
        feature_names = clean_feature_names

        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Feature names: {feature_names}")

        return X, feature_names

    except Exception as e:
        print(f"Error in data loading: {e}")
        raise


def entropy_weight_method(X, feature_names):
    """
    Calculate weights using Entropy Weight Method
    """
    print("\n" + "=" * 70)
    print("Entropy Weight Method Analysis")
    print("=" * 70)

    # Step 1: Data standardization (min-max normalization)
    print("\nStep 1: Data Standardization")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    print("Normalized data (first 5 rows):")
    print(pd.DataFrame(X_normalized[:5], columns=feature_names).round(4))

    # Step 2: Calculate the proportion of each indicator
    print("\nStep 2: Calculate Proportion Matrix")
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    X_proportion = X_normalized + epsilon
    X_proportion = X_proportion / np.sum(X_proportion, axis=0)

    print("Proportion matrix (first 5 rows):")
    print(pd.DataFrame(X_proportion[:5], columns=feature_names).round(4))

    # Step 3: Calculate entropy for each indicator
    print("\nStep 3: Calculate Entropy")
    n_samples = X.shape[0]
    k = 1 / np.log(n_samples)  # entropy coefficient

    entropy_values = -k * np.sum(X_proportion * np.log(X_proportion), axis=0)

    print("Entropy values for each indicator:")
    for name, entropy in zip(feature_names, entropy_values):
        print(f"  {name}: {entropy:.6f}")

    # Step 4: Calculate diversity factor (degree of divergence)
    print("\nStep 4: Calculate Diversity Factor")
    diversity_factors = 1 - entropy_values

    print("Diversity factors for each indicator:")
    for name, diversity in zip(feature_names, diversity_factors):
        print(f"  {name}: {diversity:.6f}")

    # Step 5: Calculate weights
    print("\nStep 5: Calculate Weights")
    weights = diversity_factors / np.sum(diversity_factors)

    # Create results dataframe
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Entropy': entropy_values,
        'Diversity_Factor': diversity_factors,
        'Weight': weights,
        'Weight_Percentage': weights * 100
    }).sort_values('Weight', ascending=False)

    print("\nFinal weights calculation:")
    print(results_df.round(6))

    return results_df


def visualize_entropy_results(results_df):
    """Visualize entropy method results"""
    print("\nCreating visualizations...")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Entropy Values
    axes[0, 0].barh(range(len(results_df)), results_df['Entropy'], color='lightblue')
    axes[0, 0].set_yticks(range(len(results_df)))
    axes[0, 0].set_yticklabels(results_df['Feature'])
    axes[0, 0].set_title('Entropy Values', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Entropy')
    for i, v in enumerate(results_df['Entropy']):
        axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

    # Plot 2: Diversity Factors
    axes[0, 1].barh(range(len(results_df)), results_df['Diversity_Factor'], color='lightcoral')
    axes[0, 1].set_yticks(range(len(results_df)))
    axes[0, 1].set_yticklabels(results_df['Feature'])
    axes[0, 1].set_title('Diversity Factors', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Diversity Factor (1 - Entropy)')
    for i, v in enumerate(results_df['Diversity_Factor']):
        axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')

    # Plot 3: Final Weights
    axes[1, 0].barh(range(len(results_df)), results_df['Weight'], color='lightgreen')
    axes[1, 0].set_yticks(range(len(results_df)))
    axes[1, 0].set_yticklabels(results_df['Feature'])
    axes[1, 0].set_title('Final Weights', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Weight')
    for i, v in enumerate(results_df['Weight']):
        axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

    # Plot 4: Weight Percentages
    axes[1, 1].barh(range(len(results_df)), results_df['Weight_Percentage'], color='gold')
    axes[1, 1].set_yticks(range(len(results_df)))
    axes[1, 1].set_yticklabels(results_df['Feature'])
    axes[1, 1].set_title('Weight Percentages', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Weight Percentage (%)')
    for i, v in enumerate(results_df['Weight_Percentage']):
        axes[1, 1].text(v + 0.5, i, f'{v:.2f}%', va='center')

    plt.tight_layout()
    plt.show()

    # Create final summary plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = plt.barh(range(len(results_df)), results_df['Weight_Percentage'], color=colors)

    plt.yticks(range(len(results_df)), results_df['Feature'])
    plt.xlabel('Weight Percentage (%)', fontsize=12)
    plt.title('Final Weight Distribution - Entropy Method', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add values on bars
    for i, (bar, weight) in enumerate(zip(bars, results_df['Weight_Percentage'])):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{weight:.2f}%', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Create pie chart for weight distribution
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(results_df['Weight_Percentage'],
                                       labels=results_df['Feature'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=plt.cm.Set3(np.linspace(0, 1, len(results_df))))

    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    plt.title('Weight Distribution - Entropy Method', fontsize=16, fontweight='bold')
    plt.show()


def calculate_comprehensive_results(results_df):
    """Calculate comprehensive results and statistics"""
    print("\n" + "=" * 70)
    print("Comprehensive Results Summary")
    print("=" * 70)

    # Basic statistics
    print(f"Number of indicators: {len(results_df)}")
    print(f"Total weight: {results_df['Weight'].sum():.6f}")
    print(f"Average weight: {results_df['Weight'].mean():.6f}")
    print(f"Standard deviation of weights: {results_df['Weight'].std():.6f}")

    # Weight distribution analysis
    max_weight = results_df['Weight'].max()
    min_weight = results_df['Weight'].min()
    weight_range = max_weight - min_weight

    print(f"\nWeight Distribution Analysis:")
    print(f"Maximum weight: {max_weight:.6f} ({results_df.loc[results_df['Weight'].idxmax(), 'Feature']})")
    print(f"Minimum weight: {min_weight:.6f} ({results_df.loc[results_df['Weight'].idxmin(), 'Feature']})")
    print(f"Weight range: {weight_range:.6f}")

    # Entropy analysis
    print(f"\nEntropy Analysis:")
    print(f"Average entropy: {results_df['Entropy'].mean():.6f}")
    print(f"Maximum entropy: {results_df['Entropy'].max():.6f}")
    print(f"Minimum entropy: {results_df['Entropy'].min():.6f}")

    # Final ranking
    print(f"\nFinal Weight Ranking:")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:30} {row['Weight_Percentage']:6.2f}%")

    return results_df


def create_sample_data():
    """Create sample 16-column data for testing"""
    np.random.seed(42)
    n_samples = 100

    # Create sample data with 16 columns
    data = {}
    for i in range(16):
        col_name = f'Column_{i + 1}'
        # Vary the distributions to create different entropy patterns
        if i in [5, 6, 7]:  # Columns 6,7,8 - high variability
            data[col_name] = np.random.normal(50, 20, n_samples)
        elif i in [9, 10, 11]:  # Columns 10,11,12 - medium variability
            data[col_name] = np.random.normal(50, 10, n_samples)
        elif i in [12, 14, 15]:  # Columns 13,15,16 - low variability
            data[col_name] = np.random.normal(50, 5, n_samples)
        else:  # Other columns
            data[col_name] = np.random.normal(50, 15, n_samples)

    df = pd.DataFrame(data)

    # Save sample data
    df.to_csv('sample_16column_data.csv', index=False)
    print("Sample 16-column data created: 'sample_16column_data.csv'")
    return 'sample_16column_data.csv'


# Main program
if __name__ == "__main__":
    print("Starting Entropy Weight Method Analysis...")
    print("This program analyzes columns 6,7,8,10,11,12,13,15,16 (1-indexed)")

    # Ask user if they want to use sample data
    use_sample = input("Do you want to use sample data for testing? (y/n): ").lower().strip()

    if use_sample == 'y':
        csv_file_path = create_sample_data()
    else:
        csv_file_path = "Book2.csv"

    try:
        print(f"Loading data from: {csv_file_path}")
        X, feature_names = load_and_preprocess_data(csv_file_path)

        print(f"Features: {feature_names}")
        print(f"Data shape: {X.shape}")

        # Apply Entropy Weight Method
        results_df = entropy_weight_method(X, feature_names)

        # Calculate comprehensive results
        final_results = calculate_comprehensive_results(results_df)

        # Visualize results
        visualize_entropy_results(final_results)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("The entropy weight method has successfully calculated the weights.")
        print("Lower entropy indicates higher information content and thus higher weight.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("Please check your data format and try again.")