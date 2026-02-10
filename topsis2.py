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


def load_and_preprocess_data(csv_file_path, weights_dict):
    """
    Load and preprocess data from CSV file for TOPSIS analysis
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path, header=0)

        print("First 5 rows of data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())
        print(f"\nData shape: {df.shape}")

        # Extract city names (assuming first column contains city names)
        if 'City' in df.columns:
            city_names = df['City'].values
        elif 'city' in df.columns:
            city_names = df['city'].values
        else:
            # Use index as city names
            city_names = [f"City_{i + 1}" for i in range(len(df))]
            print("No 'City' column found. Using generated city names.")

        # Select specific columns based on weights_dict keys
        feature_columns = []
        clean_feature_names = []

        for feature_name in weights_dict.keys():
            # Extract original column name from feature name in weights
            # Assuming format: "feature_name (ColX)"
            original_col_name = feature_name.split(' (Col')[0]

            if original_col_name in df.columns:
                feature_columns.append(original_col_name)
                clean_feature_names.append(feature_name)
            else:
                # Try to find similar column names
                found = False
                for col in df.columns:
                    if original_col_name.lower() in col.lower():
                        feature_columns.append(col)
                        clean_feature_names.append(feature_name)
                        print(f"Matched column: '{col}' -> '{feature_name}'")
                        found = True
                        break
                if not found:
                    print(f"Warning: Column matching '{original_col_name}' not found")

        if not feature_columns:
            raise ValueError("No valid feature columns found matching the weights")

        print(f"Selected columns: {feature_columns}")

        # Prepare feature data
        X_data = []
        final_feature_names = []

        for col, feature_name in zip(feature_columns, clean_feature_names):
            try:
                # Convert to numeric
                numeric_data = pd.to_numeric(df[col], errors='coerce')

                # Handle missing values
                missing_count = numeric_data.isna().sum()
                if missing_count > 0:
                    print(f"Warning: Column '{col}' has {missing_count} missing values. Filling with mean.")
                    numeric_data = numeric_data.fillna(numeric_data.mean())

                X_data.append(numeric_data.values)
                final_feature_names.append(feature_name)

            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                continue

        if not X_data:
            raise ValueError("No valid feature columns found after processing")

        X = np.column_stack(X_data)

        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Number of cities: {len(city_names)}")
        print(f"Number of criteria: {len(final_feature_names)}")

        return X, city_names, final_feature_names

    except Exception as e:
        print(f"Error in data loading: {e}")
        raise


def normalize_decision_matrix(X, criteria_direction):
    """
    Normalize the decision matrix for TOPSIS
    """
    print("\nStep 1: Normalizing Decision Matrix")

    # Normalize using vector normalization
    squared_matrix = X ** 2
    norm_factors = np.sqrt(np.sum(squared_matrix, axis=0))

    # Avoid division by zero
    norm_factors[norm_factors == 0] = 1e-10

    normalized_matrix = X / norm_factors

    print("Normalized matrix (first 5 rows):")
    print(pd.DataFrame(normalized_matrix[:5], columns=criteria_direction.keys()).round(4))

    return normalized_matrix


def apply_weights(normalized_matrix, weights, criteria_direction):
    """
    Apply weights to normalized decision matrix
    """
    print("\nStep 2: Applying Weights")

    # Create weight vector in correct order
    weight_vector = np.array([weights[feature] for feature in criteria_direction.keys()])

    # Apply weights
    weighted_matrix = normalized_matrix * weight_vector

    print("Weighted normalized matrix (first 5 rows):")
    print(pd.DataFrame(weighted_matrix[:5], columns=criteria_direction.keys()).round(4))

    return weighted_matrix


def calculate_ideal_solutions(weighted_matrix, criteria_direction):
    """
    Calculate positive and negative ideal solutions
    """
    print("\nStep 3: Calculating Ideal Solutions")

    positive_ideal = []
    negative_ideal = []

    for i, (feature, direction) in enumerate(criteria_direction.items()):
        if direction == 'max':  # Benefit criterion
            positive_ideal.append(np.max(weighted_matrix[:, i]))
            negative_ideal.append(np.min(weighted_matrix[:, i]))
        else:  # Cost criterion ('min')
            positive_ideal.append(np.min(weighted_matrix[:, i]))
            negative_ideal.append(np.max(weighted_matrix[:, i]))

    positive_ideal = np.array(positive_ideal)
    negative_ideal = np.array(negative_ideal)

    print("Positive Ideal Solution:")
    for feature, value in zip(criteria_direction.keys(), positive_ideal):
        print(f"  {feature}: {value:.6f}")

    print("Negative Ideal Solution:")
    for feature, value in zip(criteria_direction.keys(), negative_ideal):
        print(f"  {feature}: {value:.6f}")

    return positive_ideal, negative_ideal


def calculate_distances(weighted_matrix, positive_ideal, negative_ideal):
    """
    Calculate distances to ideal solutions
    """
    print("\nStep 4: Calculating Distances")

    # Calculate distance to positive ideal solution
    distance_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))

    # Calculate distance to negative ideal solution
    distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))

    print("Distances to positive ideal (first 10):", distance_positive[:10].round(4))
    print("Distances to negative ideal (first 10):", distance_negative[:10].round(4))

    return distance_positive, distance_negative


def calculate_topsis_scores(distance_positive, distance_negative):
    """
    Calculate TOPSIS scores
    """
    print("\nStep 5: Calculating TOPSIS Scores")

    # Avoid division by zero
    denominator = distance_positive + distance_negative
    denominator[denominator == 0] = 1e-10

    topsis_scores = distance_negative / denominator

    print("TOPSIS scores (first 10):", topsis_scores[:10].round(4))

    return topsis_scores


def determine_criteria_direction(feature_names):
    """
    Determine whether each criterion is benefit (max) or cost (min)
    Based on common sustainability indicators
    """
    criteria_direction = {}

    benefit_indicators = ['re_share_grid', 'pt_access_score', 'airport_hub_score',
                          'true_zerowaste', 'iso20121', 'recycling']

    cost_indicators = ['co2', 'precip', 'temp', 'waste', 'consumption', 'risk']

    for feature in feature_names:
        feature_lower = feature.lower()

        if any(benefit in feature_lower for benefit in benefit_indicators):
            criteria_direction[feature] = 'max'
        elif any(cost in feature_lower for cost in cost_indicators):
            criteria_direction[feature] = 'min'
        else:
            # Default to benefit criterion
            criteria_direction[feature] = 'max'
            print(f"Warning: Could not determine direction for '{feature}'. Defaulting to 'max'.")

    print("\nCriteria Directions:")
    for feature, direction in criteria_direction.items():
        print(f"  {feature}: {direction}")

    return criteria_direction


def topsis_method(X, city_names, feature_names, weights_dict):
    """
    Main TOPSIS method implementation
    """
    print("\n" + "=" * 70)
    print("TOPSIS Method Analysis")
    print("=" * 70)

    # Determine criteria directions
    criteria_direction = determine_criteria_direction(feature_names)

    # Step 1: Normalize decision matrix
    normalized_matrix = normalize_decision_matrix(X, criteria_direction)

    # Step 2: Apply weights
    weighted_matrix = apply_weights(normalized_matrix, weights_dict, criteria_direction)

    # Step 3: Calculate ideal solutions
    positive_ideal, negative_ideal = calculate_ideal_solutions(weighted_matrix, criteria_direction)

    # Step 4: Calculate distances
    distance_positive, distance_negative = calculate_distances(weighted_matrix, positive_ideal, negative_ideal)

    # Step 5: Calculate TOPSIS scores
    topsis_scores = calculate_topsis_scores(distance_positive, distance_negative)

    # Create results dataframe
    results_df = pd.DataFrame({
        'City': city_names,
        'TOPSIS_Score': topsis_scores,
        'Rank': np.argsort(np.argsort(-topsis_scores)) + 1,  # Rank 1 is best
        'Distance_Positive': distance_positive,
        'Distance_Negative': distance_negative
    }).sort_values('TOPSIS_Score', ascending=False)

    print("\nTOPSIS Results (Top 20 cities):")
    print(results_df.head(20).round(4))

    return results_df, weighted_matrix, positive_ideal, negative_ideal


def visualize_topsis_results(results_df, top_n=29):
    """Visualize TOPSIS results"""
    print("\nCreating visualizations...")

    # Plot 1: Top N cities by TOPSIS score
    plt.figure(figsize=(12, 8))
    top_cities = results_df.head(top_n)

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_cities)))
    bars = plt.barh(range(len(top_cities)), top_cities['TOPSIS_Score'], color=colors)

    plt.yticks(range(len(top_cities)), top_cities['City'])
    plt.xlabel('TOPSIS Score', fontsize=12)
    plt.title(f'Top {top_n} Cities - TOPSIS Ranking', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add values on bars
    for i, (bar, score) in enumerate(zip(bars, top_cities['TOPSIS_Score'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Plot 2: Score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['TOPSIS_Score'], bins=29, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('TOPSIS Score', fontsize=12)
    plt.ylabel('Number of Cities', fontsize=12)
    plt.title('Distribution of TOPSIS Scores', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.show()

    # Plot 3: Rank vs Score scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Rank'], results_df['TOPSIS_Score'], alpha=0.6, color='red')
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('TOPSIS Score', fontsize=12)
    plt.title('Rank vs TOPSIS Score', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.show()


def create_comprehensive_report(results_df, weights_dict, top_n=10):
    """Create comprehensive analysis report"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 70)

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total cities analyzed: {len(results_df)}")
    print(f"Average TOPSIS score: {results_df['TOPSIS_Score'].mean():.4f}")
    print(f"Standard deviation: {results_df['TOPSIS_Score'].std():.4f}")
    print(f"Maximum score: {results_df['TOPSIS_Score'].max():.4f}")
    print(f"Minimum score: {results_df['TOPSIS_Score'].min():.4f}")

    # Top performers
    print(f"\nTop {top_n} Performing Cities:")
    top_cities = results_df.head(top_n)
    for i, (_, row) in enumerate(top_cities.iterrows(), 1):
        print(f"{i:2d}. {row['City']:29} Score: {row['TOPSIS_Score']:.4f}")

    # Weight summary
    print(f"\nCriteria Weights Used:")
    for feature, weight in weights_dict.items():
        print(f"  {feature}: {weight:.4f} ({weight * 100:.2f}%)")

    # Performance categories
    print(f"\nPerformance Categories:")
    score_q1 = results_df['TOPSIS_Score'].quantile(0.75)
    score_q2 = results_df['TOPSIS_Score'].quantile(0.5)
    score_q3 = results_df['TOPSIS_Score'].quantile(0.25)

    excellent = results_df[results_df['TOPSIS_Score'] >= score_q1]
    good = results_df[(results_df['TOPSIS_Score'] >= score_q2) & (results_df['TOPSIS_Score'] < score_q1)]
    fair = results_df[(results_df['TOPSIS_Score'] >= score_q3) & (results_df['TOPSIS_Score'] < score_q2)]
    poor = results_df[results_df['TOPSIS_Score'] < score_q3]

    print(f"Excellent (Top 25%): {len(excellent)} cities")
    print(f"Good (25-50%): {len(good)} cities")
    print(f"Fair (50-75%): {len(fair)} cities")
    print(f"Poor (Bottom 25%): {len(poor)} cities")


def create_sample_data():
    """Create sample city data for testing"""
    np.random.seed(42)
    n_cities = 50

    # Create sample data with the specified features
    data = {
        'City': [f'City_{i + 1}' for i in range(n_cities)],
        'egrid_co2_kg_per_mwh': np.random.normal(500, 100, n_cities),
        're_share_grid_1': np.random.uniform(0, 100, n_cities),
        'water_risk_0to5': np.random.uniform(0, 100, n_cities),
        'feb_temp_c': np.random.uniform(-10, 30, n_cities),
        'feb_precip_mm': np.random.uniform(0, 200, n_cities),
        'pt_access_score_0to1': np.random.uniform(0, 1, n_cities),
        'airport_hub_score_0to1': np.random.uniform(0, 1, n_cities),
        'iso20121': np.random.uniform(0, 1, n_cities),
        'true_zerowaste': np.random.uniform(0, 1, n_cities)
    }

    df = pd.DataFrame(data)

    # Save sample data
    df.to_csv('sample_city_data.csv', index=False)
    print("Sample city data created: 'sample_city_data.csv'")
    return 'sample_city_data.csv'


# Main program
if __name__ == "__main__":
    print("Starting TOPSIS City Optimization Analysis...")

    # Define weights from your entropy method results
    weights_dict = {
        "iso20121 (Col15)": 0,
        "true_zerowaste (Col16)": 0,
        "re_share_grid_1 (Col7)": 0.240465,
        "egrid_co2_kg_per_mwh (Col6)": 0.193623,
        "pt_access_score_0to1 (Col12)": 0.167048,
        "airport_hub_score_0to1 (Col13)": 0.166403,
        "feb_precip_mm (Col11)": 0.070385 ,
        "feb_temp_c (Col10)": 0.086693,
        "water_risk_0to5 (Col8)": 0.232462
    }

    print("Using pre-calculated weights from Entropy Method:")
    for feature, weight in weights_dict.items():
        print(f"  {feature}: {weight:.4f} ({weight * 100:.2f}%)")

    # Ask user if they want to use sample data
    use_sample = input("\nDo you want to use sample data for testing? (y/n): ").lower().strip()

    if use_sample == 'y':
        csv_file_path = create_sample_data()
    else:
        csv_file_path = "Book2.csv"

    try:
        print(f"\nLoading data from: {csv_file_path}")
        X, city_names, feature_names = load_and_preprocess_data(csv_file_path, weights_dict)

        # Apply TOPSIS method
        results_df, weighted_matrix, pos_ideal, neg_ideal = topsis_method(
            X, city_names, feature_names, weights_dict
        )

        # Create comprehensive report
        create_comprehensive_report(results_df, weights_dict)

        # Visualize results
        visualize_topsis_results(results_df)

        # Save results to CSV
        output_file = 'topsis_city_ranking.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        print("\n" + "=" * 70)
        print("TOPSIS ANALYSIS COMPLETE")
        print("=" * 70)
        print("Cities have been ranked based on sustainability criteria.")
        print("Higher TOPSIS scores indicate better overall performance.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("Please check your data format and try again.")