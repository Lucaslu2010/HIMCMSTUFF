import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path):
    """
    加载和预处理NBA球员数据
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查列名
    expected_columns = ['Name', 'Points Per Game', 'Assists Per Game',
                        'Blocks Per Game', 'Steals Per Game', 'Misses Per Game']

    # 如果列名不匹配，使用第一行作为列名或重命名
    if len(df.columns) == 6:
        df.columns = expected_columns

    print("数据前5行：")
    print(df.head())
    print("\n数据基本信息：")
    print(df.info())

    return df


def apply_weights_and_normalization(df, weights=None):
    """
    应用权重和正向化处理
    """
    # 定义特征列
    feature_columns = ['Points Per Game', 'Assists Per Game', 'Blocks Per Game',
                       'Steals Per Game', 'Misses Per Game']

    # 默认权重（可以根据需要调整）
    if weights is None:
        weights = {
            'Points Per Game': 0.25,  # 得分权重
            'Assists Per Game': 0.20,  # 助攻权重
            'Blocks Per Game': 0.15,  # 盖帽权重
            'Steals Per Game': 0.15,  # 抢断权重
            'Misses Per Game': -0.25  # 失误为负向指标
        }

    # 复制数据框进行处理
    df_processed = df.copy()

    # 应用权重和正向化
    for col in feature_columns:
        weight = weights[col]

        if weight < 0:  # 负向指标，需要正向化
            # 使用倒数法或减法进行正向化
            df_processed[col] = 1 / (df_processed[col] + 0.1)  # 加0.1避免除零
            # 或者使用：df_processed[col] = df_processed[col].max() - df_processed[col]
            weight = abs(weight)

        # 应用权重
        df_processed[col] = df_processed[col] * weight

    print("\n应用权重后的数据统计：")
    print(df_processed[feature_columns].describe())

    return df_processed, feature_columns


def perform_pca_analysis(df_processed, feature_columns):
    """
    执行PCA分析
    """
    # 提取特征数据
    X = df_processed[feature_columns].values

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 创建包含PCA结果的数据框
    pca_columns = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['Name'] = df_processed['Name'].values

    return pca, X_scaled, df_pca, scaler


def analyze_pca_results(pca, df_pca, feature_columns):
    """
    分析PCA结果
    """
    print("\n" + "=" * 50)
    print("PCA分析结果")
    print("=" * 50)

    # 1. 解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"\n1. 主成分解释方差比：")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"   主成分 {i + 1}: {var:.4f} ({var * 100:.2f}%) - 累计: {cum_var * 100:.2f}%")

    # 2. 主成分载荷
    print(f"\n2. 主成分载荷矩阵：")
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings,
                               index=feature_columns,
                               columns=[f'PC{i + 1}' for i in range(len(feature_columns))])
    print(loadings_df.round(4))

    # 3. 选择主成分数量（通常选择累计解释方差>85%的成分）
    n_components = np.argmax(cumulative_variance >= 0.85) + 1
    print(f"\n3. 建议主成分数量: {n_components} (累计解释方差: {cumulative_variance[n_components - 1]:.2%})")

    return explained_variance_ratio, cumulative_variance, loadings_df, n_components


def visualize_results(pca, df_pca, explained_variance_ratio, feature_columns):
    """
    可视化PCA结果
    """
    # 设置图形样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 碎石图
    axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    axes[0, 0].set_title('Scree Plot - 主成分解释方差比')
    axes[0, 0].set_xlabel('主成分')
    axes[0, 0].set_ylabel('解释方差比')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 累计解释方差图
    cumulative_variance = np.cumsum(explained_variance_ratio)
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    axes[0, 1].set_title('累计解释方差')
    axes[0, 1].set_xlabel('主成分数量')
    axes[0, 1].set_ylabel('累计解释方差比')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.85, color='g', linestyle='--', label='85%阈值')
    axes[0, 1].legend()

    # 3. 前两个主成分的散点图
    if len(df_pca) > 1:
        scatter = axes[1, 0].scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)
        axes[1, 0].set_title('PCA - 前两个主成分')
        axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
        axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
        axes[1, 0].grid(True, alpha=0.3)

        # 添加球员名称标签（只显示部分，避免过于拥挤）
        for i, name in enumerate(df_pca['Name']):
            if i % max(1, len(df_pca) // 10) == 0:  # 只显示约10个标签
                axes[1, 0].annotate(name, (df_pca['PC1'].iloc[i], df_pca['PC2'].iloc[i]),
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 4. 主成分载荷热图
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    sns.heatmap(loadings,
                xticklabels=[f'PC{i + 1}' for i in range(len(feature_columns))],
                yticklabels=feature_columns,
                annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=axes[1, 1])
    axes[1, 1].set_title('主成分载荷热图')

    plt.tight_layout()
    plt.show()


def create_player_ranking(df_pca, n_components):
    """
    创建球员综合排名
    """
    # 使用前n个主成分计算综合得分
    pca_columns = [f'PC{i + 1}' for i in range(n_components)]
    df_pca['Composite_Score'] = df_pca[pca_columns].mean(axis=1)

    # 按综合得分排序
    df_ranked = df_pca[['Name', 'Composite_Score'] + pca_columns].sort_values('Composite_Score', ascending=False)

    print(f"\n4. 球员综合排名（基于前{n_components}个主成分）：")
    print(df_ranked.head(10).round(4))

    return df_ranked


def main():
    """
    主函数
    """
    # 文件路径 - 请根据实际情况修改
    file_path = "nba_players_data.xlsx"  # 替换为您的Excel文件路径

    try:
        # 1. 加载数据
        print("正在加载数据...")
        df = load_and_preprocess_data(file_path)

        # 2. 应用权重和正向化
        print("\n正在应用权重和正向化...")
        df_processed, feature_columns = apply_weights_and_normalization(df)

        # 3. 执行PCA分析
        print("\n正在执行PCA分析...")
        pca, X_scaled, df_pca, scaler = perform_pca_analysis(df_processed, feature_columns)

        # 4. 分析PCA结果
        explained_variance_ratio, cumulative_variance, loadings_df, n_components = analyze_pca_results(
            pca, df_pca, feature_columns)

        # 5. 可视化结果
        print("\n正在生成可视化图表...")
        visualize_results(pca, df_pca, explained_variance_ratio, feature_columns)

        # 6. 创建球员排名
        df_ranked = create_player_ranking(df_pca, n_components)

        # 7. 保存结果
        output_file = "nba_players_pca_analysis.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, sheet_name='原始数据', index=False)
            df_processed.to_excel(writer, sheet_name='处理后数据', index=False)
            df_pca.to_excel(writer, sheet_name='PCA结果', index=False)
            df_ranked.to_excel(writer, sheet_name='球员排名', index=False)
            loadings_df.to_excel(writer, sheet_name='主成分载荷')

        print(f"\n分析完成！结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        print("请确保文件路径正确，且文件格式为Excel")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()