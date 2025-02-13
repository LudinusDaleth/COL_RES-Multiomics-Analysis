# =============================================================================
# Import necessary libraries for data manipulation, statistics, plotting, and
# network analysis.
# =============================================================================
import pandas as pd                     # Data handling and manipulation
import numpy as np                      # Numerical operations
import seaborn as sns                   # Advanced visualization
import matplotlib.pyplot as plt         # Basic plotting
from scipy.stats import spearmanr, ttest_ind  # Statistical tests: Spearman correlation and t-test
from sklearn.decomposition import PCA   # Principal Component Analysis
from sklearn.preprocessing import StandardScaler  # Feature scaling
import matplotlib.patches as mpatches     # For custom legend patches in plots
import matplotlib.lines as mlines         # For custom legend lines
from matplotlib.patches import Ellipse    # For plotting confidence ellipses in PCA
import networkx as nx                     # Graph and network analysis
from scipy.cluster.hierarchy import dendrogram, linkage  # Hierarchical clustering

# =============================================================================
# Set a modern theme for all plots using seaborn.
# =============================================================================
sns.set_theme(style="whitegrid", context="talk")

# =============================================================================
# ----------------------- Data Loading & Processing -------------------------
# =============================================================================

# File paths for the datasets
metabolomics_path = "metabolomics_final.csv"
microbiome_path = "16s_data_final.csv"

# Load datasets from CSV files into DataFrames
metabolomics_df = pd.read_csv(metabolomics_path)
microbiome_df = pd.read_csv(microbiome_path)

# ------------------ Microbiome Data Aggregation ----------------------------
# Group the microbiome data by mouse and taxonomic information, then sum values
microbiome_grouped = microbiome_df.groupby(
    ['mouse_id', 'sample_day', 'mouse_microbiome', 'group', 'Genus']
)['value'].sum().reset_index()

# Pivot the grouped data so that each Genus becomes its own column
microbiome_pivot = microbiome_grouped.pivot_table(
    index=['mouse_id', 'sample_day', 'mouse_microbiome', 'group'],
    columns='Genus', values='value', fill_value=0
).reset_index()

# ----------------- Metabolomics Data Normalization -------------------------
# Apply a log transformation (log1p) to numeric metabolite columns (from 5th column onward)
metabolomics_numeric = metabolomics_df.iloc[:, 4:]
metabolomics_df.iloc[:, 4:] = np.log1p(metabolomics_numeric)

# ---------------- Merge Datasets ---------------------------------------------
# Merge metabolomics and microbiome data on shared columns using an inner join
merged_df = pd.merge(metabolomics_df, microbiome_pivot,
                     on=['mouse_id', 'sample_day', 'mouse_microbiome', 'group'],
                     how='inner')

# ----------------- Split Data by Microbiome Type -----------------------------
# Filter merged data to separate SPF and OMM12 mouse microbiome groups
spf_df = merged_df[merged_df['mouse_microbiome'] == 'SPF']
omm12_df = merged_df[merged_df['mouse_microbiome'] == 'OMM12']

# =============================================================================
# ---------------- Differential Fold-Change Heatmaps --------------------------
# =============================================================================

def compute_fold_change(df, feature_cols, group_col='group', control='PBS'):
    """
    Compute log-difference (fold change) for each feature relative to the control group.
    
    Parameters:
        df (DataFrame): Data containing the feature values.
        feature_cols (list): Columns representing the features.
        group_col (str): Column name defining groups (default: 'group').
        control (str): The control group to compare against (default: 'PBS').

    Returns:
        DataFrame: Fold change values with groups as rows and features as columns.
    """
    groups = df[group_col].unique()
    fc_data = {}
    # Calculate mean of features in the control group
    control_means = df[df[group_col] == control][feature_cols].mean()
    # Compute the log difference (fold change) for each group relative to the control
    for grp in groups:
        grp_means = df[df[group_col] == grp][feature_cols].mean()
        fc_data[grp] = grp_means - control_means  # Since data is log-transformed
    fc_df = pd.DataFrame(fc_data).T  # Transpose so that rows represent groups
    # Remove the control group from the result (since it's compared against itself)
    fc_df = fc_df.drop(index=control, errors='ignore')
    return fc_df

# Define the feature columns for metabolites and microbiome data
metabolite_cols = metabolomics_df.columns[4:]
microbiome_feature_cols = microbiome_pivot.columns[4:]

# Compute fold changes for each dataset (SPF and OMM12)
spf_met_fc = compute_fold_change(spf_df, metabolite_cols)
omm12_met_fc = compute_fold_change(omm12_df, metabolite_cols)
spf_micro_fc = compute_fold_change(spf_df, microbiome_feature_cols)
omm12_micro_fc = compute_fold_change(omm12_df, microbiome_feature_cols)

def plot_heatmap(data, title, xlabel='Features', ylabel='Group', save_path=None):
    """
    Plot a heatmap of fold-change values with hatch overlays to indicate direction:
      - Hatch '//' for positive values
      - Hatch '\\' for negative values
    
    Parameters:
        data (DataFrame): Data to be visualized.
        title (str): Title for the heatmap.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        save_path (str): Optional file path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(data, cmap='vlag', center=0, annot=False,
                     cbar=True, linewidths=0.5, linecolor='grey')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Overlay hatch patterns to denote positive/negative differences
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if value > 0:
                hatch = '//'
            elif value < 0:
                hatch = '\\\\'
            else:
                hatch = ''
            # Add a rectangle patch with the appropriate hatch pattern over each cell
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       hatch=hatch, edgecolor='grey', lw=0))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Generate and save fold-change heatmaps for metabolites and microbiome features
plot_heatmap(spf_met_fc, 'SPF Metabolite Fold Change (vs. PBS)', save_path='spf_metabolite_fold_change.png')
plot_heatmap(omm12_met_fc, 'OMM12 Metabolite Fold Change (vs. PBS)', save_path='omm12_metabolite_fold_change.png')
plot_heatmap(spf_micro_fc, 'SPF Microbiome (Genus) Fold Change (vs. PBS)', save_path='spf_microbiome_fold_change.png')
plot_heatmap(omm12_micro_fc, 'OMM12 Microbiome (Genus) Fold Change (vs. PBS)', save_path='omm12_microbiome_fold_change.png')

# =============================================================================
# -------------------------- PCA Visualizations -----------------------------
# =============================================================================

def plot_confidence_ellipse(x, y, ax, n_std=2.0, color='black', fill=True, alpha=0.1, **kwargs):
    """
    Plot a confidence ellipse representing n_std standard deviations for the given x and y data.
    
    Parameters:
        x, y (array-like): Data points for x and y axes.
        ax (Axes): Matplotlib Axes object to plot on.
        n_std (float): Number of standard deviations for the ellipse's radius.
        color (str): Color of the ellipse edge (and fill if applicable).
        fill (bool): Whether to fill the ellipse with a transparent color.
        alpha (float): Transparency level for the filled ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    # Compute eigenvalues and eigenvectors for the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    # Calculate the rotation angle of the ellipse
    theta = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    facecolor = color if fill else 'none'
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height,
                      angle=theta, edgecolor=color, facecolor=facecolor,
                      alpha=alpha if fill else 1, linestyle='--', **kwargs)
    ax.add_patch(ellipse)

def plot_pca(df, feature_cols, title, save_path=None):
    """
    Perform PCA on selected features and create a scatter plot that distinguishes:
      - Groups by color.
      - Mouse microbiome types by marker shape.
    Also adds 95% confidence ellipses for each microbiome type.
    
    Parameters:
        df (DataFrame): Data containing the features and metadata.
        feature_cols (list): Columns to be used for PCA.
        title (str): Title for the PCA plot.
        save_path (str): Optional file path to save the figure.
    """
    # Fill missing values with 0 and standardize features
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce dimensions to 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create a new DataFrame for PCA results and add group and microbiome information
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['group'] = df['group'].values
    pca_df['mouse_microbiome'] = df['mouse_microbiome'].values

    # Determine unique groups and microbiome types for plotting
    unique_groups = pca_df['group'].unique()
    unique_micro = pca_df['mouse_microbiome'].unique()
    
    # Define color palette for groups and marker shapes for microbiome types
    palette = sns.color_palette("deep", n_colors=len(unique_groups))
    marker_dict = {micro: marker for micro, marker in zip(unique_micro, ['o', 'X'])}
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Plot each subgroup combination with its designated color and marker
    for grp in unique_groups:
        for micro in unique_micro:
            subset = pca_df[(pca_df['group'] == grp) & (pca_df['mouse_microbiome'] == micro)]
            if not subset.empty:
                sns.scatterplot(x='PC1', y='PC2', data=subset,
                                color=palette[list(unique_groups).index(grp)],
                                marker=marker_dict[micro],
                                s=200, ax=ax,
                                label=f'{grp}, {micro}')
    
    # Add confidence ellipses for each microbiome type using a separate color palette
    ellipse_palette = sns.color_palette("Set1", n_colors=len(unique_micro))
    for idx, micro in enumerate(unique_micro):
        micro_subset = pca_df[pca_df['mouse_microbiome'] == micro]
        plot_confidence_ellipse(micro_subset['PC1'].values, micro_subset['PC2'].values, ax,
                                n_std=2.0, color=ellipse_palette[idx], fill=True, alpha=0.3)
    
    # Create custom legends for groups and microbiome types
    group_handles = [mpatches.Patch(color=palette[i], label=grp)
                     for i, grp in enumerate(unique_groups)]
    micro_handles = [mlines.Line2D([], [], color='black', marker=marker_dict[micro],
                                   linestyle='None', markersize=10, label=micro)
                     for micro in unique_micro]
    
    leg1 = ax.legend(handles=group_handles, title='Group',
                     loc='upper right', bbox_to_anchor=(1.25, 1))
    leg2 = ax.legend(handles=micro_handles, title='Mouse Microbiome',
                     loc='lower right', bbox_to_anchor=(1.25, 0))
    ax.add_artist(leg1)
    
    plt.title(title, fontsize=16)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Generate PCA plots for both metabolomics and microbiome features
plot_pca(merged_df, metabolite_cols, 'PCA of Metabolomics Data', save_path='pca_metabolomics.png')
plot_pca(merged_df, microbiome_feature_cols, 'PCA of Microbiome Data', save_path='pca_microbiome.png')

# =============================================================================
# ------------------------- Correlation Heatmaps ------------------------------
# =============================================================================

def compute_correlation(df, met_cols, mic_cols):
    """
    Compute Spearman correlations and corresponding p-values between each pair of
    metabolite and microbial genus features.
    
    Parameters:
        df (DataFrame): Merged dataset containing both types of features.
        met_cols (list): Columns representing metabolite features.
        mic_cols (list): Columns representing microbial genus features.
    
    Returns:
        correlation_df (DataFrame): Spearman correlation coefficients.
        p_values_df (DataFrame): Corresponding p-values for each correlation.
    """
    correlation_matrix = np.zeros((len(mic_cols), len(met_cols)))
    p_values = np.zeros((len(mic_cols), len(met_cols)))
    
    # Loop through each combination of microbe and metabolite features
    for i, mic in enumerate(mic_cols):
        for j, met in enumerate(met_cols):
            # Only compute correlation if both features have variation
            if df[mic].nunique() > 1 and df[met].nunique() > 1:
                corr, p_val = spearmanr(df[mic], df[met])
            else:
                corr, p_val = np.nan, np.nan
            correlation_matrix[i, j] = corr
            p_values[i, j] = p_val
            
    correlation_df = pd.DataFrame(correlation_matrix, index=mic_cols, columns=met_cols)
    p_values_df = pd.DataFrame(p_values, index=mic_cols, columns=met_cols)
    return correlation_df, p_values_df

# Compute correlations separately for SPF and OMM12 datasets
spf_corr, spf_pval = compute_correlation(spf_df, metabolite_cols, microbiome_feature_cols)
omm12_corr, omm12_pval = compute_correlation(omm12_df, metabolite_cols, microbiome_feature_cols)

def plot_correlation(corr_df, pval_df, title, save_path=None):
    """
    Plot a heatmap of Spearman correlations between metabolites and microbes.
    Overlays an asterisk '*' on cells where the correlation is statistically significant
    (p-value < 0.05).
    
    Parameters:
        corr_df (DataFrame): DataFrame containing correlation coefficients.
        pval_df (DataFrame): DataFrame containing corresponding p-values.
        title (str): Title for the heatmap.
        save_path (str): Optional file path to save the figure.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(corr_df, cmap='coolwarm', center=0, annot=False,
                     cbar=True, linewidths=0.5, linecolor='grey')
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Overlay '*' for significant correlations
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if pval_df.iloc[i, j] < 0.05:
                ax.text(j + 0.5, i + 0.5, "*", color="black",
                        ha="center", va="center", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Generate and save correlation heatmaps for both SPF and OMM12 datasets
plot_correlation(spf_corr, spf_pval, 'Metabolite-Microbiome Correlation (SPF)', save_path='spf_correlation.png')
plot_correlation(omm12_corr, omm12_pval, 'Metabolite-Microbiome Correlation (OMM12)', save_path='omm12_correlation.png')

# =============================================================================
# ---------------- Hierarchical Clustering (Dendrograms) ------------------------
# =============================================================================

def plot_dendrogram(data, title, save_path=None):
    """
    Plot a dendrogram using hierarchical clustering on the features.
    This helps visualize how features cluster together based on their patterns
    across different groups.
    
    Parameters:
        data (DataFrame): Data with groups as rows and features as columns.
        title (str): Title for the dendrogram.
        save_path (str): Optional file path to save the figure.
    """
    # Perform hierarchical clustering using Ward's method
    linked = linkage(data.T, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(linked, labels=list(data.columns), leaf_rotation=90)
    plt.title(title, fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Plot dendrograms for both metabolite and microbe fold-change data (SPF and OMM12)
plot_dendrogram(spf_met_fc, "Hierarchical Clustering of Metabolites (SPF)", save_path="dendrogram_metabolites_spf.png")
plot_dendrogram(spf_micro_fc, "Hierarchical Clustering of Microbial Genera (SPF)", save_path="dendrogram_microbes_spf.png")
plot_dendrogram(omm12_met_fc, "Hierarchical Clustering of Metabolites (OMM12)", save_path="dendrogram_metabolites_omm12.png")
plot_dendrogram(omm12_micro_fc, "Hierarchical Clustering of Microbial Genera (OMM12)", save_path="dendrogram_microbes_omm12.png")

# =============================================================================
# ------------------- Correlation Network Analysis ----------------------------
# =============================================================================

def plot_correlation_network(corr_df, pval_df, corr_threshold=0.3, pval_threshold=0.05, title="Correlation Network", save_path=None):
    """
    Build and visualize a bipartite network connecting metabolites and microbial genera.
    An edge is drawn between a metabolite and a microbe if the absolute correlation
    is above the threshold and statistically significant.
    
    Parameters:
        corr_df (DataFrame): Correlation coefficients.
        pval_df (DataFrame): Corresponding p-values.
        corr_threshold (float): Minimum absolute correlation to consider.
        pval_threshold (float): Maximum p-value to consider significance.
        title (str): Title for the network plot.
        save_path (str): Optional file path to save the figure.
    """
    G = nx.Graph()
    
    # Add microbe nodes with a prefix for clarity
    for micro in corr_df.index:
        G.add_node("M:" + micro, type="microbe")
    # Add metabolite nodes with a prefix for clarity
    for met in corr_df.columns:
        G.add_node("Met:" + met, type="metabolite")
        
    # Add edges for significant correlations
    for micro in corr_df.index:
        for met in corr_df.columns:
            corr = corr_df.loc[micro, met]
            p_val = pval_df.loc[micro, met]
            if abs(corr) >= corr_threshold and p_val < pval_threshold:
                G.add_edge("M:" + micro, "Met:" + met, weight=corr)
                
    # Generate positions for nodes using a spring layout for better readability
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(14, 12))
    
    # Separate nodes by type for custom styling
    micro_nodes = [n for n, d in G.nodes(data=True) if d["type"]=="microbe"]
    met_nodes = [n for n, d in G.nodes(data=True) if d["type"]=="metabolite"]
    nx.draw_networkx_nodes(G, pos, nodelist=micro_nodes, node_color='skyblue', node_shape='o', node_size=600)
    nx.draw_networkx_nodes(G, pos, nodelist=met_nodes, node_color='salmon', node_shape='^', node_size=600)
    
    # Draw edges with color and width scaled by the correlation value
    edges = G.edges(data=True)
    edge_colors = ['#1f78b4' if data['weight'] > 0 else '#e31a1c' for _, _, data in edges]
    edge_widths = [abs(data['weight'])*4 for _, _, data in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    # Add node labels with improved font settings
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    plt.title(title, fontsize=18)
    plt.axis('off')
    
    # Create a custom legend for node types
    micro_patch = mpatches.Patch(color='skyblue', label='Microbe')
    met_patch = mpatches.Patch(color='salmon', label='Metabolite')
    plt.legend(handles=[micro_patch, met_patch], loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Plot correlation networks for both SPF and OMM12 datasets
plot_correlation_network(spf_corr, spf_pval, corr_threshold=0.3, pval_threshold=0.05,
                         title="Correlation Network (SPF)", save_path="correlation_network_spf.png")
plot_correlation_network(omm12_corr, omm12_pval, corr_threshold=0.3, pval_threshold=0.05,
                         title="Correlation Network (OMM12)", save_path="correlation_network_omm12.png")
