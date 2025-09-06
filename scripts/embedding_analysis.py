# %% [markdown]
# Embedding Similarity Analysis
#
# This notebook analyzes similarities in the embeddings dataset in three main ways:
# 1. **Scenario-wise analysis**: Compare different models' responses for the same scenario/dilemma
# 2. **Model-wise analysis**: Compare different scenarios/dilemmas for the same model
# 3. **Reason-wise analysis**: Compare different reasoning versions for the same model in the same scenario
#
# The dataset contains embeddings from multiple models (GPT-3.5, GPT-4, Claude, Bison, Gemma, Mistral, Llama)
# responding to normative evaluation scenarios.


# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# %% Load and explore the embeddings data
def load_embeddings(embeddings_file: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load embeddings from CSV and organize them by column."""
    print("Loading embeddings...")
    df = pd.read_csv(embeddings_file)
    print(f"Dataset shape: {df.shape}")

    embedding_cols = [col for col in df.columns if col.endswith("_embedding")]
    print(f"Embedding columns:")
    for col in embedding_cols:
        print(f"  - {col}")

    embeddings_dict = {}  # Will store {column_name: numpy_array_of_all_rows}
    for col in tqdm(embedding_cols, desc="Processing embedding columns"):

        def parse_embedding(x):
            """Convert string representation to numpy array."""
            if pd.isna(x):
                # Fill with zeros if missing
                return np.zeros(768, dtype=np.float32)
            # Remove brackets and split by spaces to get float values
            return np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32)

        embeddings = df[col].apply(parse_embedding).values

        # Stack all individual embeddings into a 2D array (rows x dimensions)
        embeddings_array = np.vstack(embeddings)
        embeddings_dict[col] = embeddings_array

    return df, embeddings_dict


df, embeddings_dict = load_embeddings("../data/embeddings_sentence_transformers.csv")

# %% Identify models and reason types
models = set()
reason_types = set()

for col in embeddings_dict.keys():
    if col in ["selftext_embedding", "top_comment_embedding"]:
        continue

    parts = col.replace("_embedding", "").split("_")

    model = parts[0]
    reason = "_".join(parts[1:])

    models.add(model)
    reason_types.add(reason)

models = sorted(list(models))
reason_types = sorted(list(reason_types))

# %% Create results directory
results_dir = Path("../results/similarities")
results_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Row-wise Analysis: Model Comparison for Same Scenarios
#
# This analysis compares how different models respond to the same ethical dilemma.
# For each row (scenario), we calculate similarities between all model pairs.


# %% Row-wise similarity analysis
def analyze_row_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    models: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze inter-model agreement on the same ethical scenarios (row-wise analysis).

    This function compares how different AI models respond to identical ethical dilemmas.
    For each scenario (row) in the dataset, it calculates similarities between all possible
    model pairs by comparing their reasoning embeddings.

    The analysis accounts for models having different numbers of reasoning approaches
    (reason_1, reason_2, etc.) by comparing all available combinations and taking the
    mean similarity.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        models: List of model names to analyze
        reason_types: List of available reasoning types (e.g., ['reason_1', 'reason_2'])

    Returns:
        Dictionary with structure:
        {
            scenario_id: {
                'model1_vs_model2': mean_similarity_score,
                'model1_vs_model3': mean_similarity_score,
                ...
            }
        }

    Example:
        For 100th scenario comparing GPT-4 (has reason_1, reason_2) vs Claude (has reason_1):
        - Calculates: gpt4_reason_1[100] vs claude_reason_1[100]
        - Calculates: gpt4_reason_2[100] vs claude_reason_1[100]
        - Returns: mean of both similarities as final 'gpt4_vs_claude' score
    """

    # Find which reason types are available for each model
    model_reason_combinations = {}
    for model in models:
        available_reasons = []
        for reason_type in reason_types:
            col_name = f"{model}_{reason_type}_embedding"
            if col_name in embeddings_dict:
                available_reasons.append(reason_type)
        if available_reasons:  # Only include models that have at least one reason type
            model_reason_combinations[model] = available_reasons

    # Get number of rows from any available embedding
    first_embedding = next(iter(embeddings_dict.values()))
    n_rows = first_embedding.shape[0]
    row_similarities = {}

    for i in tqdm(range(n_rows), desc="Processing rows"):
        row_sims = {}
        model_names = list(model_reason_combinations.keys())

        for j, model1 in enumerate(model_names):
            for model2 in model_names[j + 1 :]:
                # Get all available reason types for both models
                model1_reasons = model_reason_combinations[model1]
                model2_reasons = model_reason_combinations[model2]

                # Calculate similarities between all combinations
                pair_similarities = []
                for reason1 in model1_reasons:
                    for reason2 in model2_reasons:
                        col_name1 = f"{model1}_{reason1}_embedding"
                        col_name2 = f"{model2}_{reason2}_embedding"

                        # Get embeddings for this row
                        embedding1 = embeddings_dict[col_name1][i].reshape(1, -1)
                        embedding2 = embeddings_dict[col_name2][i].reshape(1, -1)

                        # Normalize embeddings
                        embedding1_norm = normalize(embedding1, norm="l2")
                        embedding2_norm = normalize(embedding2, norm="l2")

                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            embedding1_norm, embedding2_norm
                        )[0, 0]
                        pair_similarities.append(similarity)

                # Take mean of all reason combinations for this model pair
                mean_similarity = np.mean(pair_similarities)
                row_sims[f"{model1}_vs_{model2}"] = float(mean_similarity)

        row_similarities[i] = row_sims

    return row_similarities


# Analyze with all available reason types
row_similarities = analyze_row_similarities(embeddings_dict, models, reason_types)


# %% Visualize row-wise similarities
def plot_row_similarity_distribution(row_similarities: Dict, save_path: str = None):
    """Plot distribution of similarities across rows.

    Shows mean similarities across all available reason type combinations
    for each model pair.
    """

    pair_similarities = {}
    for row_data in row_similarities.values():
        for pair, sim in row_data.items():
            if pair not in pair_similarities:
                pair_similarities[pair] = []
            pair_similarities[pair].append(sim)

    n_pairs = len(pair_similarities)
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    for i, (pair, similarities) in enumerate(pair_similarities.items()):
        if i < len(axes):
            axes[i].hist(similarities, bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(
                f'{pair.replace("_vs_", " vs ")}\nMean: {np.mean(similarities):.3f}'
            )
            axes[i].set_xlabel("Cosine Similarity")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

    for i in range(len(pair_similarities), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        f"Distribution of Row-wise Similarities Between Models\n({n_pairs} pairs)",
        y=1.02,
        fontsize=16,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_row_similarity_distribution(
    row_similarities, results_dir / "row_similarities_distribution.png"
)


# %% Statistical summary of row-wise similarities
def summarize_row_characteristics(row_similarities: Dict):
    """Provide statistical summary of inter-model agreement patterns."""

    print(f"=== ROW-WISE SIMILARITY SUMMARY ===\n")

    pair_stats = {}
    for row_data in row_similarities.values():
        for pair, similarity in row_data.items():
            if pair not in pair_stats:
                pair_stats[pair] = []
            pair_stats[pair].append(similarity)

    summary_data = []
    for pair, similarities in pair_stats.items():
        model1, model2 = pair.split("_vs_")
        summary_data.append(
            {
                "Model_1": model1,
                "Model_2": model2,
                "Mean_Similarity": np.mean(similarities),
                "Std_Similarity": np.std(similarities),
                "Min_Similarity": np.min(similarities),
                "Max_Similarity": np.max(similarities),
                "Q25": np.percentile(similarities, 25),
                "Q75": np.percentile(similarities, 75),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values(["Model_1", "Model_2"])
    print(summary_df.round(4))

    all_similarities = []
    for similarities in pair_stats.values():
        all_similarities.extend(similarities)

    print(f"\n=== OVERALL INTER-MODEL AGREEMENT ===")
    print(f"Mean agreement across all model pairs: {np.mean(all_similarities):.4f}")
    print(f"Standard deviation: {np.std(all_similarities):.4f}")
    print(f"Range: {np.min(all_similarities):.4f} - {np.max(all_similarities):.4f}")

    return summary_df


row_summary = summarize_row_characteristics(row_similarities)

# %% [markdown]
# ## 2. Column-wise Analysis: Scenario Comparison for Same Models
#
# This analysis compares how similar different scenarios are when processed by the same model.
# We'll calculate intra-model similarities across all scenarios.


# %% Column-wise similarity analysis
def analyze_column_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    models: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-model consistency across different ethical scenarios (column-wise analysis).

    This function measures how consistent each AI model is when responding to different
    ethical dilemmas. It calculates the similarity between all pairs of scenarios for
    each model.

    The analysis averages all available reasoning approaches (reason_1, reason_2, etc.)
    per scenario, then computes similarities between these aggregated representations
    across all scenario pairs.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        models: List of model names to analyze
        reason_types: List of available reasoning types to aggregate

    Returns:
        Dictionary with structure:
        {
            'model_name': {
                'similarities': numpy_array_of_all_pairwise_similarities,
                'mean_similarity': average_consistency_score,
                'std_similarity': variability_in_consistency
            }
        }

    1. For each model, aggregate all reasoning types per scenario (mean embedding)
    2. Compute cosine similarity matrix between all aggregated scenario pairs
    3. Extract upper triangle to get unique pairwise similarities
    4. Calculate statistics on the resulting similarity distribution
    """

    model_similarities = {}

    for model in models:
        # Find which reason types are available for this model
        available_reasons = []
        model_embeddings = {}
        for reason_type in reason_types:
            col_name = f"{model}_{reason_type}_embedding"
            if col_name in embeddings_dict:
                available_reasons.append(reason_type)
                model_embeddings[reason_type] = embeddings_dict[col_name]

        if not available_reasons:  # Skip if no reason types available
            continue

        # Stack all reason embeddings for this model
        all_reason_embeddings = []
        for reason in available_reasons:
            embeddings = model_embeddings[reason]
            all_reason_embeddings.append(embeddings)

        # Calculate mean across reason types for each row
        mean_embeddings = np.mean(all_reason_embeddings, axis=0)

        # Normalize embeddings
        mean_embeddings_norm = normalize(mean_embeddings, norm="l2")

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(mean_embeddings_norm)

        # Extract upper triangle to get unique pairs
        upper_triangle = np.triu(similarity_matrix, k=1)
        similarities = upper_triangle[upper_triangle != 0]

        model_similarities[model] = {
            "similarities": similarities,
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
        }

    return model_similarities


column_similarities = analyze_column_similarities(embeddings_dict, models, reason_types)


# %% Visualize column-wise similarities
def plot_column_similarity_comparison(column_similarities: Dict, save_path: str = None):
    """Compare intra-model similarity distributions."""

    model_names = list(column_similarities.keys())
    n_models = len(model_names)

    height_ratios = [2.5] * n_models + [3]

    fig, axes = plt.subplots(
        n_models + 1,
        1,
        figsize=(12, 2.5 * n_models + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )
    colors = sns.color_palette("husl", n_models)

    for i, (model, data) in enumerate(column_similarities.items()):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=50,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            density=True,
        )

        mean_sim = data["mean_similarity"]
        ax.set_title(f"{model.upper()}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        ax.axvline(
            mean_sim,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Mean: {mean_sim:.3f}",
        )
        ax.legend()

    ax_box = axes[-1]
    similarities_data = [data["similarities"] for data in column_similarities.values()]

    box_plot = ax_box.boxplot(similarities_data, labels=model_names, patch_artist=True)

    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_xlabel("Model")
    ax_box.set_ylabel("Cosine Similarity")
    ax_box.set_title("Intra-Model Similarity Comparison")
    ax_box.tick_params(axis="x", rotation=45)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_column_similarity_comparison(
    column_similarities, results_dir / "column_similarities_comparison.png"
)


# %% Statistical summary of model differences
def summarize_column_characteristics(column_similarities: Dict):
    """Provide statistical summary of each model's characteristics."""

    print(f"=== COLUMN-WISE SIMILARITY SUMMARY ===\n")

    summary_data = []
    for model, data in column_similarities.items():
        summary_data.append(
            {
                "Model": model,
                "Mean Similarity": data["mean_similarity"],
                "Std Similarity": data["std_similarity"],
                "Min Similarity": np.min(data["similarities"]),
                "Max Similarity": np.max(data["similarities"]),
                "Q25": np.percentile(data["similarities"], 25),
                "Q75": np.percentile(data["similarities"], 75),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values("Model")

    print(summary_df.round(4))

    return summary_df


model_summary = summarize_column_characteristics(column_similarities)

# %% [markdown]
# ## 3. Reason-wise Analysis: Different Reasoning Approaches for Same Model-Scenario
#
# This analysis compares how consistent each model is across its different reasoning
# approaches for the same ethical scenarios.


# %% Reason-wise similarity analysis
def analyze_reason_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    models: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-model reasoning consistency across different reasoning approaches (reason-wise analysis).

    This function examines how consistent each AI model is when applying different reasoning
    approaches to the same ethical scenario. It measures the similarity between a model's
    various reasoning types (reason_1, reason_2, etc.) when confronted with identical
    ethical dilemmas.

    For each model-scenario combination, the function compares all available reasoning
    approaches pairwise and aggregates the results across all scenarios.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        models: List of model names to analyze
        reason_types: List of available reasoning types to compare

    Returns:
        Dictionary with structure:
        {
            'model_name': {
                'similarities': numpy_array_of_all_reason_pair_similarities,
                'mean_similarity': average_reasoning_consistency,
                'std_similarity': variability_in_reasoning_consistency,
                'available_reasons': list_of_reasoning_types_for_model
            }
        }

    1. For each model, identify available reasoning approaches
    2. For each scenario, compare all unique reasoning type pairs
    3. Aggregate similarity scores across all scenarios and reason pairs
    4. Calculate statistics on the resulting consistency distribution
    """

    reason_similarities = {}

    for model in tqdm(models, desc="Processing models"):
        # Find which reason types are available for this model
        available_reasons = []
        model_embeddings = {}
        for reason_type in reason_types:
            col_name = f"{model}_{reason_type}_embedding"
            if col_name in embeddings_dict:
                available_reasons.append(reason_type)
                model_embeddings[reason_type] = embeddings_dict[col_name]

        n_rows = model_embeddings[available_reasons[0]].shape[0]

        # Calculate similarities between all reason pairs for each scenario
        all_similarities = []

        for i, reason1 in enumerate(available_reasons):
            for reason2 in available_reasons[i + 1 :]:  # Only unique pairs
                pair_similarities = []
                for row_idx in range(n_rows):
                    # Get embeddings for this scenario
                    embedding1 = model_embeddings[reason1][row_idx].reshape(1, -1)
                    embedding2 = model_embeddings[reason2][row_idx].reshape(1, -1)

                    # Normalize embeddings
                    embedding1_norm = normalize(embedding1, norm="l2")
                    embedding2_norm = normalize(embedding2, norm="l2")

                    # Calculate cosine similarity
                    similarity = cosine_similarity(embedding1_norm, embedding2_norm)[
                        0, 0
                    ]
                    pair_similarities.append(similarity)

                all_similarities.extend(pair_similarities)

        reason_similarities[model] = {
            "similarities": np.array(all_similarities),
            "mean_similarity": np.mean(all_similarities),
            "std_similarity": np.std(all_similarities),
            "available_reasons": available_reasons,
        }

    return reason_similarities


reason_similarities = analyze_reason_similarities(embeddings_dict, models, reason_types)


# %% Visualize reason-wise similarities
def plot_reason_similarity_comparison(reason_similarities: Dict, save_path: str = None):
    """Compare reason-wise similarity distributions with separate subplots for each model."""

    model_names = list(reason_similarities.keys())
    n_models = len(model_names)

    height_ratios = [2.5] * n_models + [3]

    fig, axes = plt.subplots(
        n_models + 1,
        1,
        figsize=(12, 2.5 * n_models + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    if n_models == 1:
        axes = [axes[0], axes[1]]

    colors = sns.color_palette("husl", n_models)

    for i, (model, data) in enumerate(reason_similarities.items()):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=50,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            density=True,
        )

        mean_sim = data["mean_similarity"]
        n_reasons = len(data["available_reasons"])
        ax.set_title(f"{model.upper()} ({n_reasons} reasonings)")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        ax.axvline(
            mean_sim,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Mean: {mean_sim:.3f}",
        )
        ax.legend()

    ax_box = axes[-1]
    similarities_data = [data["similarities"] for data in reason_similarities.values()]

    box_plot = ax_box.boxplot(similarities_data, labels=model_names, patch_artist=True)

    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_xlabel("Model")
    ax_box.set_ylabel("Cosine Similarity")
    ax_box.set_title("Reason-wise Similarity Comparison")
    ax_box.tick_params(axis="x", rotation=45)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_reason_similarity_comparison(
    reason_similarities, results_dir / "reason_similarities_comparison.png"
)


# %% Statistical summary of reason-wise characteristics
def summarize_reason_characteristics(reason_similarities: Dict):
    """Provide statistical summary of each model's reason-wise characteristics."""

    if not reason_similarities:
        print("No reason-wise data available for analysis")
        return pd.DataFrame()

    print(f"=== REASON-WISE SIMILARITY SUMMARY ===\n")

    summary_data = []
    for model, data in reason_similarities.items():
        summary_data.append(
            {
                "Model": model,
                "Mean Similarity": data["mean_similarity"],
                "Std Similarity": data["std_similarity"],
                "Min Similarity": np.min(data["similarities"]),
                "Max Similarity": np.max(data["similarities"]),
                "Q25": np.percentile(data["similarities"], 25),
                "Q75": np.percentile(data["similarities"], 75),
                "Num Reasons": len(data["available_reasons"]),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values("Model")

    print(summary_df.round(4))

    return summary_df


reason_summary = summarize_reason_characteristics(reason_similarities)


# %% Cross-analysis: Model consistency vs. diversity
def analyze_model_consistency_vs_diversity(
    row_similarities: Dict, column_similarities: Dict, reason_similarities: Dict
):
    """Analyze the relationship between inter-model agreement and intra-model diversity."""

    # Calculate inter-model similarity for each model
    inter_model_means = {}
    for model in column_similarities.keys():
        # Find all pairs that include this model
        model_pairs = [
            pair for pair in list(row_similarities.values())[0].keys() if model in pair
        ]
        all_similarities = []

        # Collect similarities across all scenarios for this model's pairs
        for row_data in row_similarities.values():
            for pair in model_pairs:
                if pair in row_data:
                    all_similarities.append(row_data[pair])

        # Calculate mean inter-model similarity across all scenarios and pairs
        if all_similarities:
            inter_model_means[model] = np.mean(all_similarities)

    # Create comparison dataset combining inter-model and intra-model metrics
    comparison_data = []
    for model in column_similarities.keys():
        if model in inter_model_means:
            comparison_data.append(
                {
                    "Model": model,
                    "Intra-Model_Diversity_Score": 1
                    - column_similarities[model]["mean_similarity"],
                    "Inter-Model_Consistency_Score": inter_model_means[model],
                    "Reason_Consistency_Score": reason_similarities[model][
                        "mean_similarity"
                    ],
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    plt.figure(figsize=(12, 8))

    plot_df = comparison_df.dropna(subset=["Reason_Consistency_Score"])
    scatter = plt.scatter(
        plot_df["Intra-Model_Diversity_Score"],
        plot_df["Inter-Model_Consistency_Score"],
        s=120,
        alpha=0.8,
        c=plot_df["Reason_Consistency_Score"],
        cmap="RdYlBu_r",
        edgecolors="black",
        linewidth=0.5,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Reason-wise Consistency Score", rotation=270, labelpad=20)

    for i, row in plot_df.iterrows():
        plt.annotate(
            row["Model"],
            (row["Intra-Model_Diversity_Score"], row["Inter-Model_Consistency_Score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.xlabel("Intra-Model Diversity Score (1 - Intra-Model Similarity)")
    plt.ylabel("Inter-Model Consistency Score (Inter-Model Similarity)")

    title = "Model Diversity vs. Consistency Analysis (Color = Reason-wise Consistency)"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "diversity_vs_consistency.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("=== COMPLETE ANALYSIS RESULTS ===")
    display_cols = [
        "Model",
        "Intra-Model_Diversity_Score",
        "Inter-Model_Consistency_Score",
        "Reason_Consistency_Score",
    ]
    print(comparison_df[display_cols].round(4))

    return comparison_df


consistency_analysis = analyze_model_consistency_vs_diversity(
    row_similarities, column_similarities, reason_similarities
)

# %% [markdown]
# ## Summary of Findings
#
# This comprehensive analysis examines embedding similarities across all available reasoning types (reason_1 through reason_5) for 7 AI models on 10,806 ethical scenarios, revealing some behavioral patterns in AI ethical decision-making.
#
# ### Key Findings:
#
# 1. **Inter-Model Agreement** (Mean: 69.4% ± 5.4%):
#    - Models show reasonable consensus across ethical scenarios (range: 48.5% - 85.7%)
#    - **Claude** shows highest agreement with other models (73.9% consistency score)
#    - **Bison** shows lowest agreement with other models (63.7% consistency score)
#    - Agreement varies significantly by scenario, suggesting some ethical dilemmas create more consensus than others
#    - 90% of scenarios fall between 62.3% - 76.2% inter-model agreement, showing generally stable but varied consensus
#
# 2. **Intra-Model Agreement** (Range: 28.5% - 49.8%):
#    - **Gemma** is most internally consistent (49.8% ± 10.1%) - most predictable responses
#    - **Bison** is least internally consistent (28.5% ± 12.1%) - most unpredictable responses
#    - Internal consistency range of 21.3% indicates significant diversity in model architectures and training
#
# 3. **Reason-wise Consistency** (Range: 72.9% - 90.6%):
#    - **Claude** shows highest reasoning coherence (90.6% ± 5.6%) - most consistent across different reasoning approaches
#    - **Mistral** shows lowest reasoning coherence (72.9% ± 11.7%) - most variable across reasoning approaches
#    - Mean reason-wise consistency (80.7% ± 6.1%) much higher than intra-model consistency, indicating models are more consistent within reasoning types than across scenarios
#
# 4. **Three-Dimensional Model Profiles**:
#    - **Claude**: High inter-model agreement (73.9%), moderate intra-model consistency (43.1%), highest reasoning coherence (90.6%)
#    - **Gemma**: Moderate inter-model agreement (69.3%), highest intra-model consistency (49.8%), moderate reasoning coherence (76.4%)
#    - **GPT-4**: Moderate inter-model agreement (67.9%), lowest intra-model consistency (31.9%), moderate reasoning coherence (76.7%)
#    - **Bison**: Lowest inter-model agreement (63.7%), lowest intra-model consistency (28.5%), high reasoning coherence (82.4%)
#
# ### Practical Implications:
#
# - **Most Predictable Ethics**: Gemma (highest internal consistency across scenarios)
# - **Most Coherent Reasoning**: Claude (most consistent across different reasoning approaches)
# - **Most Diverse Perspectives**: GPT-4 and Bison (high variability in responses)
# - **Best Overall Balance**: Claude (reliable consensus + coherent reasoning + moderate diversity)

# %% [markdown]
# ## 4. Human vs LLM Analysis: Comparing AI Reasoning with Human Responses
#
# This analysis compares how well different LLMs align with human moral reasoning
# by calculating similarities between LLM-generated reasoning and Reddit top comments
# (human responses) for the same ethical scenarios.


# %% Human-LLM similarity analysis
def calculate_model_human_similarity(
    model_name: str, scenario_idx: int, embeddings_dict: Dict[str, np.ndarray]
) -> float:
    """Calculate similarity between model's average reasoning and human response for a specific scenario.

    Args:
        model_name: Name of the model (e.g., 'gpt3.5', 'claude')
        scenario_idx: Index of the scenario in the dataset
        embeddings_dict: Dictionary containing all embedding arrays

    Returns:
        Cosine similarity score between model's average reasoning and human response
    """
    # Find all available reasoning types for this model
    available_reasons = []
    for col in embeddings_dict.keys():
        if col.startswith(f"{model_name}_reason_") and col.endswith("_embedding"):
            available_reasons.append(col)

    if not available_reasons:
        return None

    # Average all reasoning embeddings for this model-scenario
    model_embeddings = []
    for reason_col in available_reasons:
        model_embeddings.append(embeddings_dict[reason_col][scenario_idx])

    model_avg_embedding = np.mean(model_embeddings, axis=0).reshape(1, -1)
    human_embedding = embeddings_dict["top_comment_embedding"][scenario_idx].reshape(
        1, -1
    )

    # Normalize and calculate similarity
    model_norm = normalize(model_avg_embedding, norm="l2")
    human_norm = normalize(human_embedding, norm="l2")

    return cosine_similarity(model_norm, human_norm)[0, 0]


def analyze_human_llm_similarities(
    embeddings_dict: Dict[str, np.ndarray], models: List[str]
) -> Dict[str, np.ndarray]:
    """Analyze similarities between each LLM's reasoning and human responses.

    For each model, calculates the average similarity between the model's reasoning
    (averaged across all available reasoning approaches) and human responses across
    all scenarios.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        models: List of model names to analyze

    Returns:
        Dictionary with structure:
        {
            'model_name': numpy_array_of_similarities_with_humans
        }
    """
    print("Analyzing human-LLM similarities...")

    model_human_similarities = {}
    n_scenarios = embeddings_dict["top_comment_embedding"].shape[0]

    for model in models:
        similarities = []

        for scenario_idx in tqdm(
            range(n_scenarios), desc=f"Processing {model}", leave=False
        ):
            sim = calculate_model_human_similarity(model, scenario_idx, embeddings_dict)
            if sim is not None:
                similarities.append(sim)

        model_human_similarities[model] = np.array(similarities)
        print(
            f"{model}: {len(similarities)} scenarios, mean similarity: {np.mean(similarities):.4f}"
        )

    return model_human_similarities


# Run the human-LLM similarity analysis
human_llm_similarities = analyze_human_llm_similarities(embeddings_dict, models)


# %% Visualize human-LLM similarities
def plot_human_llm_similarity_comparison(
    human_llm_similarities: Dict[str, np.ndarray], save_path: str = None
):
    """Compare human-LLM similarity distributions across models."""

    model_names = list(human_llm_similarities.keys())
    n_models = len(model_names)

    # Create figure with separate subplots for each model + box plot
    height_ratios = [2.5] * n_models + [3]

    fig, axes = plt.subplots(
        n_models + 1,
        1,
        figsize=(12, 2.5 * n_models + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    if n_models == 1:
        axes = [axes[0], axes[1]]

    colors = sns.color_palette("husl", n_models)

    # Plot individual distributions for each model
    for i, (model, similarities) in enumerate(human_llm_similarities.items()):
        ax = axes[i]
        ax.hist(
            similarities,
            bins=50,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            density=True,
        )

        mean_sim = np.mean(similarities)
        ax.set_title(f"{model.upper()}")
        ax.set_xlabel("Cosine Similarity with Human Responses")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        ax.axvline(
            mean_sim,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Mean: {mean_sim:.3f}",
        )
        ax.legend()

    # Box plot comparison at the bottom
    ax_box = axes[-1]
    similarities_data = list(human_llm_similarities.values())

    box_plot = ax_box.boxplot(similarities_data, labels=model_names, patch_artist=True)

    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_xlabel("Model")
    ax_box.set_ylabel("Cosine Similarity with Human Responses")
    ax_box.set_title("Human-LLM Similarity Comparison")
    ax_box.tick_params(axis="x", rotation=45)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_human_llm_similarity_comparison(
    human_llm_similarities, results_dir / "human_llm_similarities_comparison.png"
)


# %% Statistical summary of human-LLM similarities
def summarize_human_llm_characteristics(human_llm_similarities: Dict[str, np.ndarray]):
    """Provide statistical summary of each model's alignment with human responses."""

    print(f"=== HUMAN-LLM SIMILARITY SUMMARY ===\n")

    summary_data = []
    for model, similarities in human_llm_similarities.items():
        summary_data.append(
            {
                "Model": model,
                "Mean_Human_Similarity": np.mean(similarities),
                "Std_Human_Similarity": np.std(similarities),
                "Min_Human_Similarity": np.min(similarities),
                "Max_Human_Similarity": np.max(similarities),
                "Q25": np.percentile(similarities, 25),
                "Q75": np.percentile(similarities, 75),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values(
        "Mean_Human_Similarity", ascending=False
    )

    print(summary_df.round(4))

    # Overall statistics
    all_similarities = []
    for similarities in human_llm_similarities.values():
        all_similarities.extend(similarities)

    print(f"\n=== OVERALL HUMAN-LLM ALIGNMENT ===")
    print(
        f"Mean human-LLM similarity across all models: {np.mean(all_similarities):.4f}"
    )
    print(f"Standard deviation: {np.std(all_similarities):.4f}")
    print(f"Range: {np.min(all_similarities):.4f} - {np.max(all_similarities):.4f}")

    return summary_df


human_llm_summary = summarize_human_llm_characteristics(human_llm_similarities)


# %%
