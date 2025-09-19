# %% [markdown]
# Embedding Similarity Analysis
#
# This notebook analyzes similarities in the embeddings dataset in three main ways:
# 1. **Scenario-wise analysis**: Compare different actors' responses for the same scenario/dilemma
# 2. **Actor-wise analysis**: Compare different scenarios/dilemmas for the same actor
# 3. **Reason-wise analysis**: Compare different reasoning versions for the same actor in the same scenario
#
# The dataset contains embeddings from multiple actors (GPT-3.5, GPT-4, Claude, Bison, Gemma, Mistral, Llama)
# responding to normative evaluation scenarios.
# %% Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle
import json
from pathlib import Path

plt.rcParams["figure.max_open_warning"] = 0
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# %% Load and explore the embeddings data
def load_embeddings(embeddings_file: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load embeddings from CSV and organize them by column."""
    df = pd.read_csv(embeddings_file)

    embedding_cols = [col for col in df.columns if col.endswith("_embedding")]

    embeddings_dict = {}
    for col in tqdm(embedding_cols, desc="Processing embedding columns"):

        def parse_embedding(x):
            """Convert string representation to numpy array."""
            if pd.isna(x):
                return np.zeros(768, dtype=np.float32)
            return np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32)

        embeddings = df[col].apply(parse_embedding).values

        embeddings_array = np.vstack(embeddings)
        embeddings_dict[col] = embeddings_array

    return df, embeddings_dict


df, embeddings_dict = load_embeddings("../data/embeddings_sentence_transformers.csv")

# %% Identify actors and reason types
actors = set()
reason_types = set()

for col in embeddings_dict.keys():
    if col in ["selftext_embedding"]:
        continue

    if col == "top_comment_embedding":
        actors.add("human")
        reason_types.add("top_comment")
        continue

    parts = col.replace("_embedding", "").split("_")

    actor = parts[0]
    reason = "_".join(parts[1:])

    actors.add(actor)
    reason_types.add(reason)

actors = sorted(list(actors))
reason_types = sorted(list(reason_types))

# %% [markdown]
# ## 1. Row-wise Analysis: Actor Comparison for Same Scenarios
#
# This analysis compares how human redditors and LLM models respond to the same ethical dilemma.
# For each row (scenario), we calculate similarities between all actor pairs.


# %% Row-wise similarity analysis
def analyze_row_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze inter-actor agreement on the same ethical scenarios (row-wise analysis).

    This function compares how different LLM models and human redditors respond to identical ethical dilemmas.
    For each scenario (row) in the dataset, it calculates similarities between all possible
    actor pairs by comparing their reasoning embeddings.

    The analysis accounts for actors having different numbers of reasoning approaches
    (reason_1, reason_2, etc.) by comparing all available combinations and taking the
    mean similarity.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        actors: List of actors to analyze (LLM models and human redditors)
        reason_types: List of available reasoning types (e.g., ['reason_1', 'reason_2'])

    Returns:
        Dictionary with structure:
        {
            scenario_id: {
                'actor1_vs_actor2': mean_similarity_score,
                'actor1_vs_actor3': mean_similarity_score,
                ...
            }
        }

    Example:
        For 100th scenario comparing GPT-4 (has reason_1, reason_2) vs Claude (has reason_1):
        - Calculates: gpt4_reason_1[100] vs claude_reason_1[100]
        - Calculates: gpt4_reason_2[100] vs claude_reason_1[100]
        - Returns: mean of both similarities as final 'gpt4_vs_claude' score
    """

    actor_reason_combinations = {}
    for actor in actors:
        available_reasons = []
        if actor == "human":
            if "top_comment_embedding" in embeddings_dict:
                available_reasons.append("top_comment")
        else:
            for reason_type in reason_types:
                col_name = f"{actor}_{reason_type}_embedding"
                if col_name in embeddings_dict:
                    available_reasons.append(reason_type)
        if available_reasons:
            actor_reason_combinations[actor] = available_reasons

    first_embedding = next(iter(embeddings_dict.values()))
    n_rows = first_embedding.shape[0]
    row_similarities = {}

    for i in tqdm(range(n_rows), desc="Processing rows"):
        row_sims = {}
        actor_names = list(actor_reason_combinations.keys())

        for j, actor1 in enumerate(actor_names):
            for actor2 in actor_names[j + 1 :]:
                # Get all available reason types for both actors
                actor1_reasons = actor_reason_combinations[actor1]
                actor2_reasons = actor_reason_combinations[actor2]

                # Calculate similarities between all combinations
                pair_similarities = []
                for reason1 in actor1_reasons:
                    for reason2 in actor2_reasons:
                        if actor1 == "human":
                            col_name1 = "top_comment_embedding"
                        else:
                            col_name1 = f"{actor1}_{reason1}_embedding"

                        if actor2 == "human":
                            col_name2 = "top_comment_embedding"
                        else:
                            col_name2 = f"{actor2}_{reason2}_embedding"

                        embedding1 = embeddings_dict[col_name1][i].reshape(1, -1)
                        embedding2 = embeddings_dict[col_name2][i].reshape(1, -1)

                        embedding1_norm = normalize(embedding1, norm="l2")
                        embedding2_norm = normalize(embedding2, norm="l2")

                        similarity = cosine_similarity(
                            embedding1_norm, embedding2_norm
                        )[0, 0]
                        pair_similarities.append(similarity)

                mean_similarity = np.mean(pair_similarities)
                row_sims[f"{actor1}_vs_{actor2}"] = float(mean_similarity)

        row_similarities[i] = row_sims

    return row_similarities


cache_path = Path("../results/row_similarities.pkl")
if cache_path.exists():
    print("Loading row_similarities from cache...")
    with open(cache_path, "rb") as f:
        row_similarities = pickle.load(f)
else:
    row_similarities = analyze_row_similarities(embeddings_dict, actors, reason_types)
    with open(cache_path, "wb") as f:
        pickle.dump(row_similarities, f)
    print(f"Saved row_similarities to {cache_path}")


# %% Visualize row-wise similarities
def plot_row_similarity_distribution(row_similarities: Dict):
    """Plot distribution of similarities across rows.

    Shows mean similarities across all available reason type combinations
    for each actor pair.
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

    for i, (pair, similarities) in enumerate(
        tqdm(pair_similarities.items(), desc="Plotting row similarities")
    ):
        if i < len(axes):
            axes[i].hist(
                similarities,
                bins=20,
                alpha=0.7,
                edgecolor="black",
                color="skyblue",
                density=True,
            )

            mean_sim = np.mean(similarities)
            median_sim = np.median(similarities)

            axes[i].axvline(
                mean_sim,
                color="red",
                linestyle="--",
                alpha=0.8,
                label=f"Mean: {mean_sim:.3f}",
            )
            axes[i].axvline(
                median_sim,
                color="blue",
                linestyle="-",
                alpha=0.8,
                label=f"Median: {median_sim:.3f}",
            )

            axes[i].set_title(
                f'{pair.replace("_vs_", " vs ")}\nMean: {mean_sim:.3f}, Median: {median_sim:.3f}'
            )
            axes[i].set_xlabel("Cosine Similarity")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

    for i in range(len(pair_similarities), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        f"Distribution of Row-wise Similarities Between Actors\n({n_pairs} pairs)",
        y=1.02,
        fontsize=16,
    )
    plt.show()


plot_row_similarity_distribution(row_similarities)


# %% Statistical summary of row-wise similarities
def summarize_row_characteristics(row_similarities: Dict):
    """Provide statistical summary of inter-actor agreement patterns."""

    print(f"=== ROW-WISE SIMILARITY SUMMARY ===\n")

    pair_stats = {}
    for row_data in row_similarities.values():
        for pair, similarity in row_data.items():
            if pair not in pair_stats:
                pair_stats[pair] = []
            pair_stats[pair].append(similarity)

    summary_data = []
    for pair, similarities in pair_stats.items():
        actor1, actor2 = pair.split("_vs_")
        summary_data.append(
            {
                "Actor_1": actor1,
                "Actor_2": actor2,
                "Mean_Similarity": np.mean(similarities),
                "Std_Similarity": np.std(similarities),
                "Min_Similarity": np.min(similarities),
                "Max_Similarity": np.max(similarities),
                "Q25": np.percentile(similarities, 25),
                "Q75": np.percentile(similarities, 75),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values(["Actor_1", "Actor_2"])
    print(summary_df.round(4))

    return summary_df


row_summary_df = summarize_row_characteristics(row_similarities)

# %% Save row-wise analysis results
# Create results directory if it doesn't exist
results_dir = Path("../results")
results_dir.mkdir(exist_ok=True)

# Save row-wise analysis results as JSON
row_summary_dict = row_summary_df.to_dict("records")
with open(results_dir / "row_wise_analysis_results.json", "w") as f:
    json.dump(row_summary_dict, f, indent=2)
print(
    f"Row-wise analysis results saved to {results_dir / 'row_wise_analysis_results.json'}"
)


# %% Display extreme LLM-Human similarity cases
def display_extreme_llm_human_similarities(
    row_similarities: Dict, embeddings_dict: Dict
):
    """Display the top 5 highest and lowest LLM-Human similarity cases with actual sentences."""

    import pandas as pd

    df_cleaned = pd.read_csv(
        "../data/normative_evaluation_everyday_dilemmas_dataset_cleaned.csv"
    )

    # Extract all LLM-Human similarity scores
    llm_human_similarities = []

    for row_idx, row_data in row_similarities.items():
        for pair, similarity in row_data.items():
            if "human" in pair and pair != "human_vs_human":
                llm_model = pair.replace("_vs_human", "").replace("human_vs_", "")
                llm_human_similarities.append(
                    {
                        "row_idx": row_idx,
                        "llm_model": llm_model,
                        "similarity": similarity,
                        "pair": pair,
                    }
                )

    # Sort by similarity
    llm_human_similarities.sort(key=lambda x: x["similarity"])

    # Get top 5 lowest and highest
    lowest_5 = llm_human_similarities[:5]
    highest_5 = llm_human_similarities[-5:]

    print("=" * 80)
    print("EXTREME LLM-HUMAN SIMILARITY CASES")
    print("=" * 80)

    print("\n🔴 TOP 5 LOWEST SIMILARITY CASES (Most Different)")
    print("-" * 60)

    for i, case in enumerate(lowest_5, 1):
        row_idx = case["row_idx"]
        llm_model = case["llm_model"]
        similarity = case["similarity"]

        # Get scenario info
        scenario = df_cleaned.iloc[row_idx]
        scenario_id = scenario["submission_id"]
        title = (
            scenario["title"][:100] + "..."
            if len(scenario["title"]) > 100
            else scenario["title"]
        )

        # Get human comment
        human_comment = scenario["top_comment"]

        # Get LLM reasoning (try different reason columns)
        llm_reason = None
        for reason_col in [
            f"{llm_model}_reason_1",
            f"{llm_model}_reason_2",
            f"{llm_model}_reason_3",
        ]:
            if reason_col in scenario and pd.notna(scenario[reason_col]):
                llm_reason = scenario[reason_col]
                break

        print(
            f"\n{i}. Similarity: {similarity:.4f} | Row: {row_idx} | Model: {llm_model.upper()}"
        )
        print(f"   Scenario ID: {scenario_id}")
        print(f"   Title: {title}")
        print(
            f"   Human Comment: {human_comment[:200]}{'...' if len(human_comment) > 200 else ''}"
        )
        print(
            f"   {llm_model.upper()} Reasoning: {llm_reason[:200]}{'...' if llm_reason and len(llm_reason) > 200 else ''}"
        )

    print("\n🟢 TOP 5 HIGHEST SIMILARITY CASES (Most Similar)")
    print("-" * 60)

    for i, case in enumerate(highest_5, 1):
        row_idx = case["row_idx"]
        llm_model = case["llm_model"]
        similarity = case["similarity"]

        # Get scenario info
        scenario = df_cleaned.iloc[row_idx]
        scenario_id = scenario["submission_id"]
        title = (
            scenario["title"][:100] + "..."
            if len(scenario["title"]) > 100
            else scenario["title"]
        )

        # Get human comment
        human_comment = scenario["top_comment"]

        # Get LLM reasoning (try different reason columns)
        llm_reason = None
        for reason_col in [
            f"{llm_model}_reason_1",
            f"{llm_model}_reason_2",
            f"{llm_model}_reason_3",
        ]:
            if reason_col in scenario and pd.notna(scenario[reason_col]):
                llm_reason = scenario[reason_col]
                break

        print(
            f"\n{i}. Similarity: {similarity:.4f} | Row: {row_idx} | Model: {llm_model.upper()}"
        )
        print(f"   Scenario ID: {scenario_id}")
        print(f"   Title: {title}")
        print(
            f"   Human Comment: {human_comment[:200]}{'...' if len(human_comment) > 200 else ''}"
        )
        print(
            f"   {llm_model.upper()} Reasoning: {llm_reason[:200]}{'...' if llm_reason and len(llm_reason) > 200 else ''}"
        )

    # Summary of models in extreme cases
    all_extreme_models = [case["llm_model"] for case in lowest_5 + highest_5]
    model_counts = pd.Series(all_extreme_models).value_counts()

    print("\n📊 LLM MODELS IN EXTREME SIMILARITY CASES:")
    print("-" * 40)
    for model, count in model_counts.items():
        print(f"   {model.upper()}: {count} cases")

    return lowest_5, highest_5


# Display extreme cases
lowest_similarities, highest_similarities = display_extreme_llm_human_similarities(
    row_similarities, embeddings_dict
)

# %% [markdown]
# ## 2. Column-wise Analysis: Scenario Comparison for Same Actors
#
# This analysis compares how similar the embeddings ofdifferent scenarios are when processed by the same actor.
# We'll calculate intra-actor similarities across all scenarios.


# %% Column-wise similarity analysis
def analyze_column_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-actor similarity across different ethical scenarios (column-wise analysis).

    This function measures how consistent each actor is when responding to different
    ethical dilemmas. It calculates the similarity between all pairs of scenarios for
    each actor.

    The analysis averages all available reasoning approaches (reason_1, reason_2, etc.)
    per scenario, then computes similarities between these aggregated representations
    across all scenario pairs.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        actors: List of actor names to analyze
        reason_types: List of available reasoning types to aggregate

    Returns:
        Dictionary with structure:
        {
            'actor_name': {
                'similarities': numpy_array_of_all_pairwise_similarities,
                'mean_similarity': average_similarity_score,
                'std_similarity': variability_in_similarity
            }
        }

    1. For each actor, aggregate all reasoning types per scenario (mean embedding)
    2. Compute cosine similarity matrix between all aggregated scenario pairs
    3. Extract upper triangle to get unique pairwise similarities
    4. Calculate statistics on the resulting similarity distribution
    """

    actor_similarities = {}

    for actor in actors:
        available_reasons = []
        actor_embeddings = {}

        if actor == "human":
            if "top_comment_embedding" in embeddings_dict:
                available_reasons.append("top_comment")
                actor_embeddings["top_comment"] = embeddings_dict[
                    "top_comment_embedding"
                ]
        else:
            for reason_type in reason_types:
                col_name = f"{actor}_{reason_type}_embedding"
                if col_name in embeddings_dict:
                    available_reasons.append(reason_type)
                    actor_embeddings[reason_type] = embeddings_dict[col_name]

        if not available_reasons:
            continue

        all_reason_embeddings = []
        for reason in available_reasons:
            embeddings = actor_embeddings[reason]
            all_reason_embeddings.append(embeddings)

        mean_embeddings = np.mean(all_reason_embeddings, axis=0)

        mean_embeddings_norm = normalize(mean_embeddings, norm="l2")

        similarity_matrix = cosine_similarity(mean_embeddings_norm)

        upper_triangle = np.triu(similarity_matrix, k=1)
        similarities = upper_triangle[upper_triangle != 0]

        actor_similarities[actor] = {
            "similarities": similarities,
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
        }

    return actor_similarities


column_similarities = analyze_column_similarities(embeddings_dict, actors, reason_types)


# %% Visualize column-wise similarities
def plot_column_similarity_comparison(column_similarities: Dict):
    """Compare intra-actor similarity distributions."""

    actor_names = list(column_similarities.keys())
    n_actors = len(actor_names)

    height_ratios = [2.5] * n_actors + [3]

    fig, axes = plt.subplots(
        n_actors + 1,
        1,
        figsize=(12, 2.5 * n_actors + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    for i, (actor, data) in enumerate(
        tqdm(column_similarities.items(), desc="Plotting histograms")
    ):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
        )

        mean_sim = data["mean_similarity"]
        median_sim = np.median(data["similarities"])
        ax.set_title(f"{actor.upper()}")
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
        ax.axvline(
            median_sim,
            color="blue",
            linestyle="-",
            alpha=0.8,
            label=f"Median: {median_sim:.3f}",
        )
        ax.set_xlim(0, 1)
        ax.legend()

    ax_box = axes[-1]

    similarities_data = [data["similarities"] for data in column_similarities.values()]
    actor_names = list(column_similarities.keys())

    box_plot = ax_box.boxplot(
        similarities_data, tick_labels=actor_names, patch_artist=True
    )

    for patch in box_plot["boxes"]:
        patch.set_facecolor("skyblue")
        patch.set_alpha(0.8)
    ax_box.set_title("Intra-Actor Similarity Comparison")
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_column_similarity_comparison(column_similarities)


# %% Statistical summary of actor differences
def summarize_column_characteristics(column_similarities: Dict):
    """Provide statistical summary of each actor's characteristics."""

    print(f"=== COLUMN-WISE SIMILARITY SUMMARY ===\n")

    summary_data = []
    for actor, data in column_similarities.items():
        summary_data.append(
            {
                "Actor": actor,
                "Mean Similarity": data["mean_similarity"],
                "Std Similarity": data["std_similarity"],
                "Min Similarity": np.min(data["similarities"]),
                "Max Similarity": np.max(data["similarities"]),
                "Q25": np.percentile(data["similarities"], 25),
                "Q75": np.percentile(data["similarities"], 75),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values("Actor")

    print(summary_df.round(4))

    return summary_df


column_summary_df = summarize_column_characteristics(column_similarities)

# %% Save column-wise analysis results
# Save column-wise analysis results as JSON
column_summary_dict = column_summary_df.to_dict("records")
with open(results_dir / "column_wise_analysis_results.json", "w") as f:
    json.dump(column_summary_dict, f, indent=2)
print(
    f"Column-wise analysis results saved to {results_dir / 'column_wise_analysis_results.json'}"
)

# %% [markdown]
# ## 3. Reason-wise Analysis: Different Reasoning Approaches for Same Actor-Scenario
#
# This analysis compares how consistent each actor is across its different reasoning
# approaches for the same ethical scenarios.


# %% Reason-wise similarity analysis
def analyze_reason_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-actor reasoning consistency across different reasoning approaches (reason-wise analysis).

    This function examines how consistent each actor is when applying different reasoning
    approaches to the same ethical scenario. It measures the similarity between a actor's
    various reasoning types (reason_1, reason_2, etc.) when confronted with identical
    ethical dilemmas.

    For each actor-scenario combination, the function compares all available reasoning
    approaches pairwise and aggregates the results across all scenarios.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays
        actors: List of actor names to analyze
        reason_types: List of available reasoning types to compare

    Returns:
        Dictionary with structure:
        {
            'actor_name': {
                'similarities': numpy_array_of_all_reason_pair_similarities,
                'mean_similarity': average_reasoning_similarity,
                'std_similarity': variability_in_reasoning_consistency,
                'available_reasons': list_of_reasoning_types_for_actor
            }
        }

    1. For each actor, identify available reasoning approaches
    2. For each scenario, compare all unique reasoning type pairs
    3. Aggregate similarity scores across all scenarios and reason pairs
    4. Calculate statistics on the resulting similarity distribution
    """

    reason_similarities = {}

    for actor in tqdm(actors, desc="Processing actors"):
        available_reasons = []
        actor_embeddings = {}

        if actor == "human":
            if "top_comment_embedding" in embeddings_dict:
                available_reasons.append("top_comment")
                actor_embeddings["top_comment"] = embeddings_dict[
                    "top_comment_embedding"
                ]
                # For human actor, reason consistency is 1.0 (perfect consistency since only one comment per scenario)
                reason_similarities[actor] = {
                    "similarities": np.array([1.0]),
                    "mean_similarity": 1.0,
                    "std_similarity": 0.0,
                    "available_reasons": available_reasons,
                }
            continue
        else:
            for reason_type in reason_types:
                col_name = f"{actor}_{reason_type}_embedding"
                if col_name in embeddings_dict:
                    available_reasons.append(reason_type)
                    actor_embeddings[reason_type] = embeddings_dict[col_name]

        if not available_reasons:
            continue

        n_rows = actor_embeddings[available_reasons[0]].shape[0]

        # Calculate similarities between all reason pairs for each scenario
        all_similarities = []

        for i, reason1 in enumerate(available_reasons):
            for reason2 in available_reasons[i + 1 :]:  # Only unique pairs
                pair_similarities = []
                for row_idx in tqdm(
                    range(n_rows), desc=f"Processing {actor} reason pairs", leave=False
                ):
                    embedding1 = actor_embeddings[reason1][row_idx].reshape(1, -1)
                    embedding2 = actor_embeddings[reason2][row_idx].reshape(1, -1)

                    embedding1_norm = normalize(embedding1, norm="l2")
                    embedding2_norm = normalize(embedding2, norm="l2")

                    similarity = cosine_similarity(embedding1_norm, embedding2_norm)[
                        0, 0
                    ]
                    pair_similarities.append(similarity)

                all_similarities.extend(pair_similarities)

        reason_similarities[actor] = {
            "similarities": np.array(all_similarities),
            "mean_similarity": np.mean(all_similarities),
            "std_similarity": np.std(all_similarities),
            "available_reasons": available_reasons,
        }

    return reason_similarities


cache_path = Path("../results/reason_similarities.pkl")
if cache_path.exists():
    print("Loading reason_similarities from cache...")
    with open(cache_path, "rb") as f:
        reason_similarities = pickle.load(f)
else:
    reason_similarities = analyze_reason_similarities(
        embeddings_dict, actors, reason_types
    )
    with open(cache_path, "wb") as f:
        pickle.dump(reason_similarities, f)
    print(f"Saved reason_similarities to {cache_path}")


# %% Visualize reason-wise similarities
def plot_reason_similarity_comparison(reason_similarities: Dict):
    """Compare reason-wise similarity distributions with separate subplots for each actor."""

    actor_names = list(reason_similarities.keys())
    n_actors = len(actor_names)

    height_ratios = [2.5] * n_actors + [3]

    fig, axes = plt.subplots(
        n_actors + 1,
        1,
        figsize=(12, 2.5 * n_actors + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    if n_actors == 1:
        axes = [axes[0], axes[1]]

    for i, (actor, data) in enumerate(
        tqdm(reason_similarities.items(), desc="Plotting reason similarities")
    ):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
        )

        mean_sim = data["mean_similarity"]
        median_sim = np.median(data["similarities"])
        n_reasons = len(data["available_reasons"])
        ax.set_title(f"{actor.upper()} ({n_reasons} reasonings)")
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
        ax.axvline(
            median_sim,
            color="blue",
            linestyle="-",
            alpha=0.8,
            label=f"Median: {median_sim:.3f}",
        )
        ax.set_xlim(0, 1)
        ax.legend()

    ax_box = axes[-1]

    similarities_data = [data["similarities"] for data in reason_similarities.values()]
    actor_names = list(reason_similarities.keys())

    box_plot = ax_box.boxplot(
        similarities_data, tick_labels=actor_names, patch_artist=True
    )

    for patch in box_plot["boxes"]:
        patch.set_facecolor("skyblue")
        patch.set_alpha(0.8)
    ax_box.set_title("Reason-wise Similarity Comparison")
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_reason_similarity_comparison(reason_similarities)


# %% Statistical summary of reason-wise characteristics
def summarize_reason_characteristics(reason_similarities: Dict):
    """Provide statistical summary of each actor's reason-wise characteristics."""

    if not reason_similarities:
        print("No reason-wise data available for analysis")
        return pd.DataFrame()

    print(f"=== REASON-WISE SIMILARITY SUMMARY ===\n")

    summary_data = []
    for actor, data in reason_similarities.items():
        summary_data.append(
            {
                "Actor": actor,
                "Mean Similarity": data["mean_similarity"],
                "Std Similarity": data["std_similarity"],
                "Min Similarity": np.min(data["similarities"]),
                "Max Similarity": np.max(data["similarities"]),
                "Q25": np.percentile(data["similarities"], 25),
                "Q75": np.percentile(data["similarities"], 75),
                "Num Reasons": len(data["available_reasons"]),
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values("Actor")

    print(summary_df.round(4))

    return summary_df


reason_summary_df = summarize_reason_characteristics(reason_similarities)

# %% Save reason-wise analysis results
# Save reason-wise analysis results as JSON
reason_summary_dict = reason_summary_df.to_dict("records")
with open(results_dir / "reason_wise_analysis_results.json", "w") as f:
    json.dump(reason_summary_dict, f, indent=2)
print(
    f"Reason-wise analysis results saved to {results_dir / 'reason_wise_analysis_results.json'}"
)


# %% Cross-analysis: Intra-Actor similarity vs. inter-Actor similarity
def cross_analyze_actor_similarity(
    row_similarities: Dict, column_similarities: Dict, reason_similarities: Dict
):
    """Analyze the relationship between inter-actor similarity and intra-actor similarity."""

    inter_actor_means = {}
    for actor in column_similarities.keys():
        actor_pairs = [
            pair for pair in list(row_similarities.values())[0].keys() if actor in pair
        ]
        all_similarities = []

        for row_data in row_similarities.values():
            for pair in actor_pairs:
                if pair in row_data:
                    all_similarities.append(row_data[pair])

        if all_similarities:
            inter_actor_means[actor] = np.mean(all_similarities)

    comparison_data = []
    for actor in column_similarities.keys():
        if actor in inter_actor_means:
            comparison_data.append(
                {
                    "Actor": actor,
                    "Intra-Actor_Diversity_Score": 1
                    - column_similarities[actor]["mean_similarity"],
                    "Inter-Actor_Similarity_Score": inter_actor_means[actor],
                    "Reason_Consistency_Score": reason_similarities[actor][
                        "mean_similarity"
                    ],
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    plt.figure(figsize=(12, 8))

    plot_df = comparison_df.dropna(subset=["Reason_Consistency_Score"])

    from matplotlib.colors import Normalize

    min_val = plot_df["Reason_Consistency_Score"].min()
    max_val = plot_df["Reason_Consistency_Score"].max()

    norm = Normalize(vmin=min_val, vmax=max_val)

    scatter = plt.scatter(
        plot_df["Intra-Actor_Diversity_Score"],
        plot_df["Inter-Actor_Similarity_Score"],
        s=120,
        alpha=0.8,
        c=plot_df["Reason_Consistency_Score"],
        cmap="Blues",
        norm=norm,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.colorbar(scatter, label="Reason Consistency Score")

    for i, row in plot_df.iterrows():
        plt.annotate(
            row["Actor"],
            (row["Intra-Actor_Diversity_Score"], row["Inter-Actor_Similarity_Score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.xlabel("1 - Intra-Actor Similarity")
    plt.ylabel("Inter-Actor Similarity")

    title = "Intra-Actor similarity vs. inter-Actor similarity Analysis (Color = Reason-wise consistency)"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("=== COMPLETE ANALYSIS RESULTS ===")
    display_cols = [
        "Actor",
        "Intra-Actor_Diversity_Score",
        "Inter-Actor_Similarity_Score",
        "Reason_Consistency_Score",
    ]
    print(comparison_df[display_cols].round(4))

    return comparison_df


cross_analysis_df = cross_analyze_actor_similarity(
    row_similarities, column_similarities, reason_similarities
)

# %% Save cross-analysis results
# Save cross-analysis results as JSON
cross_analysis_dict = cross_analysis_df.to_dict("records")
with open(results_dir / "cross_analysis_results.json", "w") as f:
    json.dump(cross_analysis_dict, f, indent=2)
print(f"Cross-analysis results saved to {results_dir / 'cross_analysis_results.json'}")

# %% [markdown]
# ## Summary of Findings
#
# This analysis examines embedding similarities across all available reasoning types for 7 LLM actors and human responses on ethical scenarios, revealing distinct behavioral patterns in ethical decision-making.
#
# ### Key Findings:
#
# 1. **Inter-Actor Agreement** (LLM-to-LLM similarity range: 59.1% - 79.6%):
#    - **Claude** shows highest agreement with other LLMs (77.5% with GPT-3.5, 79.6% with Llama)
#    - **Bison** shows lowest agreement with other LLMs (59.1% with Mistral, 61.6% with Gemma)
#    - LLM actors generally show moderate to high consensus (mean ~68%) across ethical scenarios
#    - **Human responses** show much lower agreement with LLMs (40.1% - 46.9% similarity)
#    - Human-LLM alignment is consistently lower than inter-LLM agreement, indicating distinct reasoning patterns
#
# 2. **Intra-Actor Consistency** (Range: 18.4% - 49.8%):
#    - **Gemma** is most internally consistent (49.8% ± 10.1%) - most predictable across scenarios
#    - **Human** responses are least internally consistent (18.4% ± 12.2%) - most context-dependent
#    - **GPT-4** shows low internal consistency (31.9% ± 12.4%) - most diverse across scenarios
#    - **Bison** shows moderate internal consistency (28.5% ± 12.1%) - balanced variability
#    - Internal consistency range of 31.4% indicates significant diversity in actor response patterns
#
# 3. **Reason-wise Consistency** (Range: 72.9% - 100%):
#    - **Human** shows perfect reasoning consistency (100%) - single reasoning approach per scenario
#    - **Claude** shows highest LLM reasoning coherence (90.6% ± 5.6%) - most consistent across reasoning approaches
#    - **Mistral** shows lowest reasoning coherence (72.9% ± 11.7%) - most variable across reasoning approaches
#    - Mean LLM reason-wise consistency (80.7% ± 6.1%) much higher than intra-actor consistency
#    - This indicates actors are more consistent within reasoning types than across different scenarios
#
# 4. **Three-Dimensional Actor Profiles**:
#    - **Claude**: High inter-actor agreement (69.7%), moderate intra-actor consistency (43.1%), highest reasoning coherence (90.6%)
#    - **Gemma**: Moderate inter-actor agreement (65.4%), highest intra-actor consistency (49.8%), moderate reasoning coherence (76.4%)
#    - **GPT-4**: Moderate inter-actor agreement (64.7%), lowest intra-actor consistency (31.9%), moderate reasoning coherence (76.7%)
#    - **Bison**: Lowest inter-actor agreement (61.3%), low intra-actor consistency (28.5%), high reasoning coherence (82.4%)
#    - **Human**: Low inter-actor agreement (43.4%), lowest intra-actor consistency (18.4%), perfect reasoning coherence (100%)
#
# 5. **Human-LLM Alignment** (Range: 40.1% - 46.9%):
#    - **Bison** shows highest human alignment (46.9% ± 16.5%) - most human-like reasoning
#    - **Mistral** shows lowest human alignment (40.1% ± 13.8%) - least human-like reasoning
#    - **GPT-4** shows moderate human alignment (45.3% ± 15.7%) - balanced human similarity
#    - Range of 6.8% indicates moderate variation in human alignment across LLMs
#    - All LLMs show substantial variability (13.8% - 16.5% std), suggesting context-dependent human alignment
#
# ### Practical Implications:
#
# - **Most Predictable Ethics**: Gemma (highest internal consistency across scenarios)
# - **Most Coherent Reasoning**: Claude (most consistent across different reasoning approaches)
# - **Most Diverse Perspectives**: GPT-4 and Human (high variability in responses)
# - **Most Human-Like Reasoning**: Bison (highest human alignment at 46.9%)
# - **Best Overall Balance**: Claude (reliable consensus + coherent reasoning + moderate diversity)
# - **Best for Human Collaboration**: Bison and GPT-4 (highest human alignment scores)
# - **Most Independent Reasoning**: Mistral and Human (lowest inter-actor alignment, most unique perspectives)
# - **Most Context-Dependent**: Human responses (lowest internal consistency, suggesting high situational awareness)

# %%
