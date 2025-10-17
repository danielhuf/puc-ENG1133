# %% [markdown]
### Embedding Similarity Analysis
#
# This notebook analyzes similarities between the embeddings of the ethical dilemma dataset in three main ways:
# 1. **Scenario-wise analysis**: Compare different actors' responses to the same scenario (ethical dilemma)
# 2. **Actor-wise analysis**: Compare a same actor's responses to different scenarios
# 3. **Reason-wise analysis**: Compare different reasoning versions for a same actor in the same scenario
#
# The actors considered for this analysis are:
# - **LLM Models**: GPT-3.5, GPT-4, Claude Haiku, PaLM 2 Bison, Gemma 7B, Mistral 7B, and Llama 2.
# - **Human Redditors**: The author of the top comment of each scenario submission.
# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle
import json
from pathlib import Path
from embedding_utils import (
    load_embeddings,
    analyze_row_similarities,
    identify_actors_and_reasons,
    plot_row_similarity_distribution,
    summarize_row_characteristics,
    display_edge_llm_human_similarities,
    analyze_column_similarities,
    plot_column_similarity_comparison,
    summarize_column_characteristics,
    display_edge_scenario_similarities,
)

plt.rcParams["figure.max_open_warning"] = 0
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# %% Load and explore the embeddings data
df, embeddings_dict = load_embeddings("../data/embeddings.csv")

# %% Identify actors and reason types
actors, reason_types = identify_actors_and_reasons(embeddings_dict)

# %% [markdown]
# ## 1. Scenario-wise Analysis
#
# This analysis compares how human redditors and LLM models respond to the same ethical dilemma.
# For each scenario (row), embedding similarities are calculated between all pairs of actors.


# %% Scenario-wise similarity analysis
cache_path = Path("../results/base/row_similarities.pkl")
cache_path.parent.mkdir(parents=True, exist_ok=True)
if cache_path.exists():
    with open(cache_path, "rb") as f:
        row_similarities = pickle.load(f)
    print(f"row_similarities loaded from cache")
else:
    row_similarities = analyze_row_similarities(embeddings_dict, actors, reason_types)
    with open(cache_path, "wb") as f:
        pickle.dump(row_similarities, f)
    print(f"Saved row_similarities to {cache_path}")


# %% Visualize scenario-wise similarities
plot_row_similarity_distribution(row_similarities)


# %% Statistical summary of scenario-wise similarities
row_summary_df = summarize_row_characteristics(row_similarities)

# %% Save scenario-wise analysis results
results_dir = Path("../results/base")
results_dir.mkdir(exist_ok=True)

row_summary_dict = row_summary_df.to_dict("records")
with open(results_dir / "scenario_wise_analysis_results.json", "w") as f:
    json.dump(row_summary_dict, f, indent=2)
print(
    f"Scenario-wise analysis results saved to {results_dir / 'scenario_wise_analysis_results.json'}"
)


# %% Display LLM-Human similarity edge cases
df_cleaned = pd.read_csv("../data/ethical_dilemmas_cleaned.csv")
display_edge_llm_human_similarities(row_similarities, df_cleaned)

# %% [markdown]
# ## 2. Actor-wise Analysis
#
# This analysis compares how a same actor responds to different ethical dilemmas.
# For each actor, we calculate the similarity between all pairs of scenarios.


# %% Actor-wise similarity analysis
column_similarities = analyze_column_similarities(embeddings_dict, actors, reason_types)


# %% Visualize actor-wise similarities
plot_column_similarity_comparison(column_similarities)


# %% Statistical summary of actor-wise differences
column_summary_df = summarize_column_characteristics(column_similarities)

# %% Save column-wise analysis results
column_summary_dict = column_summary_df.to_dict("records")
with open(results_dir / "actor_wise_analysis_results.json", "w") as f:
    json.dump(column_summary_dict, f, indent=2)
print(
    f"Actor-wise analysis results saved to {results_dir / 'actor_wise_analysis_results.json'}"
)


# %% Display scenario similarity edge cases
display_edge_scenario_similarities(embeddings_dict, actors, reason_types, df_cleaned)

# %% [markdown]
# ## 3. Reason-wise Analysis
#
# This analysis compares how consistent each actor's reasonings are when answering the same ethical dilemma.


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

        all_similarities = []

        for i, reason1 in enumerate(available_reasons):
            for reason2 in available_reasons[i + 1 :]:
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


cache_path = Path("../results/base/reason_similarities.pkl")
if cache_path.exists():
    with open(cache_path, "rb") as f:
        reason_similarities = pickle.load(f)
    print(f"reason_similarities loaded from cache")
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

    _, axes = plt.subplots(
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

    for _, row in plot_df.iterrows():
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

    print("=== CROSS-ANALYSIS RESULTS ===")
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
# 2. **Intra-Actor Agreement** (Range: 18.4% - 49.8%):
#    - **Gemma** shows highest intra-actor agreement (49.8% ± 10.1%) - most predictable across scenarios
#    - **Human** responses are least internally consistent (18.4% ± 12.2%) - most context-dependent
#    - **GPT-4** shows low intra-actor agreement (31.9% ± 12.4%) - most diverse across scenarios
#    - **Bison** shows moderate intra-actor agreement (28.5% ± 12.1%) - balanced variability
#    - Internal agreement range of 31.4% indicates significant diversity in actor response patterns
#
# 3. **Reason-wise Consistency** (Range: 72.9% - 100%):
#    - **Human** shows perfect reasoning consistency (100%) - single reasoning approach per scenario
#    - **Claude** shows highest LLM reasoning coherence (90.6% ± 5.6%) - most consistent across reasoning approaches
#    - **Mistral** shows lowest reasoning coherence (72.9% ± 11.7%) - most variable across reasoning approaches
#    - Mean LLM reason-wise consistency (80.7% ± 6.1%) much higher than intra-actor consistency
#    - This indicates actors are more consistent within reasoning types than across different scenarios
#
# 4. **Three-Dimensional Actor Profiles**:
#    - **Claude**: High inter-actor agreement (69.7%), moderate intra-actor agreement (43.1%), highest reasoning consistency (90.6%)
#    - **Gemma**: Moderate inter-actor agreement (65.4%), highest intra-actor agreement (49.8%), moderate reasoning consistency (76.4%)
#    - **GPT-4**: Moderate inter-actor agreement (64.7%), lowest intra-actor agreement (31.9%), moderate reasoning consistency (76.7%)
#    - **Bison**: Lowest inter-actor agreement (61.3%), low intra-actor agreement (28.5%), high reasoning consistency (82.4%)
#    - **Human**: Low inter-actor agreement (43.4%), lowest intra-actor agreement (18.4%), perfect reasoning consistency (100%)
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
