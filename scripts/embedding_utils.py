"""
Common utilities for embedding similarity analysis.

This module contains shared functions used by the embedding analysis notebooks
to avoid code duplication and ensure consistency across different language analyses.
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import Normalize
import json

plt.rcParams["figure.max_open_warning"] = 0
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

plt.style.use("seaborn-v0_8")


def get_language_color(language_code: str) -> str:
    """Get color based on language code.

    Args:
        language_code: Language code (Base, Portuguese, German, Spanish, French)

    Returns:
        Color name for plotting
    """
    color_map = {
        "Base": "steelblue",
        "Portuguese": "lightgreen",
        "German": "dimgray",
        "Spanish": "goldenrod",
        "French": "skyblue",
    }
    return color_map.get(language_code, "skyblue")


def load_embeddings(embeddings_file: str) -> Dict[str, np.ndarray]:
    """Load embeddings from CSV and organize them by column.

    Args:
        embeddings_file: Path to the CSV file containing embeddings

    Returns:
        Dictionary where keys are column names and values are numpy arrays
    """
    df = pd.read_csv(embeddings_file)
    embedding_cols = [col for col in df.columns if col.endswith("_embedding")]
    embeddings_dict = {}

    for col in embedding_cols:

        def parse_embedding(x):
            """Convert string representation to numpy array."""
            if pd.isna(x):
                return np.zeros(384, dtype=np.float32)
            return np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32)

        embeddings = df[col].apply(parse_embedding).values
        embeddings_array = np.vstack(embeddings)
        embeddings_dict[col] = embeddings_array

    return embeddings_dict


def analyze_row_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze inter-actor similarity on the same moral scenarios.

    This function compares how different LLM models and human redditors respond to a same moral dilemma.
    For each scenario (row) in the dataset, it calculates similarities between all possible
    actor pairs by comparing their reasoning embeddings.

    The analysis accounts for actors having different numbers of reasoning approaches
    by comparing all available combinations and taking the mean similarity.

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
        if actor == "redditor":
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
                actor1_reasons = actor_reason_combinations[actor1]
                actor2_reasons = actor_reason_combinations[actor2]

                pair_similarities = []
                for reason1 in actor1_reasons:
                    for reason2 in actor2_reasons:
                        if actor1 == "redditor":
                            col_name1 = "top_comment_embedding"
                        else:
                            col_name1 = f"{actor1}_{reason1}_embedding"

                        if actor2 == "redditor":
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


def identify_actors_and_reasons(
    embeddings_dict: Dict[str, np.ndarray],
) -> Tuple[List[str], List[str]]:
    """Identify actors and reason types from embedding column names.

    Args:
        embeddings_dict: Dictionary mapping column names to embedding arrays

    Returns:
        Tuple of (actors, reason_types) sorted lists
    """
    actors = set()
    reason_types = set()

    for col in embeddings_dict.keys():
        if col in ["selftext_embedding"]:
            continue

        if col == "top_comment_embedding":
            actors.add("redditor")
            reason_types.add("top_comment")
            continue

        parts = col.replace("_embedding", "").split("_")

        actor = parts[0]
        reason = "_".join(parts[1:])

        actors.add(actor)
        reason_types.add(reason)

    desired_order = [
        "redditor",
        "gpt3.5",
        "gpt4",
        "claude",
        "bison",
        "gemini",
        "llama",
        "mistral",
        "gemma",
    ]
    actors_list = list(actors)
    actors_known = [a for a in actors_list if a in desired_order]
    actors_unknown = sorted([a for a in actors_list if a not in desired_order])
    actors_sorted = (
        sorted(actors_known, key=lambda a: desired_order.index(a)) + actors_unknown
    )

    actors = actors_sorted
    reason_types = sorted(list(reason_types))

    return actors, reason_types


def load_or_compute_similarities(
    language_code, embeddings_dict, actors, reason_types, similarity_type
):
    """
    Load similarities from cache or compute them if not cached.

    Args:
        language_code: Language code (br, de, es, fr)
        embeddings_dict: Dictionary containing embeddings
        actors: List of actors
        reason_types: List of reason types
        similarity_type: Type of similarity ('row' or 'reason')

    Returns:
        Similarities data
    """
    cache_path = Path(f"../results/{language_code}/{similarity_type}_similarities.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            similarities = pickle.load(f)
    else:
        if similarity_type == "row":
            similarities = analyze_row_similarities(
                embeddings_dict, actors, reason_types
            )
        elif similarity_type == "reason":
            similarities = analyze_reason_similarities(
                embeddings_dict, actors, reason_types
            )
        else:
            raise ValueError("similarity_type must be 'row' or 'reason'")

        with open(cache_path, "wb") as f:
            pickle.dump(similarities, f)

    return similarities


def compute_global_similarity_bounds(
    all_row_similarities: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Compute global min/max values for similarity and density across all datasets.

    Args:
        all_row_similarities: Dictionary mapping language codes to row similarities

    Returns:
        Tuple of (global_min_sim, global_max_sim, global_min_density, global_max_density)
    """
    all_similarities = []
    all_densities = []

    for language_code, row_similarities in all_row_similarities.items():
        pair_similarities = {}
        for row_data in row_similarities.values():
            for pair, sim in row_data.items():
                if pair not in pair_similarities:
                    pair_similarities[pair] = []
                pair_similarities[pair].append(sim)

        for similarities in pair_similarities.values():
            all_similarities.extend(similarities)

            hist, bins = np.histogram(similarities, bins=15, density=True)
            all_densities.extend(hist)

    global_min_sim = min(all_similarities)
    global_max_sim = max(all_similarities)
    global_min_density = min(all_densities)
    global_max_density = max(all_densities)

    return global_min_sim, global_max_sim, global_min_density, global_max_density


def plot_row_similarity_comparison(
    row_similarities: Dict,
    language_code: str,
    global_bounds: Tuple[float, float, float, float] = None,
):
    """Plot distribution of similarities across scenarios.

    Displays mean similarities across all available reason type combinations
    for each actor pair.

    Args:
        row_similarities: Dictionary containing similarity data
        language_code: Language code for coloring
        global_bounds: Optional tuple of (min_sim, max_sim, min_density, max_density)
                      for standardized axis ranges across all datasets
    """

    pair_similarities = {}
    for row_data in row_similarities.values():
        for pair, sim in row_data.items():
            if pair not in pair_similarities:
                pair_similarities[pair] = []
            pair_similarities[pair].append(sim)

    # Extract unique actors and apply custom ordering
    actors_set = set()
    for pair in pair_similarities.keys():
        actor1, actor2 = pair.split("_vs_")
        actors_set.add(actor1)
        actors_set.add(actor2)

    # Apply custom actor ordering
    desired_order = [
        "redditor",
        "gpt3.5",
        "gpt4",
        "claude",
        "bison",
        "gemini",
        "llama",
        "mistral",
        "gemma",
    ]
    actors_list = list(actors_set)
    actors_known = [a for a in actors_list if a in desired_order]
    actors_unknown = sorted([a for a in actors_list if a not in desired_order])
    actors = sorted(actors_known, key=lambda a: desired_order.index(a)) + actors_unknown

    n_actors = len(actors)

    _, axes = plt.subplots(
        n_actors - 1, n_actors - 1, figsize=(3 * (n_actors - 1), 3 * (n_actors - 1))
    )

    if n_actors == 2:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_actors - 1):
        for j in range(n_actors - 1):
            ax = axes[i, j]

            if i >= j:
                pair_name = f"{actors[j]}_vs_{actors[i+1]}"
                if pair_name in pair_similarities:
                    similarities = pair_similarities[pair_name]

                    ax.hist(
                        similarities,
                        bins=15,
                        alpha=0.7,
                        color=get_language_color(language_code),
                        edgecolor="black",
                        density=True,
                    )

                    median_sim = np.median(similarities)

                    ax.axvline(
                        median_sim, color="red", linestyle="--", alpha=0.8, linewidth=1
                    )

                    ax.set_title(
                        f"{actors[j]} vs {actors[i+1]}\nMedian: {median_sim:.3f}",
                        fontsize=12,
                        pad=4,
                    )
                    ax.set_xlabel("Similarity", fontsize=10)
                    ax.set_ylabel("Density", fontsize=10)
                    ax.tick_params(labelsize=8)
                    ax.grid(True, alpha=0.3)

                    if global_bounds is not None:
                        min_sim, max_sim, min_density, max_density = global_bounds
                        ax.set_xlim(min_sim, max_sim)
                        ax.set_ylim(min_density, max_density)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)

    for i in range(n_actors - 1):
        axes[i, 0].set_ylabel(actors[i + 1].upper(), fontsize=14, fontweight="bold")
        axes[-1, i].set_xlabel(actors[i].upper(), fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.suptitle(
        f"Scenario-wise Similarity Distributions ({language_code})",
        y=1.02,
        fontsize=24,
        fontweight="bold",
    )
    plt.show()


def summarize_row_characteristics(row_similarities: Dict, language_code: str):
    """Provide statistical summary of inter-actor similarity patterns."""

    print(f"\n=== SCENARIO-WISE SIMILARITY SUMMARY ({language_code.upper()}) ===\n")

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


def save_analysis_results(
    language_code: str, summary_df: pd.DataFrame, analysis_type: str
):
    """
    Save analysis results to JSON file.

    Args:
        language_code: Language code (br, de, es, fr)
        summary_df: DataFrame containing summary statistics
        analysis_type: Type of analysis ('scenario_wise', 'actor_wise', 'reason_wise')
    """
    results_dir = Path(f"../results/{language_code}")
    results_dir.mkdir(exist_ok=True)

    summary_dict = summary_df.to_dict("records")
    output_file = results_dir / f"{analysis_type}_analysis_results.json"

    with open(output_file, "w") as f:
        json.dump(summary_dict, f, indent=2)


def display_edge_llm_human_similarities(
    row_similarities: Dict, df_cleaned: pd.DataFrame, language_code: str
):
    """Display the top 5 answers with highest and lowest LLM-Redditor similarity."""

    llm_redditor_similarities = []

    for row_idx, row_data in row_similarities.items():
        for pair, similarity in row_data.items():
            if "redditor" in pair and pair != "redditor_vs_redditor":
                llm_model = pair.replace("_vs_redditor", "").replace("redditor_vs_", "")
                llm_redditor_similarities.append(
                    {
                        "row_idx": row_idx,
                        "llm_model": llm_model,
                        "similarity": similarity,
                        "pair": pair,
                    }
                )

    llm_redditor_similarities.sort(key=lambda x: x["similarity"])

    lowest_5 = llm_redditor_similarities[:5]
    highest_5 = llm_redditor_similarities[-5:]

    print("\n" + "=" * 80)
    print(f"EDGE LLM-REDDITOR SIMILARITY CASES ({language_code.upper()})\n")
    print("=" * 80)

    print("\nTOP 5 LOWEST SIMILARITY CASES (Most semantically different answers)")
    print("-" * 60)

    for i, case in enumerate(lowest_5, 1):
        row_idx = case["row_idx"]
        llm_model = case["llm_model"]
        similarity = case["similarity"]

        scenario = df_cleaned.iloc[row_idx]
        scenario_id = scenario["submission_id"]
        title = scenario["title"]

        human_comment = scenario["top_comment"]

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
            f"\n{i}. Similarity: {similarity:.4f} | Scenario ID: {scenario_id} | Model: {llm_model.upper()}"
        )
        print(f"   Title: {title}")
        print(f"   Redditor Comment: {human_comment}")
        print(f"   {llm_model.upper()} Reasoning: {llm_reason}")

    print("\nTOP 5 HIGHEST SIMILARITY CASES (Most semantically similar answers)")
    print("-" * 60)

    for i, case in enumerate(highest_5, 1):
        row_idx = case["row_idx"]
        llm_model = case["llm_model"]
        similarity = case["similarity"]

        scenario = df_cleaned.iloc[row_idx]
        scenario_id = scenario["submission_id"]
        title = scenario["title"]

        human_comment = scenario["top_comment"]

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
            f"\n{i}. Similarity: {similarity:.4f} | Scenario ID: {scenario_id} | Model: {llm_model.upper()}"
        )
        print(f"   Title: {title}")
        print(f"   Redditor Comment: {human_comment}")
        print(f"   {llm_model.upper()} Reasoning: {llm_reason}")

    return


def analyze_column_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-actor similarity across different moral scenarios.

    This function measures how consistent each actor is when responding to different
    moral dilemmas. It calculates the similarity between all pairs of scenarios for
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

        if actor == "redditor":
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


def compute_global_column_similarity_bounds(
    all_column_similarities: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Compute global min/max values for column similarity and density across all datasets.

    Args:
        all_column_similarities: Dictionary mapping language codes to column similarities

    Returns:
        Tuple of (global_min_sim, global_max_sim, global_min_density, global_max_density)
    """
    all_similarities = []
    all_densities = []

    for language_code, column_similarities in all_column_similarities.items():
        for actor, data in column_similarities.items():
            similarities = data["similarities"]
            all_similarities.extend(similarities)

            hist, bins = np.histogram(similarities, bins=30, density=True)
            all_densities.extend(hist)

    global_min_sim = min(all_similarities)
    global_max_sim = max(all_similarities)
    global_min_density = min(all_densities)
    global_max_density = max(all_densities)

    return global_min_sim, global_max_sim, global_min_density, global_max_density


def plot_column_similarity_comparison(
    column_similarities: Dict,
    language_code: str,
    global_bounds: Tuple[float, float, float, float] = None,
):
    """Compare intra-actor similarity distributions.

    Args:
        column_similarities: Dictionary containing similarity data
        language_code: Language code for coloring
        global_bounds: Optional tuple of (min_sim, max_sim, min_density, max_density)
                      for standardized axis ranges across all datasets
    """

    actor_names = list(column_similarities.keys())
    n_actors = len(actor_names)

    height_ratios = [2.5] * n_actors + [3]

    fig, axes = plt.subplots(
        n_actors + 1,
        1,
        figsize=(12, 2.5 * n_actors + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    plt.suptitle(
        f"Actor-wise Similarity Distributions ({language_code.upper()})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for i, (actor, data) in enumerate(
        tqdm(column_similarities.items(), desc="Plotting histograms")
    ):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=30,
            alpha=0.7,
            color=get_language_color(language_code),
            edgecolor="black",
            density=True,
        )

        median_sim = np.median(data["similarities"])
        ax.set_title(f"{actor.upper()}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        ax.axvline(
            median_sim,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Median: {median_sim:.3f}",
        )

        if global_bounds is not None:
            min_sim, max_sim, min_density, max_density = global_bounds
            ax.set_xlim(min_sim, max_sim)
            ax.set_ylim(min_density, max_density)
        else:
            ax.set_xlim(0, 1)
        ax.legend()

    ax_box = axes[-1]

    similarities_data = [data["similarities"] for data in column_similarities.values()]
    actor_names_upper = [name.upper() for name in actor_names]

    box_plot = ax_box.boxplot(
        similarities_data, tick_labels=actor_names_upper, patch_artist=True
    )

    for patch in box_plot["boxes"]:
        patch.set_facecolor(get_language_color(language_code))
        patch.set_alpha(0.8)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def summarize_column_characteristics(column_similarities: Dict, language_code: str):
    """Provide statistical summary of each actor's characteristics."""

    print(f"\n=== ACTOR-WISE SIMILARITY SUMMARY ({language_code.upper()}) ===\n")

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


def display_edge_scenario_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
    df_cleaned: pd.DataFrame,
    language_code: str,
):
    """Display the top 5 answers with highest and lowest scenario similarity for both redditors and LLMs."""

    actor_scenario_similarities = {}

    for actor_idx, actor in enumerate(actors, 1):
        print(f"Processing actor {actor_idx}/{len(actors)}: {actor}")
        available_reasons = []
        actor_embeddings = {}

        if actor == "redditor":
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

        n_scenarios = mean_embeddings_norm.shape[0]

        lowest_pairs = [(float("inf"), -1, -1)] * 5
        highest_pairs = [(float("-inf"), -1, -1)] * 5

        chunk_size = 500

        for i in range(0, n_scenarios, chunk_size):
            end_i = min(i + chunk_size, n_scenarios)
            chunk_embeddings = mean_embeddings_norm[i:end_i]

            chunk_similarities = cosine_similarity(
                chunk_embeddings, mean_embeddings_norm
            )

            for local_i, global_i in enumerate(range(i, end_i)):
                for j in range(global_i + 1, n_scenarios):
                    similarity = chunk_similarities[local_i, j]

                    if similarity < lowest_pairs[-1][0]:
                        lowest_pairs[-1] = (similarity, global_i, j)
                        lowest_pairs.sort(key=lambda x: x[0])

                    if similarity > highest_pairs[-1][0]:
                        highest_pairs[-1] = (similarity, global_i, j)
                        highest_pairs.sort(key=lambda x: x[0], reverse=True)

        lowest_pairs = [pair for pair in lowest_pairs if pair[1] != -1]
        highest_pairs = [pair for pair in highest_pairs if pair[1] != -1]

        lowest_pairs.sort(key=lambda x: x[0])
        highest_pairs.sort(key=lambda x: x[0])

        actor_scenario_similarities[actor] = {
            "lowest": lowest_pairs,
            "highest": highest_pairs,
        }

    human_data = actor_scenario_similarities.get(
        "redditor", {"lowest": [], "highest": []}
    )
    human_lowest_5 = human_data["lowest"]
    human_highest_5 = human_data["highest"]

    all_llm_lowest = []
    all_llm_highest = []

    for actor in actors:
        if actor != "redditor":
            actor_data = actor_scenario_similarities.get(
                actor, {"lowest": [], "highest": []}
            )
            for case in actor_data["lowest"]:
                all_llm_lowest.append((actor, case))
            for case in actor_data["highest"]:
                all_llm_highest.append((actor, case))

    all_llm_lowest.sort(key=lambda x: x[1][0])
    all_llm_highest.sort(key=lambda x: x[1][0], reverse=True)

    llm_lowest_5 = all_llm_lowest[:5]
    llm_highest_5 = all_llm_highest[:5]

    print("\n" + "=" * 100)
    print(f"EDGE SCENARIO SIMILARITY CASES ({language_code.upper()})")
    print("=" * 100)

    print("\n👥 REDDITOR RESPONSES")
    print("=" * 60)

    print("\nTOP 5 LOWEST SIMILARITY CASES (Most semantically different answers)")
    print("-" * 60)

    for i, case in enumerate(human_lowest_5, 1):
        similarity, idx1, idx2 = case

        scenario1 = df_cleaned.iloc[idx1]
        scenario2 = df_cleaned.iloc[idx2]

        title1 = scenario1["title"]
        title2 = scenario2["title"]

        comment1 = scenario1["top_comment"]
        comment2 = scenario2["top_comment"]

        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   Scenario 1 (ID: {scenario1['submission_id']}): {title1}")
        print(f"   Redditor Comment 1: {comment1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")
        print(f"   Redditor Comment 2: {comment2}")

    print("\nTOP 5 HIGHEST SIMILARITY CASES (Most semantically similar answers)")
    print("-" * 60)

    for i, case in enumerate(human_highest_5, 1):
        similarity, idx1, idx2 = case

        scenario1 = df_cleaned.iloc[idx1]
        scenario2 = df_cleaned.iloc[idx2]

        title1 = scenario1["title"]
        title2 = scenario2["title"]

        comment1 = scenario1["top_comment"]
        comment2 = scenario2["top_comment"]

        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   Scenario 1 (ID: {scenario1['submission_id']}): {title1}")
        print(f"   Redditor Comment 1: {comment1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")
        print(f"   Redditor Comment 2: {comment2}")

    print("\n🤖 LLM RESPONSES")
    print("=" * 60)

    print("\nTOP 5 LOWEST SIMILARITY CASES (Most semantically different answers)")
    print("-" * 60)

    for i, (actor, case) in enumerate(llm_lowest_5, 1):
        similarity, idx1, idx2 = case

        scenario1 = df_cleaned.iloc[idx1]
        scenario2 = df_cleaned.iloc[idx2]

        title1 = scenario1["title"]
        title2 = scenario2["title"]

        print(f"\n{i}. Similarity: {similarity:.4f} | Model: {actor.upper()}")
        print(f"   Scenario 1 (ID: {scenario1['submission_id']}): {title1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")

        llm_reason1 = None
        llm_reason2 = None
        for reason_col in [
            f"{actor}_reason_1",
            f"{actor}_reason_2",
            f"{actor}_reason_3",
        ]:
            if (
                reason_col in scenario1
                and pd.notna(scenario1[reason_col])
                and llm_reason1 is None
            ):
                llm_reason1 = scenario1[reason_col]
            if (
                reason_col in scenario2
                and pd.notna(scenario2[reason_col])
                and llm_reason2 is None
            ):
                llm_reason2 = scenario2[reason_col]

        reason1_preview = llm_reason1
        reason2_preview = llm_reason2

        print(f"   {actor.upper()} Reasoning:")
        print(f"     Scenario 1: {reason1_preview}")
        print(f"     Scenario 2: {reason2_preview}")

    print("\nTOP 5 HIGHEST SIMILARITY CASES (Most semantically similar answers)")
    print("-" * 60)

    for i, (actor, case) in enumerate(llm_highest_5, 1):
        similarity, idx1, idx2 = case

        scenario1 = df_cleaned.iloc[idx1]
        scenario2 = df_cleaned.iloc[idx2]

        title1 = scenario1["title"]
        title2 = scenario2["title"]

        print(f"\n{i}. Similarity: {similarity:.4f} | Model: {actor.upper()}")
        print(f"   Scenario 1 (ID: {scenario1['submission_id']}): {title1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")

        llm_reason1 = None
        llm_reason2 = None
        for reason_col in [
            f"{actor}_reason_1",
            f"{actor}_reason_2",
            f"{actor}_reason_3",
        ]:
            if (
                reason_col in scenario1
                and pd.notna(scenario1[reason_col])
                and llm_reason1 is None
            ):
                llm_reason1 = scenario1[reason_col]
            if (
                reason_col in scenario2
                and pd.notna(scenario2[reason_col])
                and llm_reason2 is None
            ):
                llm_reason2 = scenario2[reason_col]

        reason1_preview = llm_reason1
        reason2_preview = llm_reason2

        print(f"   {actor.upper()} Reasoning:")
        print(f"     Scenario 1: {reason1_preview}")
        print(f"     Scenario 2: {reason2_preview}")

    return


def analyze_reason_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-actor reasoning consistency across different reasoning approaches (reason-wise analysis).

    This function examines how consistent each actor is when applying different reasoning
    approaches to the same moral scenario. It measures the similarity between a actor's
    various reasoning types (reason_1, reason_2, etc.) when confronted with identical
    moral dilemmas.

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

        if actor == "redditor":
            if "top_comment_embedding" in embeddings_dict:
                available_reasons.append("top_comment")
                actor_embeddings["top_comment"] = embeddings_dict[
                    "top_comment_embedding"
                ]
                # For redditor actor, reason consistency is 1.0 (perfect consistency since only one comment per scenario)
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


def compute_global_reason_similarity_bounds(
    all_reason_similarities: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Compute global min/max values for reason similarity and density across all datasets.

    Args:
        all_reason_similarities: Dictionary mapping language codes to reason similarities

    Returns:
        Tuple of (global_min_sim, global_max_sim, global_min_density, global_max_density)
    """
    all_similarities = []
    all_densities = []

    for language_code, reason_similarities in all_reason_similarities.items():
        for actor, data in reason_similarities.items():
            if actor != "redditor":
                similarities = data["similarities"]
                all_similarities.extend(similarities)

                hist, bins = np.histogram(similarities, bins=30, density=True)
                all_densities.extend(hist)

    global_min_sim = min(all_similarities)
    global_max_sim = max(all_similarities)
    global_min_density = min(all_densities)
    global_max_density = max(all_densities)

    return global_min_sim, global_max_sim, global_min_density, global_max_density


def plot_reason_similarity_comparison(
    reason_similarities: Dict,
    language_code: str,
    global_bounds: Tuple[float, float, float, float] = None,
):
    """Compare reason-wise similarity distributions with separate subplots for each actor.

    Args:
        reason_similarities: Dictionary containing similarity data
        language_code: Language code for coloring
        global_bounds: Optional tuple of (min_sim, max_sim, min_density, max_density)
                      for standardized axis ranges across all datasets
    """

    filtered_reason_similarities = {
        k: v for k, v in reason_similarities.items() if k != "redditor"
    }
    actor_names = list(filtered_reason_similarities.keys())
    n_actors = len(actor_names)

    height_ratios = [2.5] * n_actors + [3]

    fig, axes = plt.subplots(
        n_actors + 1,
        1,
        figsize=(12, 2.5 * n_actors + 3),
        gridspec_kw={"height_ratios": height_ratios},
    )

    plt.suptitle(
        f"Reason-wise Similarity Distributions ({language_code.upper()})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if n_actors == 1:
        axes = [axes[0], axes[1], axes[2]]

    for i, (actor, data) in enumerate(
        tqdm(filtered_reason_similarities.items(), desc="Plotting reason similarities")
    ):
        ax = axes[i]
        ax.hist(
            data["similarities"],
            bins=30,
            alpha=0.7,
            color=get_language_color(language_code),
            edgecolor="black",
            density=True,
        )

        median_sim = np.median(data["similarities"])
        n_reasons = len(data["available_reasons"])
        ax.set_title(f"{actor.upper()} ({n_reasons} reasonings)")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        ax.axvline(
            median_sim,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Median: {median_sim:.3f}",
        )

        if global_bounds is not None:
            min_sim, max_sim, min_density, max_density = global_bounds
            ax.set_xlim(min_sim, max_sim)
            ax.set_ylim(min_density, max_density)
        else:
            ax.set_xlim(0, 1)
        ax.legend()

    ax_box = axes[-1]

    similarities_data = [
        data["similarities"] for data in filtered_reason_similarities.values()
    ]
    actor_names_upper = [name.upper() for name in actor_names]

    box_plot = ax_box.boxplot(
        similarities_data, tick_labels=actor_names_upper, patch_artist=True
    )

    for patch in box_plot["boxes"]:
        patch.set_facecolor(get_language_color(language_code))
        patch.set_alpha(0.8)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def summarize_reason_characteristics(reason_similarities: Dict, language_code: str):
    """Provide statistical summary of each actor's reason-wise characteristics."""

    print(f"\n=== REASON-WISE SIMILARITY SUMMARY ({language_code.upper()}) ===\n")

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


def cross_analyze_actor_similarity(
    row_similarities: Dict,
    column_similarities: Dict,
    reason_similarities: Dict,
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

    return comparison_df


def cross_analyze_multiple_languages(
    all_similarities_data: Dict[str, Dict],
    language_codes: List[str],
):
    """Analyze the relationship between inter-actor similarity and intra-actor similarity across multiple languages.
    Shows Base as a single plot and other languages in a 2x2 grid."""

    def create_single_plot(ax, language_code, data, norm, is_base=False):
        """Helper function to create a single cross-analysis plot."""
        row_similarities = data["row_similarities"]
        column_similarities = data["column_similarities"]
        reason_similarities = data["reason_similarities"]

        inter_actor_means = {}
        for actor in column_similarities.keys():
            actor_pairs = [
                pair
                for pair in list(row_similarities.values())[0].keys()
                if actor in pair
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
        plot_df = comparison_df.dropna(subset=["Reason_Consistency_Score"])

        if not plot_df.empty:

            scatter = ax.scatter(
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

            for _, row in plot_df.iterrows():
                ax.annotate(
                    row["Actor"],
                    (
                        row["Intra-Actor_Diversity_Score"],
                        row["Inter-Actor_Similarity_Score"],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=8 if not is_base else 10,
                )

            ax.set_xlabel("1 - Intra-Actor Similarity", fontsize=14)
            ax.set_ylabel("Inter-Actor Similarity", fontsize=14)
            ax.set_title(f"{language_code}", fontweight="bold", fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.2, 1.0)
            ax.set_ylim(0, 0.8)

            return scatter
        return None

    base_data = all_similarities_data["Base"]
    other_languages = [lang for lang in language_codes if lang != "Base"]

    all_reason_scores = []
    for lang_code in language_codes:
        reason_similarities = all_similarities_data[lang_code]["reason_similarities"]
        for actor, data in reason_similarities.items():
            all_reason_scores.append(data["mean_similarity"])

    global_min = min(all_reason_scores)
    global_max = max(all_reason_scores)
    norm = Normalize(vmin=global_min, vmax=global_max)

    fig1, ax_base = plt.subplots(figsize=(12, 8))
    scatter_base = create_single_plot(ax_base, "Base", base_data, norm, is_base=True)

    if scatter_base:
        fig1.colorbar(scatter_base, ax=ax_base, label="Reason Consistency Score")

    plt.suptitle(
        "Intra-Actor Similarity vs. Inter-Actor Similarity Analysis (Color = Reason-wise Similarity)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    fig2, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = axes.flatten()

    scatter_last = None
    for i, language_code in enumerate(other_languages):
        ax = axes_flat[i]
        scatter = create_single_plot(
            ax, language_code, all_similarities_data[language_code], norm, is_base=False
        )
        if scatter is not None:
            scatter_last = scatter

    plt.suptitle(
        "Intra-Actor Similarity vs. Inter-Actor Similarity Analysis (Color = Reason-wise similarity)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    if scatter_last:
        cbar_ax = fig2.add_axes([0.93, 0.15, 0.02, 0.7])
        fig2.colorbar(scatter_last, cax=cbar_ax, label="Reason Similarity Score")

    plt.show()

    return fig1, fig2


def extract_llm_human_similarities_for_anova(
    all_embeddings_dict: Dict[str, Dict[str, np.ndarray]],
    language_codes: List[str],
) -> pd.DataFrame:
    """Extract all LLM vs human similarity pairs for ANOVA analysis.

    Creates a table with all reasoning comparisons from each model for each scenario,
    allowing statistical comparison of how different models and languages compare to humans.
    All reasonings from the same model are treated as samples from that model.

    Args:
        all_embeddings_dict: Dictionary mapping language codes to embedding dictionaries
        language_codes: List of language codes to process

    Returns:
        DataFrame with columns: actor1, actor2, language, similarity, scenario_idx
        where actor1 is the model name (e.g. "claude"), actor2 is always "human"
    """

    results = []

    for language_code in tqdm(language_codes, desc="Processing languages"):
        embeddings_dict = all_embeddings_dict[language_code]
        actors, reason_types = identify_actors_and_reasons(embeddings_dict)

        # Get human embedding
        if "top_comment_embedding" not in embeddings_dict:
            continue
        human_embeddings = embeddings_dict["top_comment_embedding"]
        n_scenarios = human_embeddings.shape[0]

        # Process each LLM actor
        for actor in actors:
            if actor == "redditor":
                continue

            # Get all available reasoning types for this actor
            for reason_type in reason_types:
                col_name = f"{actor}_{reason_type}_embedding"
                if col_name not in embeddings_dict:
                    continue

                llm_embeddings = embeddings_dict[col_name]

                # Compare each scenario
                for scenario_idx in range(n_scenarios):
                    llm_emb = llm_embeddings[scenario_idx].reshape(1, -1)
                    human_emb = human_embeddings[scenario_idx].reshape(1, -1)

                    llm_emb_norm = normalize(llm_emb, norm="l2")
                    human_emb_norm = normalize(human_emb, norm="l2")

                    similarity = cosine_similarity(llm_emb_norm, human_emb_norm)[0, 0]

                    results.append(
                        {
                            "actor1": actor,
                            "actor2": "human",
                            "language": language_code,
                            "similarity": float(similarity),
                            "scenario_idx": scenario_idx,
                        }
                    )

    return pd.DataFrame(results)
