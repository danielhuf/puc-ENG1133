"""
Common utilities for embedding similarity analysis.

This module contains shared functions used by the embedding analysis notebooks
to avoid code duplication and ensure consistency across different language analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Tuple, Dict
from tqdm import tqdm


def load_embeddings(embeddings_file: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load embeddings from CSV and organize them by column.

    Args:
        embeddings_file: Path to the CSV file containing embeddings

    Returns:
        Tuple of (DataFrame, embeddings_dict) where embeddings_dict maps column names to numpy arrays
    """
    df = pd.read_csv(embeddings_file)

    embedding_cols = [col for col in df.columns if col.endswith("_embedding")]

    embeddings_dict = {}
    for col in tqdm(embedding_cols, desc="Processing embedding columns"):

        def parse_embedding(x):
            """Convert string representation to numpy array."""
            if pd.isna(x):
                return np.zeros(384, dtype=np.float32)
            return np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32)

        embeddings = df[col].apply(parse_embedding).values

        embeddings_array = np.vstack(embeddings)
        embeddings_dict[col] = embeddings_array

    return df, embeddings_dict


def analyze_row_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze inter-actor similarity on the same ethical scenarios.

    This function compares how different LLM models and human redditors respond to a same ethical dilemma.
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
                actor1_reasons = actor_reason_combinations[actor1]
                actor2_reasons = actor_reason_combinations[actor2]

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

    return actors, reason_types


def plot_row_similarity_distribution(row_similarities: Dict):
    """Plot distribution of similarities across scenarios.

    Displays mean similarities across all available reason type combinations
    for each actor pair.
    """

    pair_similarities = {}
    for row_data in row_similarities.values():
        for pair, sim in row_data.items():
            if pair not in pair_similarities:
                pair_similarities[pair] = []
            pair_similarities[pair].append(sim)

    actors = set()
    for pair in pair_similarities.keys():
        actor1, actor2 = pair.split("_vs_")
        actors.add(actor1)
        actors.add(actor2)
    actors = sorted(list(actors))

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
                        color="skyblue",
                        edgecolor="black",
                        density=True,
                    )

                    mean_sim = np.mean(similarities)
                    median_sim = np.median(similarities)

                    ax.axvline(
                        mean_sim, color="red", linestyle="--", alpha=0.8, linewidth=1
                    )
                    ax.axvline(
                        median_sim, color="blue", linestyle="-", alpha=0.8, linewidth=1
                    )

                    ax.set_title(
                        f"{actors[j]} vs {actors[i+1]}\nMean: {mean_sim:.3f}",
                        fontsize=12,
                        pad=4,
                    )
                    ax.set_xlabel("Similarity", fontsize=10)
                    ax.set_ylabel("Density", fontsize=10)
                    ax.tick_params(labelsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)

    for i in range(n_actors - 1):
        axes[i, 0].set_ylabel(actors[i + 1].upper(), fontsize=14, fontweight="bold")
        axes[-1, i].set_xlabel(actors[i].upper(), fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.suptitle(
        "Distribution of Scenario-wise Similarities Between Actors",
        y=-0.02,
        fontsize=24,
        fontweight="bold",
    )
    plt.show()


def summarize_row_characteristics(row_similarities: Dict):
    """Provide statistical summary of inter-actor similarity patterns."""

    print(f"=== SCENARIO-WISE SIMILARITY SUMMARY ===\n")

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


def display_edge_llm_human_similarities(
    row_similarities: Dict, df_cleaned: pd.DataFrame
):
    """Display the top 5 answers with highest and lowest LLM-Human similarity."""

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

    llm_human_similarities.sort(key=lambda x: x["similarity"])

    lowest_5 = llm_human_similarities[:5]
    highest_5 = llm_human_similarities[-5:]

    print("=" * 80)
    print("EDGE LLM-HUMAN SIMILARITY CASES")
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
        print(f"   Human Comment: {human_comment}")
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
        print(f"   Human Comment: {human_comment}")
        print(f"   {llm_model.upper()} Reasoning: {llm_reason}")

    return


def analyze_column_similarities(
    embeddings_dict: Dict[str, np.ndarray],
    actors: List[str],
    reason_types: List[str],
) -> Dict:
    """Analyze intra-actor similarity across different ethical scenarios.

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


def plot_column_similarity_comparison(column_similarities: Dict):
    """Compare intra-actor similarity distributions."""

    actor_names = list(column_similarities.keys())
    n_actors = len(actor_names)

    height_ratios = [2.5] * n_actors + [3]

    _, axes = plt.subplots(
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


def summarize_column_characteristics(column_similarities: Dict):
    """Provide statistical summary of each actor's characteristics."""

    print(f"=== ACTOR-WISE SIMILARITY SUMMARY ===\n")

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
):
    """Display the top 5 answers with highest and lowest scenario similarity for both humans and LLMs."""

    actor_scenario_similarities = {}

    for actor_idx, actor in enumerate(actors, 1):
        print(f"Processing actor {actor_idx}/{len(actors)}: {actor}")
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

    human_data = actor_scenario_similarities.get("human", {"lowest": [], "highest": []})
    human_lowest_5 = human_data["lowest"]
    human_highest_5 = human_data["highest"]

    all_llm_lowest = []
    all_llm_highest = []

    for actor in actors:
        if actor != "human":
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

    print("=" * 100)
    print("EDGE SCENARIO SIMILARITY CASES")
    print("=" * 100)

    print("\n👥 HUMAN RESPONSES")
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
        print(f"   Human Comment 1: {comment1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")
        print(f"   Human Comment 2: {comment2}")

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
        print(f"   Human Comment 1: {comment1}")
        print(f"   Scenario 2 (ID: {scenario2['submission_id']}): {title2}")
        print(f"   Human Comment 2: {comment2}")

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
