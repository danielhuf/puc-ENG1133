# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import krippendorff
from itertools import combinations
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# %%
def extract_actor_labels(df):
    """
    Extract actor names and ALL their label columns from the dataframe.
    Returns a dict mapping actor name to a list of their label column names.
    """
    allowed_actors = [
        "reddit",
        "gpt3.5",
        "gpt4",
        "claude",
        "bison",
        "llama",
        "mistral",
        "gemma",
        "gemini",
    ]

    actor_dict = {}

    for actor in allowed_actors:
        actor_cols = [col for col in df.columns if col.startswith(f"{actor}_label")]
        if actor_cols:
            actor_dict[actor] = actor_cols

    return actor_dict


# %%
def calculate_verdict_fractions(df, actor_dict):
    """
    Calculate the fraction of each verdict for each actor.
    Aggregates counts across all label columns for each actor.
    Returns (fractions_df, counts_df, totals_series).
    """
    verdicts = ["NTA", "YTA", "NAH", "ESH", "INFO"]
    fraction_results = {}
    count_results = {}
    totals = {}

    for actor, cols in actor_dict.items():
        all_verdicts = pd.Series(dtype=object)
        for col in cols:
            all_verdicts = pd.concat([all_verdicts, df[col].dropna()])

        all_verdicts = all_verdicts.replace("INF", "INFO")

        verdict_counts = all_verdicts.value_counts()
        total = len(all_verdicts)

        fractions = {}
        counts = {}
        for verdict in verdicts:
            count = verdict_counts.get(verdict, 0)
            counts[verdict] = count
            fractions[verdict] = count / total if total > 0 else 0

        fraction_results[actor] = fractions
        count_results[actor] = counts
        totals[actor] = total

    fractions_df = pd.DataFrame(fraction_results).T
    counts_df = pd.DataFrame(count_results).T.fillna(0).astype(int)
    totals_series = pd.Series(totals)

    return fractions_df, counts_df, totals_series


# %%
def plot_verdict_distribution(fractions_df, totals_series, title):
    """
    Create a grouped bar chart showing verdict distribution for each actor.
    """
    actor_name_map = {
        "reddit": "Redditor",
        "gpt3.5": "GPT-3.5",
        "gpt4": "GPT-4",
        "claude": "Claude Haiku",
        "bison": "PaLM 2 Bison",
        "gemini": "Gemini 2",
        "llama": "Llama 2 7B",
        "mistral": "Mistral 7B",
        "gemma": "Gemma 7B",
    }

    ordered_actors = [
        actor for actor in actor_name_map.keys() if actor in fractions_df.index
    ]
    if ordered_actors:
        fractions_df = fractions_df.loc[ordered_actors]
        totals_series = totals_series.loc[ordered_actors]

    actor_keys = list(fractions_df.index)
    display_index = [actor_name_map.get(actor, actor.upper()) for actor in actor_keys]
    fractions_display = fractions_df.copy()
    fractions_display.index = display_index

    _, ax = plt.subplots(figsize=(12, 6))

    verdicts = ["NTA", "YTA", "NAH", "ESH", "INFO"]
    colors = [
        "#3274A1",
        "#E1812C",
        "#3A923A",
        "#C03D3E",
        "#9372B2",
        "#8B4513",
        "#FF69B4",
        "#808080",
    ]

    x = np.arange(len(verdicts))
    width = 0.1
    n_actors = len(fractions_df)

    offset = width * n_actors / 2 - width / 2

    for i, actor_key in enumerate(actor_keys):
        row = fractions_df.loc[actor_key]
        display_name = fractions_display.index[i]
        values = [row[verdict] for verdict in verdicts]
        total = totals_series.get(actor_key, 0)
        errors = [np.sqrt(p * (1 - p) / total) if total > 0 else 0 for p in values]
        position = x + (i * width) - offset
        _ = ax.bar(
            position,
            values,
            width,
            label=display_name,
            color=colors[i % len(colors)],
        )

        ax.errorbar(
            position,
            values,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=2,
            alpha=0.5,
            linewidth=0.5,
        )

    ax.set_ylabel("Fraction of Posts", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(verdicts, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


# %%
def encode_verdicts(series):
    """
    Encode verdicts as numeric values for Krippendorff's alpha.
    """
    encoding = {"NTA": 0, "YTA": 1, "NAH": 2, "ESH": 3, "INFO": 4, "INF": 4}
    return series.map(encoding)


# %%
def calculate_krippendorff_alpha(df, col1, col2):
    """
    Calculate Krippendorff's alpha between two label columns.
    Returns alpha value or np.nan if calculation fails.
    """
    data1 = encode_verdicts(df[col1])
    data2 = encode_verdicts(df[col2])

    valid_mask = data1.notna() & data2.notna()
    if valid_mask.sum() < 2:
        return np.nan

    reliability_data = np.array([data1.values, data2.values])

    try:
        alpha = krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="nominal"
        )
        return alpha
    except Exception as e:
        print(f"  Warning: Could not calculate alpha for {col1} vs {col2}: {e}")
        return np.nan


# %%
def calculate_pairwise_alpha_average(df, actor1_cols, actor2_cols):
    """
    Calculate Krippendorff's alpha for all pairs of columns between two actors
    and return the average.
    """
    alphas = []

    for col1 in actor1_cols:
        for col2 in actor2_cols:
            if col1 == col2:
                continue

            alpha = calculate_krippendorff_alpha(df, col1, col2)
            if not np.isnan(alpha):
                alphas.append(alpha)

    return np.mean(alphas) if alphas else np.nan


# %%
def calculate_diagonal_alpha(df, actor_cols):
    """
    Calculate average Krippendorff's alpha between all pairs within the same actor.
    This measures intra-rater reliability.
    """
    if len(actor_cols) < 2:
        return 1.0

    alphas = []
    for col1, col2 in combinations(actor_cols, 2):
        alpha = calculate_krippendorff_alpha(df, col1, col2)
        if not np.isnan(alpha):
            alphas.append(alpha)

    return np.mean(alphas) if alphas else np.nan


# %%
def build_krippendorff_matrix(df, actor_dict):
    """
    Build a matrix of Krippendorff's alpha values for all actor pairs.
    """
    actors = list(actor_dict.keys())
    n_actors = len(actors)
    alpha_matrix = np.zeros((n_actors, n_actors))

    for i, actor1 in enumerate(actors):
        for j, actor2 in enumerate(actors):
            if i == j:
                alpha_matrix[i, j] = calculate_diagonal_alpha(df, actor_dict[actor1])
            else:
                alpha_matrix[i, j] = calculate_pairwise_alpha_average(
                    df, actor_dict[actor1], actor_dict[actor2]
                )

    return pd.DataFrame(alpha_matrix, index=actors, columns=actors)


# %%
def plot_krippendorff_heatmap(alpha_df, title):
    """
    Create a heatmap of Krippendorff's alpha values.
    """
    actor_name_map = {
        "reddit": "Redditor",
        "gpt3.5": "GPT-3.5",
        "gpt4": "GPT-4",
        "claude": "Claude Haiku",
        "bison": "PaLM 2 Bison",
        "gemini": "Gemini 2",
        "llama": "Llama 2 7B",
        "mistral": "Mistral 7B",
        "gemma": "Gemma 7B",
    }

    ordered_actors = [
        actor for actor in actor_name_map.keys() if actor in alpha_df.index
    ]
    if ordered_actors:
        alpha_df = alpha_df.loc[ordered_actors, ordered_actors]

    alpha_df.index = [
        actor_name_map.get(actor, actor.upper()) for actor in alpha_df.index
    ]
    alpha_df.columns = [
        actor_name_map.get(actor, actor.upper()) for actor in alpha_df.columns
    ]

    plt.figure(figsize=(10, 8))

    colors = ["#A50026", "#FFFFFF", "#000000"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    sns.heatmap(
        alpha_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Krippendorff's Alpha", "shrink": 0.8},
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()


# %%
datasets = [
    ("moral_dilemmas_cleaned.csv", "English (Base)"),
    ("moral_dilemmas_cleaned_br.csv", "Portuguese"),
    ("moral_dilemmas_cleaned_de.csv", "German"),
    ("moral_dilemmas_cleaned_fr.csv", "French"),
    ("moral_dilemmas_cleaned_es.csv", "Spanish"),
]

data_dir = Path("../data")

for filename, language in datasets:
    print(f"\n{'='*60}")
    print(f"{language}")
    print(f"{'='*60}")

    df = pd.read_csv(data_dir / filename)

    actor_dict = extract_actor_labels(df)

    fractions_df, counts_df, totals_series = calculate_verdict_fractions(df, actor_dict)
    print(f"\nVerdict fractions:\n{fractions_df}")
    print(f"\nVerdict counts:\n{counts_df}")

    title = f"Verdict Distribution - {language}"
    plot_verdict_distribution(fractions_df, totals_series, title)

    alpha_df = build_krippendorff_matrix(df, actor_dict)
    print(f"\nKrippendorff's Alpha Matrix:\n{alpha_df}")

    title = f"Inter-Model Agreement - {language}"
    plot_krippendorff_heatmap(alpha_df, title)

# %%
