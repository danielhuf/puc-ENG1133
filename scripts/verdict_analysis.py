# %% [markdown]
# ## 4. Verdict-wise Analysis
#
# This analyzes verdicts given by the LLM actors and human redditors to the moral dilemmas.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
    Returns a DataFrame with actors as rows and verdicts as columns.
    """
    verdicts = ["NTA", "YTA", "NAH", "ESH", "INFO"]
    results = {}

    for actor, cols in actor_dict.items():
        all_verdicts = pd.Series(dtype=object)
        for col in cols:
            all_verdicts = pd.concat([all_verdicts, df[col].dropna()])

        all_verdicts = all_verdicts.replace("INF", "INFO")

        verdict_counts = all_verdicts.value_counts()
        total = len(all_verdicts)

        fractions = {}
        for verdict in verdicts:
            fractions[verdict] = (
                verdict_counts.get(verdict, 0) / total if total > 0 else 0
            )

        results[actor] = fractions

    return pd.DataFrame(results).T


# %%
def plot_verdict_distribution(fractions_df, title):
    """
    Create a grouped bar chart showing verdict distribution for each actor.
    """
    actor_name_map = {
        "reddit": "Redditor",
        "gpt3.5": "GPT-3.5",
        "gpt4": "GPT-4",
        "claude": "Claude Haiku",
        "bison": "PaLM 2 Bison",
        "llama": "Llama 2 7B",
        "mistral": "Mistral 7B",
        "gemma": "Gemma 7B",
        "gemini": "Gemini 2",
    }

    fractions_df.index = [
        actor_name_map.get(actor, actor.upper()) for actor in fractions_df.index
    ]

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

    for i, (actor, row) in enumerate(fractions_df.iterrows()):
        values = [row[verdict] for verdict in verdicts]
        position = x + (i * width) - offset
        _ = ax.bar(position, values, width, label=actor, color=colors[i % len(colors)])

        errors = [0.01] * len(verdicts)
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

    fractions_df = calculate_verdict_fractions(df, actor_dict)
    print(f"\nVerdict fractions:\n{fractions_df}")

    title = f"Verdict Distribution - {language}"
    plot_verdict_distribution(fractions_df, title)

# %%
