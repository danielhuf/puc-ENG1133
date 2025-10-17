# %% [markdown]
### Embedding Similarity Analysis
#
# This notebook analyzes similarities between the embeddings of the ethical dilemma dataset in the german language in three main ways:
# 1. **Scenario-wise analysis**: Compare different actors' responses to the same scenario (ethical dilemma)
# 2. **Actor-wise analysis**: Compare a same actor's responses to different scenarios
# 3. **Reason-wise analysis**: Compare different reasoning versions for a same actor in the same scenario
#
# The actors considered for this analysis are:
# - **LLM Models**: GPT-3.5, GPT-4, Claude Haiku, Gemini 2, Gemma 7B, Mistral 7B, and Llama 2.
# - **Human Redditors**: The author of the top comment of each scenario submission.
# %% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
df, embeddings_dict = load_embeddings("../data/embeddings_de.csv")

# %% Identify actors and reason types
actors, reason_types = identify_actors_and_reasons(embeddings_dict)

# %% [markdown]
# ## 1. Scenario-wise Analysis
#
# This analysis compares how human redditors and LLM models respond to the same ethical dilemma.
# For each scenario (row), embedding similarities are calculated between all pairs of actors.


# %% Scenario-wise similarity analysis
cache_path = Path("../results/de/row_similarities.pkl")
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
results_dir = Path("../results/de")
results_dir.mkdir(exist_ok=True)

row_summary_dict = row_summary_df.to_dict("records")
with open(results_dir / "scenario_wise_analysis_results.json", "w") as f:
    json.dump(row_summary_dict, f, indent=2)
print(
    f"Scenario-wise analysis results saved to {results_dir / 'scenario_wise_analysis_results.json'}"
)

# %% Display LLM-Human similarity edge cases
df_cleaned = pd.read_csv("../data/ethical_dilemmas_cleaned_de.csv")
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
