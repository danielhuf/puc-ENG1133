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
    analyze_reason_similarities,
    plot_reason_similarity_comparison,
    summarize_reason_characteristics,
    cross_analyze_actor_similarity,
)


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
plot_reason_similarity_comparison(reason_similarities)


# %% Statistical summary of reason-wise characteristics
reason_summary_df = summarize_reason_characteristics(reason_similarities)

# %% Save reason-wise analysis results
reason_summary_dict = reason_summary_df.to_dict("records")
with open(results_dir / "reason_wise_analysis_results.json", "w") as f:
    json.dump(reason_summary_dict, f, indent=2)
print(
    f"Reason-wise analysis results saved to {results_dir / 'reason_wise_analysis_results.json'}"
)


# %% Cross-analysis: Intra-Actor similarity vs. inter-Actor similarity
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
