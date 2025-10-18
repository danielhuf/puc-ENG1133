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

# %% [markdown]
# ## 3. Reason-wise Analysis
#
# This analysis compares how consistent each actor's reasonings are when answering the same ethical dilemma.

# %% Reason-wise similarity analysis
cache_path = Path("../results/de/reason_similarities.pkl")
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
# This analysis examines embedding similarities across all available reasoning types for 7 LLM actors and human responses on ethical scenarios in German, revealing distinct behavioral patterns in ethical decision-making.
#
# ### Key Findings:
#
# 1. **Inter-Actor Agreement** (LLM-to-LLM similarity range: 9.5% - 55.9%):
#    - **GPT-3.5** and **GPT-4** show highest agreement with other LLMs (55.9% similarity)
#    - **Gemma** shows lowest agreement with other LLMs (9.5% with Llama, 17.4% with Mistral)
#    - LLM actors generally show low to moderate consensus (mean ~25%) across ethical scenarios
#    - **Human responses** show much lower agreement with LLMs (3.7% - 31.0% similarity)
#    - Human-LLM alignment is consistently lower than inter-LLM agreement, indicating distinct reasoning patterns
#
# 2. **Intra-Actor Agreement** (Range: 27.1% - 44.8%):
#    - **Gemma** shows highest intra-actor agreement (44.8% ± 9.4%) - most predictable across scenarios
#    - **Mistral** responses are least internally consistent (27.1% ± 12.7%) - most context-dependent
#    - **Human** shows moderate intra-actor agreement (29.9% ± 9.0%) - balanced variability
#    - **GPT-4** shows moderate intra-actor agreement (42.7% ± 8.5%) - consistent across scenarios
#    - Internal agreement range of 17.7% indicates moderate diversity in actor response patterns
#
# 3. **Reason-wise Consistency** (Range: 29.2% - 100%):
#    - **Human** shows perfect reasoning consistency (100%) - single reasoning approach per scenario
#    - **Llama** shows highest LLM reasoning coherence (72.2% ± 15.3%) - most consistent across reasoning approaches
#    - **Mistral** shows lowest reasoning coherence (29.2% ± 19.1%) - most variable across reasoning approaches
#    - Mean LLM reason-wise consistency (59.1% ± 15.8%) much higher than intra-actor consistency
#    - This indicates actors are more consistent within reasoning types than across different scenarios
#
# 4. **Three-Dimensional Actor Profiles**:
#    - **Claude**: Moderate inter-actor agreement (36.8%), moderate intra-actor agreement (44.1%), moderate reasoning consistency (66.0%)
#    - **Gemma**: Low inter-actor agreement (24.4%), highest intra-actor agreement (44.8%), low reasoning consistency (38.3%)
#    - **GPT-4**: Moderate inter-actor agreement (37.9%), moderate intra-actor agreement (42.7%), moderate reasoning consistency (63.8%)
#    - **Llama**: Low inter-actor agreement (14.5%), moderate intra-actor agreement (42.2%), highest reasoning consistency (72.2%)
#    - **Human**: Low inter-actor agreement (24.8%), moderate intra-actor agreement (29.9%), perfect reasoning consistency (100%)
#
# 5. **Human-LLM Alignment** (Range: 3.7% - 31.0%):
#    - **Claude** shows highest human alignment (31.0% ± 9.9%) - most human-like reasoning
#    - **Llama** shows lowest human alignment (3.7% ± 7.6%) - least human-like reasoning
#    - **GPT-3.5** shows moderate human alignment (31.7% ± 9.5%) - balanced human similarity
#    - Range of 27.3% indicates high variation in human alignment across LLMs
#    - All LLMs show substantial variability (7.6% - 9.9% std), suggesting context-dependent human alignment
#
# ### Practical Implications:
#
# - **Most Predictable Ethics**: Gemma (highest internal consistency across scenarios)
# - **Most Coherent Reasoning**: Llama (most consistent across different reasoning approaches)
# - **Most Diverse Perspectives**: Mistral and Human (high variability in responses)
# - **Most Human-Like Reasoning**: Claude and GPT-3.5 (highest human alignment at ~31%)
# - **Best Overall Balance**: GPT-4 (reliable consensus + coherent reasoning + moderate diversity)
# - **Best for Human Collaboration**: Claude and GPT-3.5 (highest human alignment scores)
# - **Most Independent Reasoning**: Llama and Mistral (lowest inter-actor alignment, most unique perspectives)
# - **Most Context-Dependent**: Mistral responses (lowest internal consistency, suggesting high situational awareness)
