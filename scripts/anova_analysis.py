"""
ANOVA Analysis of LLM vs Human Similarities

This script performs ANOVA to compare the variance in similarity scores
between different LLM models and humans across multiple languages.
"""

# %%
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols  # type: ignore
from statsmodels.stats.anova import anova_lm  # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from embedding_utils import (
    load_embeddings,
    extract_llm_human_similarities_for_anova,
)


# %%
all_embeddings = {
    "Base": load_embeddings("../data/embeddings.csv"),
    "Portuguese": load_embeddings("../data/embeddings_br.csv"),
    "German": load_embeddings("../data/embeddings_de.csv"),
    "Spanish": load_embeddings("../data/embeddings_es.csv"),
    "French": load_embeddings("../data/embeddings_fr.csv"),
}

language_codes = ["Base", "Portuguese", "German", "Spanish", "French"]

# %%
df_anova = extract_llm_human_similarities_for_anova(all_embeddings, language_codes)
df_anova = df_anova[
    ~df_anova["actor1"].isin(["gemini", "bison"])
]  # Remove Gemini and Bison from analysis

# %%
output_file = "../results/anova_data.csv"
df_anova.to_csv(output_file, index=False)

# %%
print("\n=== DESCRIPTIVE STATISTICS ===")
print("\nBy Actor (LLM + reasoning type):")
print(df_anova.groupby("actor1")["similarity"].describe())

print("\nBy Language:")
print(df_anova.groupby("language")["similarity"].describe())

# # %%
# print("\n=== ONE-WAY ANOVA: Comparing Actors ===")
# actor_model = ols("similarity ~ C(actor1)", data=df_anova).fit()
# actor_anova_table = anova_lm(actor_model, typ=2)
# print(actor_anova_table)

# # %%
# print("\n=== ONE-WAY ANOVA: Comparing Languages ===")
# language_model = ols("similarity ~ C(language)", data=df_anova).fit()
# language_anova_table = anova_lm(language_model, typ=2)
# print(language_anova_table)

# %%
print("\n=== TWO-WAY ANOVA: Actor × Language ===")
model = ols("similarity ~ C(actor1) * C(language)", data=df_anova).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# %%
print("\n=== POST-HOC TEST: Tukey's HSD ===")
tukey_results = pairwise_tukeyhsd(
    df_anova["similarity"],
    df_anova["actor1"] + "_" + df_anova["language"],
    alpha=0.005,
    use_var="unequal",
)

tukey_df = pd.DataFrame(
    tukey_results._results_table.data[1:],  # type: ignore[attr-defined]
    columns=tukey_results._results_table.data[0],  # type: ignore[attr-defined]
)
tukey_df[["model1", "language1"]] = tukey_df["group1"].str.split("_", n=1, expand=True)
tukey_df[["model2", "language2"]] = tukey_df["group2"].str.split("_", n=1, expand=True)
filtered_tukey = tukey_df[
    (tukey_df["model1"] == tukey_df["model2"])
    | (tukey_df["language1"] == tukey_df["language2"])
]
print(
    filtered_tukey[
        ["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]
    ].to_string(index=False)
)

# %%
