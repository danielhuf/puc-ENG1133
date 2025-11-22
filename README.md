# Moral Dilemma Analysis: LLM Response Comparison

This project investigates how different Large Language Models (LLMs) respond to everyday moral dilemmas sourced from Reddit's "Am I the Asshole" (AITA) community across five languages. The research analyzes the consistency, diversity, and semantic similarities between AI models' moral reasoning compared to human judgments.

## Overview

The project scrapes moral dilemmas from AITA subreddits in English (base), Portuguese, German, Spanish, and French. Multiple LLM models (GPT-3.5, GPT-4, Claude Haiku, Gemini 2, Llama 2, Mistral 7B, Gemma 7B, PaLM 2 Bison) are prompted to judge each dilemma and provide reasoning. Their responses are compared to human Redditor judgments through three types of analysis: verdict agreement, embedding similarity, and statistical tests.

## Pipeline

1. **Data Collection** (`reddit_scraper_international.py`): Scrapes submissions and top comments from AITA subreddits in multiple languages.

2. **Data Cleaning** (`data_cleaning.py`, `data_cleaning_international.py`): Filters submissions by score and text length, removes bot comments, and extracts top-rated human responses.

3. **LLM Prompting** (`llm_prompting_international.py`): Sends moral dilemmas to various LLM models and collects their verdicts (YTA/NTA/ESH/NAH/INFO) and reasoning in the appropriate language.

4. **Response Cleaning** (`llm_cleaning_international.py`): Cleans LLM responses by removing formatting artifacts, extracting verdicts, and standardizing labels across languages.

5. **Embedding Generation** (`generate_embeddings.py`, `generate_embeddings_international.py`): Creates semantic embeddings for all responses using Sentence Transformers (all-MiniLM-L6-v2).

6. **Embedding Analysis** (`embedding_analysis.py`, `embedding_utils.py`): Computes three types of similarity:

   - **Scenario-wise**: How different actors respond to the same dilemma
   - **Actor-wise**: How each actor responds across different dilemmas
   - **Reason-wise**: How consistent each actor's multiple reasoning approaches are

7. **Verdict Analysis** (`verdict_analysis.py`): Analyzes verdict distributions and calculates inter-actor agreement using Krippendorff's alpha coefficient.

8. **Statistical Tests** (`anova_analysis.py`, `statistical_tests.py`): Performs two-way ANOVA (Actor × Language) and post-hoc tests (Tukey's HSD, Games-Howell) to identify significant differences in LLM-human similarity across models and languages.
