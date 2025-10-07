# Moral Dilemma Analysis: LLM Response Comparison

This project investigates how different Large Language Models (LLMs) respond to everyday moral dilemmas sourced from Reddit's "Am I the Asshole" (AITA) community. The research analyzes the consistency, diversity, and semantic similarities between AI models' moral reasoning.

## Project Overview

The study evaluates 7 different LLMs (GPT-3.5, GPT-4, Claude, PaLM 2 Bison, Llama 2, Mistral, and Gemma) on 10,826 real-world moral dilemmas. Each model provides multiple reasoning outputs for each dilemma, which are then analyzed using semantic embeddings to understand inter-model agreement and reasoning patterns.

## Scripts Description

### 1. `scripts/process_data.py`

**Purpose**: Data preprocessing and cleaning of the normative evaluation dataset.

**What it does**:
- Removes unnecessary columns from the raw dataset
- Filters out rows with missing reasoning responses

**Input**: `data/ethical_dilemmas.csv`  
**Output**: `data/ethical_dilemmas_cleaned.csv`

### 2. `scripts/generate_embeddings.py`

**Purpose**: Convert text reasoning outputs into semantic vector representations using sentence transformers.

**What it does**:
- Uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate 768-dimensional embeddings
- Processes multiple text columns including:
  - Original Reddit posts (`selftext`)
  - Top comments (`top_comment`)
  - LLM reasoning outputs for each model (e.g., `gpt3.5_reason_1`, `claude_reason_2`, etc.)

**Input**: `data/ethical_dilemmas_cleaned.csv`  
**Output**: `data/embeddings.csv` (1.5GB file with embeddings)

### 3. `scripts/embedding_analysis.py`

**Purpose**: Comprehensive analysis of semantic similarities between different models' moral reasoning.

**What it does**:
- **Scenario-wise analysis**: Compares how different models respond to the same moral dilemma
- **Model-wise analysis**: Compares different scenarios for the same model to assess consistency
- **Reason-wise analysis**: Compares different reasoning versions from the same model on the same scenario

**Outputs**:
- Distribution plots of similarity scores
- Comparison matrices between model pairs
- Consistency vs. diversity analysis charts
- Statistical summaries of model behavior

## Dataset

The project uses data from Reddit's AITA community (October 2022 - March 2023), containing:
- **Original dataset**: 13,205 submissions, 531,813 comments
- **Filtered dataset**: 10,826 submissions, 476,183 comments
- **LLM responses**: Multiple reasoning outputs from 7 different models
- **Embeddings**: 768-dimensional vectors for semantic analysis

## Results

The analysis generates various visualizations and statistical measures stored in `results/similarities/`, including similarity distributions, model comparison matrices, and consistency analyses that provide insights into AI moral reasoning patterns in the embeddings space. 