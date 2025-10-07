#!/usr/bin/env python3
import os
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_ALTS_UNTRUSTED"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("google").setLevel(logging.ERROR)

import pandas as pd
import openai
import anthropic
import google.generativeai as genai
import replicate
import re
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration constants for the LLM prompting system."""

    MAX_TOKENS = 500
    DEFAULT_TEMPERATURE = 0.7
    SUPPORTED_LANGUAGES = ["br", "de", "es", "fr"]
    VALID_VERDICTS = ["YTA", "NTA", "ESH", "NAH", "INFO"]

    MODELS = {
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
        "claude": {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
        "gemini": {"provider": "gemini", "model": "gemini-2.0-flash-lite"},
        "llama": {"provider": "replicate", "model": "meta/llama-2-7b-chat"},
        "mistral": {"provider": "replicate", "model": "mistralai/mistral-7b-v0.1"},
        "gemma": {
            "provider": "replicate",
            "model": "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
        },
    }

    COLUMN_PAIRS = [
        ("gpt3.5_label_1", "gpt3.5_reason_1"),
        ("gpt3.5_label_2", "gpt3.5_reason_2"),
        ("gpt4_label_1", "gpt4_reason_1"),
        ("gpt4_label_2", "gpt4_reason_2"),
        ("claude_label_1", "claude_reason_1"),
        ("claude_label_2", "claude_reason_2"),
        ("gemini_label_1", "gemini_reason_1"),
        ("gemini_label_2", "gemini_reason_2"),
        ("llama_label_1", "llama_reason_1"),
        ("llama_label_2", "llama_reason_2"),
        ("mistral_label_1", "mistral_reason_1"),
        ("mistral_label_2", "mistral_reason_2"),
        ("gemma_label_1", "gemma_reason_1"),
        ("gemma_label_2", "gemma_reason_2"),
    ]

    MODEL_COLUMN_MAPPING = {
        "gpt-3.5-turbo": [
            ("gpt3.5_label_1", "gpt3.5_reason_1"),
            ("gpt3.5_label_2", "gpt3.5_reason_2"),
        ],
        "gpt-4o-mini": [
            ("gpt4_label_1", "gpt4_reason_1"),
            ("gpt4_label_2", "gpt4_reason_2"),
        ],
        "claude": [
            ("claude_label_1", "claude_reason_1"),
            ("claude_label_2", "claude_reason_2"),
        ],
        "gemini": [
            ("gemini_label_1", "gemini_reason_1"),
            ("gemini_label_2", "gemini_reason_2"),
        ],
        "llama": [
            ("llama_label_1", "llama_reason_1"),
            ("llama_label_2", "llama_reason_2"),
        ],
        "mistral": [
            ("mistral_label_1", "mistral_reason_1"),
            ("mistral_label_2", "mistral_reason_2"),
        ],
        "gemma": [
            ("gemma_label_1", "gemma_reason_1"),
            ("gemma_label_2", "gemma_reason_2"),
        ],
    }


@dataclass
class ProcessingConfig:
    """Configuration for processing a dataset."""

    language_code: str
    models_to_run: List[str]
    output_file: Optional[str] = None

    def __post_init__(self):
        if self.language_code not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language_code}")

        if not self.output_file:
            self.output_file = f"data/ethical_dilemmas_cleaned_{self.language_code}.csv"


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class APIError(LLMError):
    """Exception for API-related errors."""

    pass


class ConfigurationError(LLMError):
    """Exception for configuration-related errors."""

    pass


def setup_llm_provider(provider: str) -> Union[Any, None]:
    """
    Setup LLM provider API client dynamically.

    Args:
        provider: Provider name ('openai', 'anthropic', 'gemini', 'replicate')

    Returns:
        Configured client or None for providers that don't return clients

    Raises:
        ConfigurationError: If required environment variable is not set
    """
    provider_configs = {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "setup_func": lambda: setattr(
                openai, "api_key", os.getenv("OPENAI_API_KEY")
            ),
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY",
            "setup_func": lambda: anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
        },
        "gemini": {
            "env_var": "GOOGLE_API_KEY",
            "setup_func": lambda: (
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY")),
                genai.GenerativeModel("gemini-2.0-flash-lite"),
            )[-1],
        },
        "replicate": {
            "env_var": "REPLICATE_API_TOKEN",
            "setup_func": lambda: setattr(
                replicate, "api_token", os.getenv("REPLICATE_API_TOKEN")
            ),
        },
    }

    if provider not in provider_configs:
        raise ConfigurationError(
            f"Unknown provider: {provider}. Supported providers: {list(provider_configs.keys())}"
        )

    config = provider_configs[provider]
    api_key = os.getenv(config["env_var"])

    if not api_key:
        raise ConfigurationError(
            f"Please set your {config['env_var']} environment variable"
        )

    return config["setup_func"]()


def setup_all_providers() -> Dict[str, Optional[Any]]:
    """Setup all LLM providers at once."""
    providers = ["openai", "anthropic", "gemini", "replicate"]
    clients = {}

    for provider in providers:
        try:
            client = setup_llm_provider(provider)
            clients[provider] = client
        except ConfigurationError as e:
            print(f"{provider.title()} setup failed: {e}")
            clients[provider] = None

    return clients


def parse_structured_response(
    response_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the structured JSON response from LLM models.

    Args:
        response_text: JSON response from LLM

    Returns:
        Tuple of (verdict, reasoning) or (None, None) if parsing fails
    """
    try:
        if response_text.startswith("ERROR:"):
            print(f"API Error response: {response_text}")
            return None, None

        cleaned_text = response_text.strip()

        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]

        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        cleaned_text = cleaned_text.strip()

        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError:

            reasoning_match = re.search(
                r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_text, re.DOTALL
            )
            if reasoning_match:
                original_reasoning = reasoning_match.group(1)
                escaped_reasoning = (
                    original_reasoning.replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                cleaned_text = cleaned_text.replace(
                    f'"reasoning": "{original_reasoning}"',
                    f'"reasoning": "{escaped_reasoning}"',
                )

            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                return extract_verdict_from_text(cleaned_text)

        verdict = data.get("verdict", "").upper()
        reasoning = data.get("reasoning", "")

        verdict = verdict if verdict in Config.VALID_VERDICTS else None
        reasoning = reasoning if reasoning and len(reasoning.strip()) >= 5 else None

        return verdict, reasoning

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text}")
        print(f"JSON Error: {e}")
        return extract_verdict_from_text(response_text)


def extract_verdict_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract verdict and reasoning from plain text response when JSON parsing fails.

    Args:
        text: Plain text response from LLM

    Returns:
        Tuple of (verdict, reasoning) or (None, None) if extraction fails
    """

    verdict_patterns = [
        r"\b(YTA|NTA|ESH|NAH|INFO)\b",
        r"classify.*?as\s+(YTA|NTA|ESH|NAH|INFO)",
        r"would.*?classify.*?as\s+(YTA|NTA|ESH|NAH|INFO)",
        r"verdict.*?is\s+(YTA|NTA|ESH|NAH|INFO)",
    ]

    verdict = None
    for pattern in verdict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            break

    reasoning = text.strip()

    if verdict:
        reasoning = re.sub(
            r"^(Based on|Here is my reasoning:|Therefore, I would classify.*?\.)",
            "",
            reasoning,
            flags=re.IGNORECASE,
        )
        reasoning = reasoning.strip()

        if len(reasoning) > 1000:
            reasoning = reasoning[:1000] + "..."

    verdict = verdict if verdict in Config.VALID_VERDICTS else None
    reasoning = reasoning if reasoning and len(reasoning.strip()) >= 5 else None

    return verdict, reasoning


def process_model_prediction(
    df: pd.DataFrame,
    idx: int,
    row: pd.Series,
    system_message: str,
    user_message: str,
    label_col: str,
    reason_col: str,
    progress_bar: tqdm,
    pair_name: str,
    model: str,
) -> None:
    """
    Process a single model prediction for a row.

    Args:
        df: DataFrame to update
        idx: Row index
        row: Current row data
        system_message: System prompt for LLM
        user_message: User message
        label_col: Name of label column
        reason_col: Name of reason column
        progress_bar: tqdm progress bar
        pair_name: Name for progress tracking
        model: Model to use

    Returns:
        None (updates dataframe in place)
    """
    if pd.notna(row[label_col]) or pd.notna(row[reason_col]):
        progress_bar.set_postfix(**{pair_name: "skipped"})
    else:
        try:
            prompt_function = MODEL_FUNCTIONS[model]
            response = prompt_function(system_message, user_message)

            verdict, reasoning = parse_structured_response(response)
            df.at[idx, label_col] = verdict
            df.at[idx, reason_col] = reasoning

            if verdict is not None and reasoning is not None:
                progress_bar.set_postfix(**{pair_name: "complete"})
            elif verdict is not None or reasoning is not None:
                progress_bar.set_postfix(**{pair_name: "partial"})
            else:
                progress_bar.set_postfix(**{pair_name: "failed"})
        except APIError as e:
            print(f"API error for {model}: {e}")
            progress_bar.set_postfix(**{pair_name: "api_error"})
        except Exception as e:
            print(f"Unexpected error for {model}: {e}")
            progress_bar.set_postfix(**{pair_name: "error"})


def prompt_gpt(
    system_message: str,
    user_message: str,
    model: str = "gpt-3.5-turbo",
) -> str:
    """
    Send prompt to GPT model and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)
        model: Model to use (gpt-3.5-turbo or gpt-4o-mini)

    Returns:
        Response from the specified GPT model

    Raises:
        APIError: If API call fails
    """
    try:
        setup_llm_provider("openai")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.DEFAULT_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise APIError(f"Error calling OpenAI API: {e}")


def prompt_claude(
    system_message: str,
    user_message: str,
) -> str:
    """
    Send prompt to Claude Haiku 3 and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)

    Returns:
        Response from Claude Haiku 3

    Raises:
        APIError: If API call fails
    """
    try:
        client = setup_llm_provider("anthropic")
        response = client.messages.create(
            model=Config.MODELS["claude"]["model"],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.DEFAULT_TEMPERATURE,
            system=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        raise APIError(f"Error calling Anthropic API: {e}")


def prompt_gemini(
    system_message: str,
    user_message: str,
) -> str:
    """
    Send prompt to Gemini 2.0 Flash Lite and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)

    Returns:
        Response from Gemini 2.0 Flash Lite

    Raises:
        APIError: If API call fails
    """
    try:
        model = setup_llm_provider("gemini")
        prompt = f"{system_message}\n\n{user_message}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=Config.DEFAULT_TEMPERATURE,
                max_output_tokens=Config.MAX_TOKENS,
            ),
        )
        return response.text.strip()
    except Exception as e:
        raise APIError(f"Error calling Google Gemini API: {e}")


def create_gpt_prompt_function(model_name: str) -> Callable[[str, str], str]:
    """Create a wrapper for GPT models to match other model signatures."""

    def wrapper(system_message: str, user_message: str) -> str:
        return prompt_gpt(system_message, user_message, model_name)

    return wrapper


def prompt_replicate(system_message: str, user_message: str, model: str) -> str:
    """
    Send prompt to Replicate model and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)
        model: Model to use (e.g., "meta/llama-2-7b-chat", "mistralai/mistral-7b-v0.1")

    Returns:
        Response from the specified Replicate model

    Raises:
        APIError: If API call fails
    """
    try:
        setup_llm_provider("replicate")

        combined_prompt = f"{system_message}\n\n{user_message}"

        output = replicate.run(
            model,
            input={
                "prompt": combined_prompt,
                "temperature": Config.DEFAULT_TEMPERATURE,
                "max_new_tokens": Config.MAX_TOKENS,
            },
        )

        response_text = ""
        for chunk in output:
            response_text += str(chunk)

        result = response_text.strip()
        return result

    except Exception as e:
        raise APIError(f"Error calling Replicate API: {e}")


MODEL_FUNCTIONS: Dict[str, Callable[[str, str], str]] = {
    "gpt-3.5-turbo": create_gpt_prompt_function("gpt-3.5-turbo"),
    "gpt-4o-mini": create_gpt_prompt_function("gpt-4o-mini"),
    "claude": prompt_claude,
    "gemini": prompt_gemini,
    "llama": lambda sys, user: prompt_replicate(sys, user, "meta/llama-2-7b-chat"),
    "mistral": lambda sys, user: prompt_replicate(
        sys, user, "mistralai/mistral-7b-v0.1"
    ),
    "gemma": lambda sys, user: prompt_replicate(
        sys,
        user,
        "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
    ),
}


def add_column_if_not_exists(df: pd.DataFrame, column_name: str) -> None:
    """
    Add a column to the DataFrame if it doesn't already exist.

    Args:
        df: DataFrame to modify
        column_name: Name of the column to add
    """
    if column_name not in df.columns:
        df[column_name] = pd.Series(dtype="object")
    else:
        df[column_name] = df[column_name].astype("object")


def add_all_required_columns(df: pd.DataFrame) -> None:
    """
    Add all required columns for model predictions to the DataFrame.

    Args:
        df: DataFrame to modify
    """
    for label_col, reason_col in Config.COLUMN_PAIRS:
        add_column_if_not_exists(df, label_col)
        add_column_if_not_exists(df, reason_col)


def get_system_message(language_code: str) -> str:
    """
    Get the system message in the appropriate language.

    Args:
        language_code: Language code (br, de, es, fr)

    Returns:
        System message in the specified language

    Raises:
        ConfigurationError: If language code is not supported or config file is missing
    """
    config_path = Path(__file__).parent.parent / "config" / "system_messages.json"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            system_messages = json.load(f)
    except FileNotFoundError:
        raise ConfigurationError(
            f"System messages config file not found: {config_path}"
        )
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in system messages config: {e}")

    if language_code not in system_messages:
        raise ConfigurationError(f"Unsupported language code: {language_code}")

    return system_messages[language_code]


def load_dataset(language_code: str) -> pd.DataFrame:
    """
    Load dataset for the specified language.

    Args:
        language_code: Language code (br, de, es, fr)

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If dataset file is not found
    """
    file_path = f"data/ethical_dilemmas_cleaned_{language_code}.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")


def process_models_for_row(
    df: pd.DataFrame,
    idx: int,
    row: pd.Series,
    system_message: str,
    user_message: str,
    progress_bar: tqdm,
    models_to_run: List[str],
) -> None:
    """
    Process specified models for a single row.

    Args:
        df: DataFrame to update
        idx: Row index
        row: Current row data
        system_message: System prompt for LLM
        user_message: User message
        progress_bar: tqdm progress bar
        models_to_run: List of models to process
    """
    for model in models_to_run:
        if model not in Config.MODEL_COLUMN_MAPPING:
            print(f"Warning: Unknown model {model}, skipping")
            continue

        column_pairs = Config.MODEL_COLUMN_MAPPING[model]

        for i, (label_col, reason_col) in enumerate(column_pairs, 1):
            pair_name = f"{model}_pair{i}"
            try:
                process_model_prediction(
                    df,
                    idx,
                    row,
                    system_message,
                    user_message,
                    label_col,
                    reason_col,
                    progress_bar,
                    pair_name,
                    model,
                )
            except APIError as e:
                print(f"API error for {model}: {e}")
                progress_bar.set_postfix(**{pair_name: "api_error"})
            except Exception as e:
                print(f"Unexpected error for {model}: {e}")
                progress_bar.set_postfix(**{pair_name: "error"})


def process_dataset(config: ProcessingConfig) -> None:
    """
    Process a single dataset with the specified configuration.

    Args:
        config: Processing configuration
    """
    try:
        df = load_dataset(config.language_code)
        print(f"Processing {config.language_code.upper()} dataset...")

        add_all_required_columns(df)

        system_message = get_system_message(config.language_code)

        progress_bar = tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {config.language_code.upper()}",
        )

        for idx, row in progress_bar:
            user_message = str(row["selftext"])

            process_models_for_row(
                df,
                idx,
                row,
                system_message,
                user_message,
                progress_bar,
                config.models_to_run,
            )

            df.to_csv(config.output_file, index=False)

        print(f"Completed processing {config.language_code.upper()} dataset")

    except Exception as e:
        print(f"Error processing {config.language_code.upper()} dataset: {e}")
        raise


def main() -> None:
    """Main function to process all datasets."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    models_to_run = ["gpt-3.5-turbo", "gpt-4o-mini", "claude", "gemini"]

    for language_code in Config.SUPPORTED_LANGUAGES:
        try:
            config = ProcessingConfig(
                language_code=language_code,
                models_to_run=models_to_run,
            )
            process_dataset(config)
        except Exception as e:
            print(f"Error processing {language_code.upper()} dataset: {e}")
            continue

    print("All datasets processed successfully!")


if __name__ == "__main__":
    main()
