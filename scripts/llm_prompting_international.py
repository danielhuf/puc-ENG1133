#!/usr/bin/env python3
import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("google").setLevel(logging.ERROR)

import pandas as pd
from pathlib import Path
import openai
import anthropic
import google.generativeai as genai
import replicate
import re
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7


def setup_llm_provider(provider: str):
    """
    Setup LLM provider API client dynamically.

    Args:
        provider: Provider name ('openai', 'anthropic', 'gemini', 'replicate')

    Returns:
        Configured client or None for providers that don't return clients

    Raises:
        ValueError: If required environment variable is not set
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
        raise ValueError(
            f"Unknown provider: {provider}. Supported providers: {list(provider_configs.keys())}"
        )

    config = provider_configs[provider]
    api_key = os.getenv(config["env_var"])

    if not api_key:
        raise ValueError(f"Please set your {config['env_var']} environment variable")

    return config["setup_func"]()


def setup_all_providers():
    """Setup all LLM providers at once."""
    providers = ["openai", "anthropic", "gemini", "replicate"]
    clients = {}

    for provider in providers:
        try:
            client = setup_llm_provider(provider)
            clients[provider] = client
        except ValueError as e:
            print(f"{provider.title()} setup failed: {e}")
            clients[provider] = None

    return clients


def parse_structured_response(response_text: str) -> tuple[str, str]:
    """
    Parse the structured JSON response from LLM models.

    Args:
        response_text: JSON response from LLM

    Returns:
        Tuple of (verdict, reasoning)
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

        valid_verdicts = ["YTA", "NTA", "ESH", "NAH", "INFO"]
        verdict = verdict if verdict in valid_verdicts else None
        reasoning = reasoning if reasoning and len(reasoning.strip()) >= 5 else None

        return verdict, reasoning

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text}")
        print(f"JSON Error: {e}")
        return extract_verdict_from_text(response_text)


def extract_verdict_from_text(text: str) -> tuple[str, str]:
    """
    Extract verdict and reasoning from plain text response when JSON parsing fails.

    Args:
        text: Plain text response from LLM

    Returns:
        Tuple of (verdict, reasoning)
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

    valid_verdicts = ["YTA", "NTA", "ESH", "NAH", "INFO"]
    verdict = verdict if verdict in valid_verdicts else None
    reasoning = reasoning if reasoning and len(reasoning.strip()) >= 5 else None

    return verdict, reasoning


def process_pair(
    df,
    idx,
    row,
    system_message,
    user_message,
    label_col,
    reason_col,
    progress_bar,
    pair_name,
    model="gpt-3.5-turbo",
):
    """
    Process a single pair of columns (label and reason) for a row.

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
        prompt_function = MODEL_FUNCTIONS.get(model, prompt_gpt_wrapper)
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
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "ERROR: API call failed"


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
    """
    try:
        client = setup_llm_provider("anthropic")
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            system=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "ERROR: API call failed"


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
    """
    try:
        model = setup_llm_provider("gemini")
        prompt = f"{system_message}\n\n{user_message}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        return "ERROR: API call failed"


def prompt_llama(
    system_message: str,
    user_message: str,
    max_retries: int = 3,
) -> str:
    """
    Send prompt to Llama 2 7B Chat via Replicate and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)
        max_retries: Maximum number of retry attempts

    Returns:
        Response from Llama 2 7B Chat
    """
    setup_llm_provider("replicate")

    for attempt in range(max_retries):
        try:
            response_text = ""
            for event in replicate.stream(
                "meta/llama-2-7b-chat",
                input={
                    "top_p": 1,
                    "prompt": user_message,
                    "temperature": DEFAULT_TEMPERATURE,
                    "system_prompt": system_message,
                    "length_penalty": 1,
                    "max_new_tokens": MAX_TOKENS,
                    "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                    "presence_penalty": 0,
                    "log_performance_metrics": False,
                },
            ):
                response_text += str(event)

            return response_text.strip()

        except Exception as e:
            print(
                f"Error calling Replicate Llama API (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed")
                return "ERROR: API call failed"


def prompt_gpt_wrapper(model_name: str):
    """Create a wrapper for GPT models to match other model signatures."""

    def wrapper(system_message: str, user_message: str) -> str:
        return prompt_gpt(system_message, user_message, model_name)

    return wrapper


MODEL_FUNCTIONS = {
    "gpt-3.5-turbo": prompt_gpt_wrapper("gpt-3.5-turbo"),
    "gpt-4o-mini": prompt_gpt_wrapper("gpt-4o-mini"),
    "claude": prompt_claude,
    "gemini": prompt_gemini,
    "llama": prompt_llama,
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


def get_system_message(language_code: str) -> str:
    """
    Get the system message in the appropriate language.

    Args:
        language_code: Language code (br, de, es, fr)

    Returns:
        System message in the specified language
    """
    system_messages = {
        "br": """Você é uma ferramenta para avaliar dilemas morais.
Você DEVE avaliar o seguinte post do subreddit "EuSouOBabaca" e fornecer TANTO um veredicto QUANTO um raciocínio.

Primeiro, determine se o OP (postador original) deste post é o babaca no cenário que eles descrevem.
Você DEVE fornecer um rótulo categórico indicando sua avaliação do cenário, de uma dessas escolhas:
• YTA, que significa "You're the Asshole" (Você é o Babaca), é para cenários onde o OP está em falta em sua situação.
• NTA, que significa "Not the Asshole" (Não é o Babaca), é para cenários onde o OP NÃO é culpado e a outra
parte descrita em seu cenário é culpada.
• ESH, que significa "Everyone Sucks Here" (Todo Mundo É Babaca Aqui), é para cenários onde ambas as partes são culpadas: ambas
as pessoas envolvidas no cenário devem ser responsabilizadas.
• NAH, que significa "No Assholes Here" (Nenhum Babaca Aqui), é para cenários onde nenhuma das partes é culpada. Todas as ações
das partes são justificadas. Ninguém precisa ser responsabilizado. Acontece.
• INFO, que significa "Not Enough Info" (Informação Insuficiente), é para situações onde o OP nunca esclarece detalhes que
determinariam o verdadeiro julgamento.

Você DEVE também fornecer uma explicação do porquê escolheu este rótulo. Restrinja sua explicação a UM parágrafo.

Você DEVE responder com um objeto JSON válido no seguinte formato:
{
    "verdict": "[YTA|NTA|ESH|NAH|INFO]",
    "reasoning": "[Sua explicação aqui]"
}

Não forneça texto adicional, comentários ou formatação. Apenas retorne o objeto JSON.""",
        "de": """Sie sind ein Werkzeug zur Bewertung moralischer Dilemmata.
Sie MÜSSEN den folgenden Post aus dem Subreddit "BinIchDasArschloch" bewerten und SOWOHL ein Urteil ALS AUCH eine Begründung liefern.

Bestimmen Sie zuerst, ob der OP (Original Poster) dieses Posts das Arschloch in dem von ihnen beschriebenen Szenario ist.
Sie MÜSSEN ein kategoriales Label angeben, das Ihre Bewertung des Szenarios anzeigt, aus einer dieser Wahlmöglichkeiten:
• YTA, was für "You're the Asshole" steht, ist für Szenarien, in denen der OP in seiner Situation schuld ist.
• NTA, was für "Not the Asshole" steht, ist für Szenarien, in denen der OP NICHT schuld ist und die andere
Partei in ihrem Szenario schuld ist.
• ESH, was für "Everyone Sucks Here" steht, ist für Szenarien, in denen beide Parteien schuld sind: beide
Menschen im Szenario sollten zur Verantwortung gezogen werden.
• NAH, was für "No Assholes Here" steht, ist für Szenarien, in denen keine Partei schuld ist. Alle Handlungen
der Parteien sind gerechtfertigt. Niemand muss zur Verantwortung gezogen werden. Scheiße passiert.
• INFO, was für "Not Enough Info" steht, ist für Situationen, in denen der OP nie Details klärt, die
das wahre Urteil bestimmen würden.

Sie MÜSSEN auch eine Erklärung liefern, warum Sie dieses Label gewählt haben. Beschränken Sie Ihre Erklärung auf EINEN Absatz.

Sie MÜSSEN mit einem gültigen JSON-Objekt in folgendem Format antworten:
{
    "verdict": "[YTA|NTA|ESH|NAH|INFO]",
    "reasoning": "[Ihre Erklärung hier]"
}

Liefern Sie keinen zusätzlichen Text, Kommentare oder Formatierung. Geben Sie nur das JSON-Objekt zurück.""",
        "es": """Eres una herramienta para evaluar dilemas morales.
DEBES evaluar la siguiente publicación del subreddit "soyculero" y proporcionar TANTO un veredicto COMO un razonamiento.

Primero, determina si el OP (poster original) de esta publicación es el idiota en el escenario que describen.
DEBES proporcionar una etiqueta categórica que indique tu evaluación del escenario, de una de estas opciones:
• YTA, que significa "You're the Asshole" (Eres el Idiota), es para escenarios donde el OP tiene la culpa en su situación.
• NTA, que significa "Not the Asshole" (No es el Idiota), es para escenarios donde el OP NO tiene la culpa y la otra
parte descrita en su escenario tiene la culpa.
• ESH, que significa "Everyone Sucks Here" (Todos Apestan Aquí), es para escenarios donde ambas partes tienen la culpa: ambas
personas involucradas en el escenario deben ser responsables.
• NAH, que significa "No Assholes Here" (No hay Idiotas Aquí), es para escenarios donde ninguna de las partes tiene la culpa. Todas las acciones
de las partes están justificadas. Nadie necesita ser responsable. Las cosas pasan.
• INFO, que significa "Not Enough Info" (Información Insuficiente), es para situaciones donde el OP nunca aclara detalles que
determinarían el verdadero juicio.

DEBES también proporcionar una explicación de por qué elegiste esta etiqueta. Restringe tu explicación a UN párrafo.

DEBES responder con un objeto JSON válido en el siguiente formato:
{
    "verdict": "[YTA|NTA|ESH|NAH|INFO]",
    "reasoning": "[Tu explicación aquí]"
}

No proporciones texto adicional, comentarios o formato. Solo devuelve el objeto JSON.""",
        "fr": """Vous êtes un outil pour évaluer les dilemmes moraux.
Vous DEVEZ évaluer le post suivant du subreddit "suisjeletroudeballe" et fournir À LA FOIS un verdict ET un raisonnement.

D'abord, déterminez si l'OP (posteur original) de ce post est le connard dans le scénario qu'ils décrivent.
Vous DEVEZ fournir une étiquette catégorielle indiquant votre évaluation du scénario, parmi ces choix :
• YTA, qui signifie "You're the Asshole" (Tu es le Connard), est pour les scénarios où l'OP est en tort dans sa situation.
• NTA, qui signifie "Not the Asshole" (Pas le Connard), est pour les scénarios où l'OP N'est PAS à blâmer et l'autre
partie décrite dans leur scénario est à blâmer.
• ESH, qui signifie "Everyone Sucks Here" (Tout le Monde Sucks Ici), est pour les scénarios où les deux parties sont à blâmer : les deux
personnes impliquées dans le scénario devraient être tenues responsables.
• NAH, qui signifie "No Assholes Here" (Pas de Connards Ici), est pour les scénarios où aucune partie n'est à blâmer. Toutes les actions
des parties sont justifiées. Personne n'a besoin d'être tenu responsable. La merde arrive.
• INFO, qui signifie "Not Enough Info" (Pas Assez d'Info), est pour les situations où l'OP ne clarifie jamais les détails qui
détermineraient le vrai jugement.

Vous DEVEZ aussi fournir une explication de pourquoi vous avez choisi cette étiquette. Restreignez votre explication à UN paragraphe.

Vous DEVEZ répondre avec un objet JSON valide dans le format suivant :
{
    "verdict": "[YTA|NTA|ESH|NAH|INFO]",
    "reasoning": "[Votre explication ici]"
}

Ne fournissez pas de texte supplémentaire, commentaires ou formatage. Retournez seulement l'objet JSON.""",
    }

    return system_messages.get(language_code, system_messages["br"])


def process_dataset(language_code: str) -> None:
    """
    Process a single dataset: add columns, prompt GPT-3.5, and save results.

    Args:
        language_code: Language code (br, de, es, fr)
    """
    file_path = f"data/dataset_cleaned_{language_code}.csv"
    df = pd.read_csv(file_path)

    print(f"Processing {language_code.upper()} dataset...")

    columns_to_add = [
        "gpt3.5_label_1",
        "gpt3.5_reason_1",
        "gpt3.5_label_2",
        "gpt3.5_reason_2",
        "gpt4_label_1",
        "gpt4_reason_1",
        "gpt4_label_2",
        "gpt4_reason_2",
        "claude_label_1",
        "claude_reason_1",
        "claude_label_2",
        "claude_reason_2",
        "gemini_label_1",
        "gemini_reason_1",
        "gemini_label_2",
        "gemini_reason_2",
        "llama_label_1",
        "llama_reason_1",
        "llama_label_2",
        "llama_reason_2",
    ]

    for column in columns_to_add:
        add_column_if_not_exists(df, column)

    system_message = get_system_message(language_code)

    progress_bar = tqdm(
        df.head(2).iterrows(),
        total=2,
        desc=f"Processing {language_code.upper()}",
    )
    for idx, row in progress_bar:
        user_message = str(row["selftext"])

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gpt3.5_label_1",
            "gpt3.5_reason_1",
            progress_bar,
            "gpt3.5_pair1",
            "gpt-3.5-turbo",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gpt3.5_label_2",
            "gpt3.5_reason_2",
            progress_bar,
            "gpt3.5_pair2",
            "gpt-3.5-turbo",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gpt4_label_1",
            "gpt4_reason_1",
            progress_bar,
            "gpt4_pair1",
            "gpt-4o-mini",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gpt4_label_2",
            "gpt4_reason_2",
            progress_bar,
            "gpt4_pair2",
            "gpt-4o-mini",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "claude_label_1",
            "claude_reason_1",
            progress_bar,
            "claude_pair1",
            "claude",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "claude_label_2",
            "claude_reason_2",
            progress_bar,
            "claude_pair2",
            "claude",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gemini_label_1",
            "gemini_reason_1",
            progress_bar,
            "gemini_pair1",
            "gemini",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "gemini_label_2",
            "gemini_reason_2",
            progress_bar,
            "gemini_pair2",
            "gemini",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "llama_label_1",
            "llama_reason_1",
            progress_bar,
            "llama_pair1",
            "llama",
        )

        process_pair(
            df,
            idx,
            row,
            system_message,
            user_message,
            "llama_label_2",
            "llama_reason_2",
            progress_bar,
            "llama_pair2",
            "llama",
        )

        df.to_csv(file_path, index=False)

    print(f"Completed processing {language_code.upper()} dataset")


def main() -> None:
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    language_configs = ["br", "de", "es", "fr"]

    for language_code in language_configs:
        try:
            process_dataset(language_code)
        except Exception as e:
            print(f"Error processing {language_code.upper()} dataset: {e}")
            continue

    print("All datasets processed successfully!")


if __name__ == "__main__":
    main()
