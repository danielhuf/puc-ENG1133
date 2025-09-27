#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import os
import openai
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def setup_openai():
    """Setup OpenAI API client."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set your OPENAI_API_KEY environment variable")


def parse_structured_response(response_text: str) -> tuple[str, str]:
    """
    Parse the structured JSON response from GPT-3.5.

    Args:
        response_text: JSON response from GPT-3.5

    Returns:
        Tuple of (verdict, reasoning)
    """
    try:
        data = json.loads(response_text)
        verdict = data.get("verdict", "").upper()
        reasoning = data.get("reasoning", "")

        valid_verdicts = ["YTA", "NTA", "ESH", "NAH", "INFO"]
        verdict = verdict if verdict in valid_verdicts else None
        reasoning = reasoning if reasoning and len(reasoning.strip()) >= 10 else None

        return verdict, reasoning

    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {response_text}")
        return None, None


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
):
    """
    Process a single pair of columns (label and reason) for a row.

    Args:
        df: DataFrame to update
        idx: Row index
        row: Current row data
        system_message: System prompt for LLM
        user_message: User message (selftext)
        label_col: Name of label column
        reason_col: Name of reason column
        progress_bar: tqdm progress bar
        pair_name: Name for progress tracking (e.g., "pair1")

    Returns:
        None (updates dataframe in place)
    """
    if pd.notna(row[label_col]) or pd.notna(row[reason_col]):
        progress_bar.set_postfix(**{pair_name: "skipped"})
    else:
        response = prompt_gpt35(system_message, user_message)
        verdict, reasoning = parse_structured_response(response)
        df.at[idx, label_col] = verdict
        df.at[idx, reason_col] = reasoning

        if verdict is not None and reasoning is not None:
            progress_bar.set_postfix(**{pair_name: "complete"})
        elif verdict is not None or reasoning is not None:
            progress_bar.set_postfix(**{pair_name: "partial"})
        else:
            progress_bar.set_postfix(**{pair_name: "failed"})


def prompt_gpt35(system_message: str, user_message: str) -> str:
    """
    Send prompt to GPT-3.5 and return the response.

    Args:
        system_message: System prompt for the model
        user_message: User message (selftext)

    Returns:
        Response from GPT-3.5
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "ERROR: API call failed"


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

    if "gpt3.5_label_1" not in df.columns:
        df["gpt3.5_label_1"] = pd.Series(dtype="object")
    if "gpt3.5_reason_1" not in df.columns:
        df["gpt3.5_reason_1"] = pd.Series(dtype="object")
    if "gpt3.5_label_2" not in df.columns:
        df["gpt3.5_label_2"] = pd.Series(dtype="object")
    if "gpt3.5_reason_2" not in df.columns:
        df["gpt3.5_reason_2"] = pd.Series(dtype="object")

    system_message = get_system_message(language_code)

    progress_bar = tqdm(
        df.head(5).iterrows(),
        total=5,
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
            "pair1",
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
            "pair2",
        )

        df.to_csv(file_path, index=False)

    print(f"Completed processing {language_code.upper()} dataset")


def main() -> None:
    """Main function to process all datasets with GPT-3.5."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    setup_openai()

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
