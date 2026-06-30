from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncap_rag.retrieve import retrieve


SYSTEM_PROMPT = """
You are a careful RAG assistant.

Answer the user's question using only the provided context.
If the context does not contain enough information, say that you do not know.
Do not invent facts.
Cite the source file and page number when relevant.
""".strip()


def format_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}]\n"
            f"File: {chunk['source_file']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Score: {chunk['score']:.4f}\n"
            f"Text:\n{chunk['text']}"
        )

    return "\n\n" + ("-" * 80) + "\n\n".join(context_parts)


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Build the final prompt sent to Bedrock."""
    context = format_context(retrieved_chunks)

    return f"""
Answer the question using only the context below.

Question:
{question}

Context:
{context}

Instructions:
- Use only the context.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Include source file and page number references where useful.
""".strip()


def call_bedrock(prompt: str) -> str:
    """Call AWS Bedrock using the Converse API."""
    load_dotenv()

    region = os.environ.get("AWS_REGION", "eu-central-1")
    model_id = os.environ["BEDROCK_MODEL_ID"]

    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse(
        modelId=model_id,
        system=[
            {
                "text": SYSTEM_PROMPT,
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt,
                    }
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 800,
            "temperature": 0.2,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


def answer_question(question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Retrieve local context and generate an answer with Bedrock."""
    retrieved_chunks = retrieve(question, top_k=top_k)
    prompt = build_user_prompt(question, retrieved_chunks)
    answer = call_bedrock(prompt)

    return answer, retrieved_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    answer, retrieved_chunks = answer_question(args.question, top_k=args.top_k)

    print("\nAnswer:")
    print("=" * 80)
    print(answer)

    print("\nRetrieved sources:")
    print("=" * 80)

    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(
            f"{i}. {chunk['source_file']}, page {chunk['page_number']} "
            f"(score: {chunk['score']:.4f})"
        )


if __name__ == "__main__":
    main()
