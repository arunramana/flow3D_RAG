from openai import OpenAI
from pydantic import BaseModel

from retrieve import SearchIndex

MODEL = "qwen2.5:7b"

RULES = """Answer only from the numbered context.
Cite the chunks you use as [1], [2].
The citations field must list those same numbers.
If the context does not cover the question, answer "I don't know" and leave citations empty."""


class Answer(BaseModel):
    answer: str
    citations: list[int]


def format_context(hits: list[dict]) -> str:
    """Number the chunks the model is allowed to cite."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit['heading']}\n{hit['url']}\n{hit['text']}")
    return "\n\n".join(blocks)


def show_hits(hits: list[dict]) -> None:
    for i, hit in enumerate(hits, start=1):
        print(f"\n[{i}] {hit['score']:.4f}  {hit['heading']}")
        print(hit["url"])
        print(hit["text"][:400])


class Answerer:
    """Turns retrieved hits into a cited answer. The chat client is passed in."""

    def __init__(self, client: OpenAI, model: str = MODEL) -> None:
        self.client = client
        self.model = model

    def answer(self, question: str, hits: list[dict], debug: bool = False) -> Answer:
        context = format_context(hits)
        if debug:
            show_hits(hits)
            print("\n--- context sent to the model ---\n")
            print(context)

        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": RULES},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
            ],
            response_format=Answer,
        )
        return completion.choices[0].message.parsed


def main() -> None:
    question = "What is DNS?"
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    hits = SearchIndex().retrieve(question)
    result = Answerer(client).answer(question, hits, debug=True)

    print("\n--- answer ---")
    print(result.answer)
    print("citations:", result.citations)
    for n in result.citations:
        if 1 <= n <= len(hits):
            print(f"[{n}] {hits[n - 1]['url']}")


if __name__ == "__main__":
    main()
