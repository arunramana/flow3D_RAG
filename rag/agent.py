from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from generate import MODEL, Answerer
from retrieve import SearchIndex
from tools import GrepTool, SemanticTool, Tool

MAX_SEARCHES = 2
MAX_CONTEXT = 5
PREVIEW = 400

AGENT_RULES = """Choose the next step.
- grep: the question hinges on an exact token such as DNS or VOF. Put only that token in query.
- semantic: the wording is a phrase, not one token. Put a short search phrase in query.
- answer: a listed source covers the question, or another search will not help. Leave query empty.
Search at most twice. Do not repeat a tool you already used."""


class Action(BaseModel):
    tool: Literal["grep", "semantic", "answer"]
    query: str


class Citation(BaseModel):
    n: int
    heading: str
    url: str
    text: str


class ChatResult(BaseModel):
    answer: str
    citations: list[Citation]


def previews(hits: list[dict]) -> str:
    """What the agent is allowed to see. Not the full pages."""
    if not hits:
        return "(no sources yet)"
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit['heading']}\n{hit['url']}\n{hit['text'][:PREVIEW]}")
    return "\n\n".join(blocks)


def merge(hits: list[dict], found: list[dict]) -> list[dict]:
    """One hit per URL. A longer text replaces the shorter snippet from the same page."""
    by_url = {hit["url"]: hit for hit in hits}
    for hit in found:
        old = by_url.get(hit["url"])
        if old is None:
            if len(hits) == MAX_CONTEXT:
                continue
            hits.append(hit)
            by_url[hit["url"]] = hit
        elif len(hit["text"]) > len(old["text"]):
            old["text"] = hit["text"]
    return hits


class Agent:
    """Chooses a tool, then asks Answerer.

    A new tool is another class in the list. This loop does not grow a new branch.
    """

    def __init__(self, client: OpenAI, tools: list[Tool], answerer: Answerer, model: str = MODEL) -> None:
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        self.answerer = answerer
        self.model = model

    def choose(self, question: str, hits: list[dict], used: list[str]) -> Action:
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_RULES},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Tools already used: {', '.join(used) or 'none'}\n\n"
                    f"Sources:\n{previews(hits)}"
                )},
            ],
            response_format=Action,
        )
        return completion.choices[0].message.parsed

    def run(self, question: str) -> ChatResult:
        hits: list[dict] = []
        used: list[str] = []

        for _ in range(MAX_SEARCHES):
            action = self.choose(question, hits, used)
            print(f"\n--- action ---\n{action.tool} {action.query}".rstrip())
            tool = self.tools.get(action.tool)
            if tool is None or not action.query.strip():
                break
            used.append(action.tool)
            hits = merge(hits, tool.search(action.query))

        result = self.answerer.answer(question, hits, debug=False)
        cited = []
        for n in result.citations:
            if 1 <= n <= len(hits):
                hit = hits[n - 1]
                cited.append(Citation(n=n, heading=hit["heading"], url=hit["url"], text=hit["text"]))
        return ChatResult(answer=result.answer, citations=cited)


def main() -> None:
    question = "What is DNS?"
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    index = SearchIndex()
    agent = Agent(client, [GrepTool(), SemanticTool(index)], Answerer(client))
    result = agent.run(question)
    print("\n--- answer ---")
    print(result.answer)
    print("citations:", [item.n for item in result.citations])
    for item in result.citations:
        print(f"[{item.n}] {item.url}")


if __name__ == "__main__":
    main()
