import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Script-style imports in this folder (from retrieve import ...) need rag/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent, ChatResult
from generate import Answerer
from retrieve import SearchIndex
from tools import GrepTool, SemanticTool

agent: Agent | None = None


class ChatRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    index = SearchIndex()
    agent = Agent(client, [GrepTool(), SemanticTool(index)], Answerer(client))
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResult)
def chat(body: ChatRequest) -> ChatResult:
    if agent is None:
        raise RuntimeError("agent is not ready")
    return agent.run(body.question)
