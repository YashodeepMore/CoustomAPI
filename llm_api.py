from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://your-site.com",
    "X-Title": "Your App Name"
}

app = FastAPI()

# -------- Request Schema --------
class LLMRequest(BaseModel):
    user_query: str
    masked_messages: list[str]
    mode: str = "general"


# =========================================================
# PROMPT BUILDERS (MODE-SPECIFIC)
# =========================================================

def build_general_prompt(user_query: str, messages: list[str]) -> str:
    retrieved_block = "\n".join(
        [f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)]
    )

    return f"""
You are a helpful AI assistant.

Rules:
- Answer clearly and concisely.
- Use retrieved messages only if helpful.
- Do not hallucinate facts.

User Query:
"{user_query}"

Retrieved Context:
{retrieved_block}
""".strip()


def build_private_finance_prompt(user_query: str, messages: list[str]) -> str:
    retrieved_block = "\n".join(
        [f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)]
    )

    return f"""
You are a financial reasoning assistant for a privacy-safe RAG system.

IMPORTANT RULES:
- Messages contain masked placeholders like #amount, #receiver, #date.
- NEVER modify, replace, or invent placeholders.
- NEVER infer real values.
- Use placeholders exactly as given.
- If calculation is impossible due to same placeholder reuse, say so clearly.
- Ignore messages without #amount if the question is about payments.

User Query:
"{user_query}"

Retrieved Messages:
{retrieved_block}

Task:
- Identify financial transactions.
- Reason symbolically using placeholders.
- Keep the answer short (1–2 sentences).
""".strip()


def build_learning_prompt(user_query: str, messages: list[str]) -> str:
    retrieved_block = "\n".join(
        [f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)]
    )

    return f"""
You are a patient teaching assistant.

Rules:
- Explain concepts step by step.
- Use simple language.
- Examples are allowed.
- Retrieved messages can be ignored if irrelevant.

Question:
"{user_query}"

Optional Context:
{retrieved_block}
""".strip()


# =========================================================
# PROMPT ROUTER
# =========================================================

def build_prompt(user_query: str, messages: list[str], mode: str) -> str:
    if mode == "private_finance":
        return build_private_finance_prompt(user_query, messages)
    elif mode == "learning":
        return build_learning_prompt(user_query, messages)
    else:
        return build_general_prompt(user_query, messages)


# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/ask")
def ask_llm(req: LLMRequest):
    prompt = build_prompt(
        user_query=req.user_query,
        messages=req.masked_messages,
        mode=req.mode
    )

    payload = {
        "model": "nex-agi/deepseek-v3.1-nex-n1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = response.json()

    if "choices" not in data:
        raise HTTPException(status_code=500, detail=data)

    return {
        "answer": data["choices"][0]["message"]["content"],
        "mode_used": req.mode
    }
