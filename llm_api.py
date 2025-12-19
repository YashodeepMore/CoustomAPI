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


# -------- Prompt Builder --------
def build_prompt(user_query: str, messages: list[str]) -> str:
    retrieved_block = "\n".join(
        [f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)]
    )

    prompt = f"""
You are a natural-language reasoning assistant for a RAG system.

IMPORTANT RULES:
- The text contains masked entities like #amount, #receiver, #date.
- These placeholders MUST remain EXACTLY as they appear.
- NEVER replace, modify, create, or remove placeholders.
- Never hallucinate real names, numbers, dates, or apps.
- Only summarize or compute using the placeholders given.
- Ignore messages that do NOT contain #amount if question is about payments.
- If total cannot be calculated because placeholders are the same, say so.

USER QUERY:
"{user_query}"

RETRIEVED MESSAGES:
{retrieved_block}

TASK:
- If the user query is about natural conversation, respond naturally and can neglect retrieved messages.
- Identify which messages represent payments.
- Use only the placeholders to calculate.
- If multiple different #amount placeholders exist, express total as (#amount + #amount).
- If the same placeholder repeats, say:
  "Both payments use the placeholder #amount, so the total cannot be calculated."

Give final answer in 1–2 sentences.
"""
    return prompt.strip()


# -------- API Endpoint --------
@app.post("/ask")
def ask_llm(req: LLMRequest):
    prompt = build_prompt(req.user_query, req.masked_messages)
    

    payload = {
        "model": "kwaipilot/kat-coder-pro:free",
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
        "answer": data["choices"][0]["message"]["content"]
    }
