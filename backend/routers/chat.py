# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.llm import OllamaError, call_ollama
from typing import Optional

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    mode: str = "explain"

@router.post("/")
async def chat(req: ChatRequest):
    """
    Call Ollama and return the full response as JSON.
    """
    system_prompt = build_system_prompt(req.mode, req.context)
    try:
        response = await call_ollama(system_prompt, req.message)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    return {"response": response}

def build_system_prompt(mode: str, context: Optional[str]) -> str:
    base = (
        "You are DataForge, an expert AI data scientist and statistician.\n"
        "Always be concise and precise. Use the variable name `df` for the dataframe.\n"
    )
    if context:
        base += f"\nLoaded dataset schema:\n{context}\n"

    if mode == "code":
        base += (
            "\nGenerate concise Python (pandas) code to answer the user's question."
            "\nBefore the code block, briefly explain what the code does and WHY (1-2 sentences)."
            "\nReturn code in a ```python block. After the block, mention any assumptions made."
        )
    elif mode == "plot":
        base += (
            "\nGenerate Python (matplotlib) code to create the requested chart."
            "\nBefore the code block, explain:"
            "\n  1. What the chart shows and why it is appropriate"
            "\n  2. The mathematical/statistical formula or method used (e.g. IQR, Pearson r, KDE, etc.)"
            "\nReturn code in a ```python block. Use df as the variable name. Set a dark background: plt.style.use('dark_background')."
        )
    elif mode == "explain":
        base += (
            "\nExplain clearly in plain English. When describing statistics or patterns:"
            "\n  - State the formula or method used (e.g. 'Standard deviation = sqrt(variance)')"
            "\n  - Interpret what the numbers mean in context"
            "\n  - Point out anything surprising or worth investigating"
            "\nBe concise but thorough."
        )
    return base
