from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import os

load_dotenv()

API_KEY = os.getenv("GATEWAY_URL")

client = InferenceClient(api_key=API_KEY)
TRANSLATIONS = {
    "pt": {
        "error_message": "Desculpe, ocorreu um erro: {}. Por favor, verifique sua conexão e configurações."
    },
}


class ChatRequest(BaseModel):
    message: str
    # Para suportar arquivos como imagens (exemplo simplificado)
    files: List[Dict[str, str]] = []
    history: List[Dict[str, str]]


def respond(request: ChatRequest):
    message = request.message.strip()
    history = request.history

    print(history)
    if not message:
        return {"response": "Mensagem vazia, por favor, envie uma mensagem válida."}

    try:
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        messages = history
        messages.append({"role": "user", "content": message})

        print(messages)
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=300,
            temperature=0.7,  # Reduzir a aleatoriedade
            top_p=0.9,  # Limitar a diversidade
        )
        print("Modelo carregado")

        response = completion.choices[0].message
        # Remover as quebras de linha no conteúdo da resposta
        # Substituir \n por espaço (ou outro caractere)
        response_content = response.content
        messages.append({"role": response.role, "content": response_content})
        print("Resposta gerada")

        return {"response": response_content, "history": history}

    except Exception as e:
        error_message = TRANSLATIONS["pt"]["error_message"].format(str(e))
        raise HTTPException(status_code=500, detail=error_message)


# FastAPI App
app = APIRouter()


@app.post("/respond")
def chat_respond(request: ChatRequest):
    return respond(request)
