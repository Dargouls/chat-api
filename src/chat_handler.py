from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

from google import genai
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
# API_PROVIDER = os.getenv("API_PROVIDER") # Não utilizado no código fornecido

# Configuração do cliente GenAI
# É uma boa prática configurar o cliente uma vez e reutilizá-lo.
# Considerar o tratamento de exceções se a API_KEY não estiver definida.
if not API_KEY:
    raise ValueError("API_KEY não encontrada nas variáveis de ambiente.")

# Não há genai.Client() na biblioteca google-genai.
# A configuração é feita com genai.configure() e depois se usa genai.GenerativeModel()
# ou, se você estiver usando uma versão/interface que realmente tem genai.Client,
# certifique-se de que é a correta.
# Assumindo que você está usando a API mais recente e o client para chamadas diretas:
# (Nota: Se você encontrar "AttributeError: module 'google.generativeai' has no attribute 'Client'",
#  você pode estar misturando versões ou precisa usar genai.configure(api_key=API_KEY)
#  e depois model = genai.GenerativeModel(...); model.generate_content(...))
#
# Para a biblioteca google-genai mais recente (pacote google-genai), o client é usado assim:
try:
    # Se o seu 'genai.Client' estava funcionando antes, mantenha-o.
    # Caso contrário, a forma comum é configurar e usar GenerativeModel diretamente.
    # Este exemplo assume que genai.Client() é a forma que você conseguiu inicializar.
    # Se API_KEY for para o client genai:
    client = genai.Client(
        api_key=API_KEY
    )  # Se esta linha deu erro antes, precisa ser ajustada.
    # A biblioteca mais nova `google-genai` usa `genai.configure(api_key=API_KEY)`
    # e depois `model = genai.GenerativeModel(...)` para chamadas diretas,
    # ou `genai.ChatSession` para conversas.
    # No entanto, client.models.generate_content é uma interface válida
    # no SDK mais recente (`google-genai` >= 0.6.0)
except AttributeError:
    print("Alerta: genai.Client() não encontrado. Tentando genai.configure().")
    print("Certifique-se de ter o pacote 'google-genai' instalado e atualizado.")
    genai.configure(api_key=API_KEY)
    # Neste caso, a chamada a client.models.generate_content precisaria ser model.generate_content
    # Vamos prosseguir assumindo que client.models.generate_content é o caminho desejado
    # e que o client foi instanciado corretamente como no código original.
    # Se `genai.Client` não é o caminho, você precisará instanciar um `GenerativeModel` aqui.
    # Ex: model_instance = genai.GenerativeModel(os.getenv("MODEL_ID"))
    # e depois usar model_instance.generate_content(...)

TRANSLATIONS = {
    "pt": {
        "error_message": "Desculpe, ocorreu um erro: {}. Por favor, verifique sua conexão e configurações."
    },
}


# Modelos Pydantic para uma tipagem mais clara do histórico
class Part(BaseModel):
    text: str
    # Você pode adicionar outros campos se estiver lidando com multimodalidade (ex: inline_data)


class Content(BaseModel):
    role: str
    parts: List[Part]


class ChatRequest(BaseModel):
    message: str
    files: List[Dict[str, str]] = Field(
        default_factory=list
    )  # Exemplo simplificado para arquivos
    history: List[Content] = Field(
        default_factory=list
    )  # Histórico de mensagens formatado


def respond(request: ChatRequest):
    message_text = request.message.strip()
    # É importante clonar o histórico se você não quiser modificar o objeto original da requisição
    # ou se a instância de ChatRequest for usada em outro lugar.
    # No entanto, para manter o histórico da conversa, modificar em loco é comum.
    current_history: List[Dict[str, Any]] = [
        item.model_dump() for item in request.history
    ]

    print(f"Histórico recebido: {current_history}")

    if not message_text:
        return {"response": "Por favor, envie uma mensagem válida."}

    try:
        model_id = os.getenv("MODEL_ID")
        print(f"Modelo ID: {model_id}")
        if not model_id:
            raise ValueError("MODEL_ID não encontrado nas variáveis de ambiente.")

        # Adiciona a mensagem atual do usuário ao histórico
        current_history.append({"role": "user", "parts": [{"text": message_text}]})

        print(f"Conteúdo enviado para o modelo: {current_history}")

        completion = client.models.generate_content(
            model=model_id,
            contents=current_history,
            # generation_config=genai.types.GenerationConfig( # Exemplo de como adicionar config
            #     max_output_tokens=300,
            #     temperature=0.5,
            #     top_p=0.7
            # )
        )

        print(f"Resposta bruta da API: {completion}")

        response_content = ""
        if completion.text:
            response_content = completion.text
        elif (
            completion.candidates
            and completion.candidates[0].content
            and completion.candidates[0].content.parts
        ):
            response_parts = [
                part.text
                for part in completion.candidates[0].content.parts
                if hasattr(part, "text")
            ]
            response_content = " ".join(response_parts).strip()
        else:
            if (
                hasattr(completion, "prompt_feedback")
                and completion.prompt_feedback
                and completion.prompt_feedback.block_reason
            ):
                block_reason_message = (
                    getattr(completion.prompt_feedback, "block_reason_message", "")
                    or f"Conteúdo bloqueado devido a: {completion.prompt_feedback.block_reason}"
                )
                response_content = (
                    f"Não foi possível gerar uma resposta. {block_reason_message}"
                )
            else:
                response_content = "Não foi possível obter uma resposta do modelo ou a resposta estava vazia."

        if response_content:
            current_history.append(
                {"role": "model", "parts": [{"text": response_content}]}
            )  # CORREÇÃO APLICADA AQUI
        else:
            current_history.append(
                {
                    "role": "model",
                    "parts": [
                        {"text": "[Não houve resposta do modelo ou foi bloqueada]"}
                    ],
                }
            )

        return {"response": response_content, "history": current_history}

    except Exception as e:
        error_message_template = TRANSLATIONS["pt"]["error_message"]
        if isinstance(
            e, AttributeError
        ) and "'GenerateContentResponse' object has no attribute 'choices'" in str(e):
            detailed_error = "Ocorreu um erro ao processar a resposta do modelo. A estrutura da resposta não é 'completion.choices'. Use 'completion.text' ou 'completion.candidates'."
            error_message = error_message_template.format(
                f"{type(e).__name__}: {detailed_error} (Erro original: {str(e)})"
            )
        else:
            error_message = error_message_template.format(
                f"{type(e).__name__}: {str(e)}"
            )

        print(f"Erro na função respond: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)


app = APIRouter()


@app.post("/respond")  # Usando o router renomeado
def chat_respond(request: ChatRequest):
    return respond(request)
