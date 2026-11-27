from google import genai
from pydantic import BaseModel, Field
import json

class AssessmentSchema(BaseModel):
    ctb_passage_id: str = Field(
        description="ID of the matching CTB passage from the provided evidence (or empty string if none applies)."
    )
    mbft_passage_id: str = Field(
        description="ID of the matching MBFT passage from the provided evidence (or empty string if none applies)."
    )
    explanation: str = Field(
        description="Concise justification grounded strictly in the provided evidence."
    )

def generate_assessment(obs: str, passages: list[dict], client: genai.Client, model_name: str = "gemini-3.5-flash-lite") -> dict:
    evidence: list[dict] = []
    for p in passages:
        evidence.append({"id": p["id"], "text": p["text"]})
    input_dict = {"observation": obs, "evidence": evidence}
    input_json = json.dumps(input_dict)
    task_instruction = (
        "Você é um assistente de apoio à decisão na fiscalização de trânsito. "
        "Considere os fatos relatados na observação e use apenas as passagens fornecidas "
        "como evidência para propor uma possível classificação. "
        "Se houver evidência suficiente, selecione o ID de uma passagem do CTB e o ID de uma "
        "ficha do MBFT que, juntos, sustentem a proposta. "
        "Caso contrário, deixe ambos os IDs como strings vazias e explique o que falta. "
        "Escreva o campo explanation em português do Brasil. Ao propor uma classificação, "
        "explique como um fato específico da observação se relaciona com as passagens selecionadas."
    )
    interaction = client.interactions.create(
        model = model_name,
        input = input_json,
        system_instruction=task_instruction,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AssessmentSchema.model_json_schema()
        },
    )

    assessment = AssessmentSchema.model_validate_json(interaction.output_text)

    usage = interaction.usage
    result = assessment.model_dump()

    result["usage"] = {
        "input_tokens": getattr(usage, "total_input_tokens", 0),
        "output_tokens": getattr(usage, "total_output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }

    return result
