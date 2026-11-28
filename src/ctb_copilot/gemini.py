from google import genai
from pydantic import BaseModel, Field, TypeAdapter
import json
from typing import Literal, Union, Annotated

class CandidateAssessment(BaseModel):
    outcome: Literal["candidate"]
    ctb_passage_id: str = Field(
        description="ID of the matching CTB passage from the provided evidence."
    )
    mbft_passage_id: str = Field(
        description="ID of the matching MBFT passage from the provided evidence."
    )
    explanation: str = Field(
        description="Concise justification grounded strictly in the provided evidence."
    )


class ClarificationAssessment(BaseModel):
    outcome: Literal["ask_clarification"]
    question: str = Field(
        description="A specific question in Portuguese about the missing fact needed to classify the observation."
    )


class InsufficientEvidenceAssessment(BaseModel):
    outcome: Literal["insufficient_evidence"]
    explanation: str = Field(
        description="Explanation in Portuguese of what the supplied sources do not cover."
    )


AssessmentResponse = Annotated[
    Union[
        CandidateAssessment,
        ClarificationAssessment,
        InsufficientEvidenceAssessment,
    ],
    Field(discriminator="outcome"),
]

ASSESSMENT_ADAPTER = TypeAdapter(AssessmentResponse)

def generate_assessment(obs: str, passages: list[dict], client: genai.Client, model_name: str = "gemini-3.5-flash-lite") -> dict:
    evidence: list[dict] = []
    for p in passages:
        evidence.append({"id": p["id"], "text": p["text"]})
    input_dict = {"observation": obs, "evidence": evidence}
    input_json = json.dumps(input_dict)
    task_instruction = (
        "Você é um assistente de apoio à decisão na fiscalização de trânsito. "
        "Use apenas os fatos da observação e as passagens fornecidas. "
        "Identifique condutor e passageiro, pois pode influenciar na infração. "
        "Responda em português do Brasil.\n\n"
        "Escolha o outcome seguindo esta ordem:\n"
        "- ask_clarification: se faltar um fato que possa mudar o enquadramento, escolha este "
        "resultado mesmo que alguma passagem pareça relevante. No campo question, faça uma "
        "pergunta direta sobre o fato que falta.\n"
        "- candidate: se os fatos forem suficientes e uma passagem do CTB e uma ficha do MBFT "
        "sustentarem juntas a proposta, copie os dois IDs exatamente como aparecem no campo id "
        "das passagens fornecidas. Relacione um fato da observação às passagens em explanation.\n"
        "- insufficient_evidence: se os fatos forem suficientes, mas as passagens fornecidas "
        "não sustentarem uma proposta, descreva essa lacuna no campo explanation.\n\n"
        "Inclua ctb_passage_id e mbft_passage_id apenas em candidate."
    )
    interaction = client.interactions.create(
        model = model_name,
        input = input_json,
        system_instruction=task_instruction,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ASSESSMENT_ADAPTER.json_schema()
        },
    )

    assessment = ASSESSMENT_ADAPTER.validate_json(interaction.output_text)

    usage = interaction.usage
    result = assessment.model_dump()

    result["usage"] = {
        "input_tokens": getattr(usage, "total_input_tokens", 0),
        "output_tokens": getattr(usage, "total_output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }

    return result
