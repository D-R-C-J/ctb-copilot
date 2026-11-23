# CTBCopilot — AI Traffic Infraction Assistant
`RAG · Embeddings · Vector Search`

### It's an AI-powered assistant that helps Brazilian traffic agents identify possible traffic infractions based on the Código de Trânsito Brasileiro (CTB) and Manual Brasileiro de Fiscalização de Trânsito (MBFT)
---
## What it does:
- Accept a natural-language description of a traffic situation.
- Search relevant CTB articles and MBFT infraction records using RAG.
- Identify possible infractions, legal provisions, infraction codes, and applicable penalties or administrative measures.
- Explain the reasoning and cite the relevant legal sources.
- Ask clarifying questions when essential information is missing.
- Indicate uncertainty instead of inventing legal classifications.
Technical skills: Document processing, embeddings, vector databases, semantic search, hybrid retrieval, LLM prompting, structured outputs, and source attribution.
---
## Evaluation: 
-   Test the system using predefined traffic scenarios with verified answers. Measure retrieval accuracy, correct legal article identification, correct MBFT classification, and whether answers are supported by their cited sources.
---
## MVP:
- Use a limited collection of verified infractions rather than the entire CTB and MBFT. It should remain a decision-support tool, not an automated enforcement authority.