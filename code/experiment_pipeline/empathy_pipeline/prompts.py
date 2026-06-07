from __future__ import annotations

from typing import Any

PROFILE_SYSTEM = """You are an expert in clinical psychology and conversation analysis. Your task is to analyze a dialogue between a user and an assistant, and produce a structured profile of the user that will guide empathetic response generation.

You must output exactly four sections, enclosed in the specified XML-like tags. Do not add any text outside these tags.

The four required sections are:
1. <ToM> ... </ToM>: A Theory of Mind analysis describing the user's current emotional state, underlying needs, and cognitive perspective, informed by the conversation history.
2. <Strategy> ... </Strategy>: A concise recommendation of the empathetic strategy to adopt in the next response, such as validation, normalization, reflective listening, or an open-ended question.
3. <Style> ... </Style>: A description of the user's observed language style preferences, such as formality, vocabulary, and tone.
4. <Context> ... </Context>: Key situational or factual details mentioned by the user that should be remembered for personalized responses.
"""

PROFILE_UPDATE_SYSTEM = PROFILE_SYSTEM + """
When a previous running profile is provided, update it rather than repeating it blindly. Preserve durable user context, revise emotional-state inferences if the new turn changes them, and keep the output concise enough to fit into a downstream prompt.
"""

FACT_SYSTEM = """You are an information extraction agent. Extract the factual assertions in the RAW RESPONSE that must be preserved in an empathetic rewrite.

Rules:
- Preserve exact numbers, names, dates, diagnoses, medications, locations, procedures, definitions, instructions, and causal claims.
- Do not extract greetings, emotional validation, filler phrases, or unsupported advice that is not factual.
- Split compound statements into atomic facts when possible.
- Output a JSON array of strings and nothing else.
- If there are no factual assertions, output [].
"""

FUSION_SYSTEM = """You are an empathetic response editor. Rewrite the RAW RESPONSE so it sounds more emotionally attuned and personalized to the user, while preserving the factual content.

Hard constraints:
- Preserve every item in EXTRACTED FACTS. Do not change numbers, names, dates, conditions, medication names, or other factual details.
- Do not add new factual claims, diagnoses, promises, or safety instructions not supported by the RAW RESPONSE or the conversation.
- Keep the same language as the user's latest utterance.
- Output only the final response to the user. Do not output tags, explanations, or analysis.
- Prefer concise support over long poetic language.
"""

FUSION_NO_FACT_SYSTEM = """You are an empathetic response editor. Rewrite the RAW RESPONSE so it sounds more emotionally attuned and personalized to the user.

Hard constraints:
- Preserve the meaning of the RAW RESPONSE as much as possible.
- Do not add new factual claims, diagnoses, promises, or safety instructions not supported by the RAW RESPONSE or the conversation.
- Keep the same language as the user's latest utterance.
- Output only the final response to the user. Do not output tags, explanations, or analysis.
"""


def role_label(role: str) -> str:
    if role.lower() == "assistant":
        return "Assistant"
    return "User"


def history_to_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"[{role_label(m['role'])}]: {m['content']}" for m in messages)


def raw_response_system(language: str) -> str:
    if language == "zh":
        return (
            "你是一位谨慎、专业的心理支持对话助手。请基于聊天记录直接回复用户，"
            "语气自然、温暖，但不要编造事实、不要做诊断、不要输出分析过程。"
        )
    return (
        "You are a careful, professional emotional-support dialogue assistant. "
        "Reply directly to the user based on the conversation history. Be warm and natural, "
        "but do not invent facts, diagnose the user, or reveal analysis."
    )


def build_profile_user_prompt(
    record: dict[str, Any],
    previous_profile: str | None = None,
    current_turn_only: bool = False,
) -> str:
    if current_turn_only:
        history = f"[User]: {record['user_utterance']}"
    else:
        history = record.get("history_text") or history_to_text(record["history"])

    parts = []
    if previous_profile:
        parts.append("Previous running user profile:\n" + previous_profile.strip())
    parts.append("Conversation history:\n" + history)
    parts.append("Generate the four sections based on this conversation.")
    return "\n\n".join(parts)


def build_fact_user_prompt(raw_response: str) -> str:
    return "RAW RESPONSE:\n" + raw_response


def build_fusion_user_prompt(
    record: dict[str, Any],
    raw_response: str,
    profile: str | None = None,
    facts: str | list[str] | None = None,
) -> str:
    if isinstance(facts, list):
        facts_text = "\n".join(f"- {fact}" for fact in facts)
    else:
        facts_text = facts or "[]"

    sections = [
        "CONVERSATION HISTORY:\n" + (record.get("history_text") or history_to_text(record["history"])),
        "LATEST USER UTTERANCE:\n" + record["user_utterance"],
        "RAW RESPONSE:\n" + raw_response,
    ]
    if profile:
        sections.append("USER PROFILE:\n" + profile)
    if facts is not None:
        sections.append("EXTRACTED FACTS:\n" + facts_text)
    return "\n\n".join(sections)


def chat_messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

