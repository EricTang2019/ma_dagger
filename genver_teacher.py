from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from vllm_engine import VLLMChatEngine, call_engine

logger = logging.getLogger(__name__)

GEN_SYSTEM = (
    "You are a careful math problem solver. "
    "Think step by step in <think>...</think>, but present the final answer as \\boxed{...}. "
    "Be concise and precise."
)
VER_SYSTEM = (
    "You are an exacting math verifier. Given the question and the generator's last message, output these tags strictly: \n"
    "<verdict>correct|incorrect</verdict>\n"
    "<feedback>short, actionable critique; if incorrect, point to the precise mistake</feedback>\n"
    "<fixed_answer>if you can, provide the corrected final answer inside \\boxed{...}; otherwise empty</fixed_answer>"
)
TEACHER_SYSTEM = (
    "You are a senior math teacher acting as a turn-level oracle.\n"
    "Given the same context as the student, produce an IDEAL next assistant message for the current turn.\n"
    "Always reply in STRICT JSON and nothing else."
)
TEACHER_GEN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "g_next_message": {"type": "string"},
        "justification": {"type": "string"},
        "status": {"enum": ["agree", "revise"]},
    },
    "required": ["g_next_message", "status"],
    "additionalProperties": False,
}
TEACHER_VER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "v_next_message": {"type": "string"},
        "justification": {"type": "string"},
        "status": {"enum": ["agree", "revise"]},
    },
    "required": ["v_next_message", "status"],
    "additionalProperties": False,
}
VERIFIER_USER_TEMPLATE = "Question: {question}\n---\nGenerator message:\n{generator}"
JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)
BOXED = re.compile(r"\\boxed\{([^\}]+)\}")
MAX_TEACHER_HISTORY = 12
MAX_SFT_HISTORY = 9
MAX_ROLLOUT_HISTORY = 12
TEACHER_PARSE_RETRIES = 2


def parse_json_loose(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON text from teacher.")
    try:
        return json.loads(text)
    except Exception:
        pass
    for block in JSON_BLOCK.findall(text) or []:
        try:
            return json.loads(block)
        except Exception:
            continue
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(text[i : j + 1])
    raise ValueError(f"Failed to parse teacher JSON. Head: {text[:200]}")


async def teacher_label_for_generator(
    teacher_engine: VLLMChatEngine,
    question: str,
    prompt_messages: List[Dict[str, str]],
    last_g_output: str,
    ground_truth: Optional[Any] = None,
    retries: int = TEACHER_PARSE_RETRIES,
) -> Dict[str, Any]:
    gt_str = f"{ground_truth}" if ground_truth is not None else "UNKNOWN"
    hist_payload = prompt_messages
    user_prompt = {
        "role": "user",
        "content": (
            "Produce STRICT JSON for an ideal next *generator* message given the context.\n"
            "Keys: 'g_next_message', 'justification', 'status' (one of ['agree','revise']).\n\n"
            f"Question: {question}\n"
            f"Ground truth (may be UNKNOWN): {gt_str}\n"
            "History exactly as the student generator saw it at this turn (system/user/assistant, excluding its current reply): "
            f"{json.dumps(hist_payload, ensure_ascii=False)}\n\n"
            f"Student's last generator message: {last_g_output}"
        ),
    }
    messages = [{"role": "system", "content": TEACHER_SYSTEM}, user_prompt]

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            out = await call_engine(
                teacher_engine,
                messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                sp_extra={"guided_json": TEACHER_GEN_JSON_SCHEMA},
                chat_template_kwargs={"enable_thinking": False},
            )
            data = parse_json_loose(out["content"])
            if "g_next_message" not in data:
                raise KeyError(f"Teacher JSON missing 'g_next_message': {data}")
            return data
        except (ValueError, KeyError) as err:
            last_err = err
            logger.warning(
                "Teacher generator label parse failed (attempt %d/%d): %s",
                attempt,
                retries,
                err,
            )
            await asyncio.sleep(0)
    logger.error("Teacher generator labeling failed after %d attempts: %s", retries, last_err)
    return {}


async def teacher_label_for_verifier(
    teacher_engine: VLLMChatEngine,
    question: str,
    history_before_v: List[Dict[str, str]],
    g_output: str,
    retries: int = TEACHER_PARSE_RETRIES,
) -> Dict[str, Any]:
    hist_payload = history_before_v
    user_prompt = {
        "role": "user",
        "content": (
            "Produce STRICT JSON for an ideal *verifier* message for the given generator output.\n"
            "Keys: 'v_next_message', 'justification', 'status' (one of ['agree','revise']).\n"
            "The message must contain EXACTLY the three tags: <verdict>...</verdict>, <feedback>...</feedback>, <fixed_answer>...</fixed_answer>.\n"
            "In <fixed_answer>, if the correct final numeric answer is known, ALWAYS include it as \\boxed{...}; "
            "if the generator is already correct, copy the same \\boxed{...}; otherwise leave it empty.\n\n"
            f"Question: {question}\n"
            "Verifier-visible history EXACTLY as the student verifier saw it at this turn (system/user/assistant, excluding its current reply):\n"
            f"{json.dumps(hist_payload, ensure_ascii=False)}\n\n"
            f"Generator message: {g_output}"
        ),
    }
    messages = [{"role": "system", "content": TEACHER_SYSTEM}, user_prompt]

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            out = await call_engine(
                teacher_engine,
                messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                sp_extra={"guided_json": TEACHER_VER_JSON_SCHEMA},
                chat_template_kwargs={"enable_thinking": False},
            )
            data = parse_json_loose(out["content"])
            if "v_next_message" not in data:
                raise KeyError(f"Teacher JSON missing 'v_next_message': {data}")
            return data
        except (ValueError, KeyError) as err:
            last_err = err
            logger.warning(
                "Teacher verifier label parse failed (attempt %d/%d): %s",
                attempt,
                retries,
                err,
            )
            await asyncio.sleep(0)
    logger.error("Teacher verifier labeling failed after %d attempts: %s", retries, last_err)
    return {}
