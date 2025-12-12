from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from vllm_engine import VLLMChatEngine, call_engine

# Minimal answer extractor used across scripts (e.g., run_pag_numina.py)
import re


def extract_final_answer(text: str) -> str:
    """Heuristic to pull the final numeric/text answer from a model output."""
    if not text:
        return ""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    # fallback: last equality on a line
    m = re.search(r"(?m)^.*?=\s*([^\n]+)$", text)
    if m:
        return m.group(1).strip()
    return text.strip()

logger = logging.getLogger(__name__)

GEN_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{...}."
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
ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer|答案)[:：]?\s*([^\n]+)", re.IGNORECASE)
ANSWER_TAG = re.compile(r"<(?:final_answer|answer|result)>(.*?)</(?:final_answer|answer|result)>", re.IGNORECASE | re.DOTALL)
ANSWER_IS = re.compile(r"(?:^|\n|\.\s*)\s*the answer is\s*([^\n\.]+)", re.IGNORECASE)
JSON_ANSWER_KEYS = ("final_answer", "answer", "result", "ground_truth", "label", "output")
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


def _coerce_text(message: Any) -> str:
    """Flatten possibly-structured message/content into a single string."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        # Chat-style dicts
        if "content" in message:
            return _coerce_text(message.get("content"))
        if "text" in message:
            return _coerce_text(message.get("text"))
        return str(message)
    if isinstance(message, list):
        # Chat history: prefer the last assistant-like content if present
        msg_dicts = [m for m in message if isinstance(m, dict) and "content" in m]
        if msg_dicts:
            for m in reversed(msg_dicts):
                txt = _coerce_text(m.get("content"))
                if txt:
                    return txt
        parts = [(_coerce_text(m) or "").strip() for m in message]
        return "\n".join([p for p in parts if p])
    return str(message)


def _clean_answer(ans: Any) -> Optional[str]:
    """Normalize a candidate answer string."""
    if ans is None:
        return None
    if not isinstance(ans, str):
        ans = _coerce_text(ans)
    if not ans:
        return None
    # Trim common wrappers/punctuation without touching math content.
    ans = ans.replace("</s>", "").strip()
    if BOXED.search(ans):
        ans = BOXED.findall(ans)[-1]
    ans = ans.strip().strip("`").strip()
    ans = ans.rstrip("。.;；,，")
    return ans.strip() or None


def _extract_from_json_blob(text: str) -> Optional[str]:
    """Best-effort JSON answer extraction for cases where the model emits JSON."""
    parsed: Any = None
    for parser in (json.loads, parse_json_loose):
        try:
            parsed = parser(text)
            break
        except Exception:
            continue
    if parsed is None:
        return None

    def _walk(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            for key in JSON_ANSWER_KEYS:
                if key in obj:
                    cand = _clean_answer(obj[key])
                    if cand:
                        return cand
            for v in obj.values():
                cand = _walk(v)
                if cand:
                    return cand
        elif isinstance(obj, list):
            for item in reversed(obj):
                cand = _walk(item)
                if cand:
                    return cand
        return None

    return _walk(parsed)


def extract_final_answer(message: Any) -> Optional[str]:
    """
    Extract a concise final answer from a free-form model message.

    Heuristics (in order):
    - JSON keys like "final_answer"/"answer"/"result"
    - \\boxed{...}
    - <final_answer>...</final_answer> or answer tags
    - Lines starting with "Final answer"/"Answer"/"答案"
    - Phrases like "The answer is ..."
    - Fallback to the last non-empty line
    """
    text = _coerce_text(message)
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    json_ans = _extract_from_json_blob(text)
    if json_ans:
        return json_ans

    for pattern in (BOXED, ANSWER_TAG, ANSWER_LINE, ANSWER_IS):
        matches = pattern.findall(text)
        if matches:
            cand = _clean_answer(matches[-1])
            if cand:
                return cand

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        for sep in (":", "-", "->", "="):
            if sep in tail:
                tail = tail.split(sep)[-1].strip()
        cand = _clean_answer(tail)
        if cand:
            return cand
    return None


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
