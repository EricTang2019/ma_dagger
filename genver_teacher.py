from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rllm.rewards.math_utils.utils import extract_answer
from vllm_engine import VLLMChatEngine, call_engine

logger = logging.getLogger(__name__)

GEN_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{...}."
VER_SYSTEM = (
    "You are an exacting math verifier. Given the question and the generator's last message, output these tags strictly: \n"
    "<verdict>correct|incorrect</verdict>\n"
    "<feedback>short, actionable critique; if incorrect, point to the precise mistake</feedback>\n"
    "<fixed_answer>if you can, provide the corrected final answer inside \\boxed{...}; otherwise empty</fixed_answer>"
)
VERIFIER_USER_TEMPLATE = "Question: {question}\n---\nGenerator message:\n{generator}"
BOXED = re.compile(r"\\boxed\{([^\}]+)\}")
MAX_TEACHER_HISTORY = 12
MAX_SFT_HISTORY = 9
MAX_ROLLOUT_HISTORY = 12




async def teacher_label_for_generator(
    teacher_engine: VLLMChatEngine,
    student_messages: List[Dict[str, str]],
    **_: Any,
) -> Dict[str, Any]:
    """
    Return an ideal generator reply using exactly the same messages the student sees.

    This mirrors the PAG-style behavior: deterministic decode (temperature 0, top_p 1)
    with no extra system prompts or JSON schemas. Extra positional/keyword args are
    accepted and ignored for backward compatibility.
    """
    out = await call_engine(
        teacher_engine,
        student_messages,
        chat_template_kwargs={"enable_thinking": True},
        temperature=0.6,
        top_p=0.95,
        sp_extra={"top_k": 20, "min_p": 0.0},
    )
    return {
        "g_next_message": out["content"].strip(),
        "status": "revise",
        "justification": "",
    }


async def teacher_label_for_verifier(
    teacher_engine: VLLMChatEngine,
    student_messages: List[Dict[str, str]],
    **_: Any,
) -> Dict[str, Any]:
    """
    Return an ideal verifier reply using exactly the same messages the student sees.

    Deterministic decode; no extra formatting enforcement. Extra args are ignored
    for compatibility with older callers.
    """
    out = await call_engine(
        teacher_engine,
        student_messages,
        chat_template_kwargs={"enable_thinking": True},
        temperature=0.6,
        top_p=0.95,
        sp_extra={"top_k": 20, "min_p": 0.0},
    )
    return {
        "v_next_message": out["content"].strip(),
        "status": "revise",
        "justification": "",
    }
