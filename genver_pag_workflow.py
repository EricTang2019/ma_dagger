from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import TerminationReason, Workflow
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START

from genver_teacher import (
    GEN_SYSTEM,
    MAX_ROLLOUT_HISTORY,
    MAX_SFT_HISTORY,
    teacher_label_for_generator as teacher_label_for_generator_pag,
    teacher_label_for_verifier as teacher_label_for_verifier_pag,
)
from vllm_engine import call_engine, trim_history

PAG_VERIFY_SYSTEM = "You are the verifier model."
PAG_VERIFY_PROMPT = (
    "Check the generator's math solution step-by-step. "
    "If you find a mistake: state the wrong step and explain why it's wrong. "
    "If all steps are correct, explain why. "
    "Provide the reasoning and the final judgement of the correctness. "
    "The last word of your response must be either 'correct' or 'incorrect'."
)
PAG_REGENERATE_PROMPT = (
    "The verifier indicated that your previous answer was wrong. Provide the correct solution now. "
    "Keep reasoning concise, and put the final answer at the end in \\boxed{...}."
)


def _extract_judgment(text: str) -> Optional[str]:
    if not text:
        return None
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return None
    last = words[-1]
    if last == "wrong":
        return "incorrect"
    if last in {"correct", "incorrect"}:
        return last
    return None


def _strip_think(text: str) -> str:
    if not text:
        return text
    if THOUGHT_DELIMITER_END in text:
        return text.split(THOUGHT_DELIMITER_END, 1)[1].strip()
    return text


def _think_incomplete(text: str) -> bool:
    if not text:
        return False
    if THOUGHT_DELIMITER_END in text:
        return False
    return THOUGHT_DELIMITER_START in text


def _question_text_from_raw(raw_question: Any) -> str:
    """Best-effort extraction of the user question from various schemas."""
    try:
        import numpy as np  # type: ignore

        if isinstance(raw_question, np.ndarray):
            raw_question = raw_question.tolist()
    except Exception:
        pass

    def _content_from_dict(msg: Dict[str, Any]) -> str:
        return str(
            msg.get("content")
            or msg.get("text")
            or msg.get("value")
            or msg.get("message")
            or msg.get("prompt")
            or ""
        )

    def _is_userish(role: Optional[str]) -> bool:
        if not role:
            return False
        role = role.lower()
        return role in {"user", "human", "student", "questioner", "qa.user", "customer", "client"}

    if isinstance(raw_question, list):
        last_with_content = ""
        for msg in reversed(raw_question):
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("from") or msg.get("speaker") or msg.get("name")
                content = _content_from_dict(msg)
                if _is_userish(str(role)):
                    if content:
                        return content
                if content and not last_with_content:
                    last_with_content = content
            elif isinstance(msg, str) and msg.strip():
                return msg
        return last_with_content

    if isinstance(raw_question, dict):
        role = raw_question.get("role") or raw_question.get("from") or raw_question.get("speaker") or ""
        if _is_userish(str(role)):
            return _content_from_dict(raw_question)
        return str(
            raw_question.get("content")
            or raw_question.get("question")
            or raw_question.get("prompt")
            or raw_question.get("input")
            or ""
        )

    return str(raw_question or "")


class PAGStyleWorkflow(Workflow):
    """PAG-style multi-turn workflow (verify/regenerate prompts) using rLLM rewards."""

    def __init__(
        self,
        rollout_engine,  # unused
        executor,
        gen_engine,
        ver_engine,
        teacher_engine,
        max_turns: int = 2,
        collect_for: str = "both",
        reward_function=math_reward_fn,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)
        self.gen_engine = gen_engine
        self.ver_engine = ver_engine
        self.teacher_engine = teacher_engine
        self.max_turns = max_turns
        self.collect_for = collect_for
        self.reward_fn = reward_function

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)
        raw_question = task["question"]
        gt = task.get("ground_truth")
        # Always coerce raw_question into plain text
        question_text = _question_text_from_raw(raw_question)
        gen_hist: List[Dict[str, str]] = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": question_text},
        ]
        ver_hist: List[Dict[str, str]] = [{"role": "system", "content": PAG_VERIFY_SYSTEM}]

        gen_traj = Trajectory(name="generator", steps=[])
        ver_traj = Trajectory(name="verifier", steps=[])

        is_correct: bool = False

        for t in range(self.max_turns):
            # === Generator turn ===
            gen_prompt = trim_history(gen_hist, limit=MAX_ROLLOUT_HISTORY)
            gen_prompt_for_teacher = [dict(m) for m in gen_prompt]
            g_out = await call_engine(self.gen_engine, gen_prompt)
            g_msg_raw = g_out["content"].strip()
            think_incomplete = _think_incomplete(g_msg_raw)
            g_msg_public = _strip_think(g_msg_raw)
            gen_hist.append({"role": "assistant", "content": g_msg_public})
            gen_ctx = [dict(m) for m in trim_history(gen_hist, limit=MAX_SFT_HISTORY)]

            g_teacher = None
            if self.collect_for in ("both", "gen"):
                g_teacher = await teacher_label_for_generator_pag(
                    self.teacher_engine, gen_prompt_for_teacher
                )

            gen_traj.steps.append(
                Step(
                    chat_completions=gen_ctx,
                    reward=None,
                    info={
                        "turn_index": t,
                        "student_response": g_msg_public,
                        "student_response_raw": g_msg_raw,
                        "teacher_status": (g_teacher or {}).get("status"),
                        "teacher_justification": (g_teacher or {}).get("justification"),
                        "teacher_target": (g_teacher or {}).get("g_next_message"),
                    },
                )
            )

            # === Verifier turn ===
            ver_input_msg = g_msg_raw if think_incomplete else g_msg_public
            incomplete_note = ""
            if think_incomplete:
                incomplete_note = (
                    "\n\nNote: The generator's response contains an unfinished <think> block "
                    "(missing </think>). Treat this as incorrect, but still review the reasoning "
                    "and mention any other errors you find."
                )
            verify_user = (
                f"{PAG_VERIFY_PROMPT}{incomplete_note}\n\nQuestion:\n{question_text}\n\nPrevious answer:\n{ver_input_msg}"
            )
            ver_hist.append({"role": "user", "content": verify_user})
            ver_prompt = trim_history(ver_hist, limit=MAX_ROLLOUT_HISTORY)
            v_out = await call_engine(self.ver_engine, ver_prompt)
            v_msg_raw = v_out["content"].strip()
            v_msg_public = _strip_think(v_msg_raw)
            ver_hist.append({"role": "assistant", "content": v_msg_raw})

            v_teacher = None
            if self.collect_for in ("both", "ver"):
                v_teacher = await teacher_label_for_verifier_pag(
                    self.teacher_engine, ver_prompt
                )

            # pred_for_reward = g_msg
            reward_obj = self.reward_fn({"question": question_text, "ground_truth": gt}, g_msg_raw)
            ver_traj.steps.append(
                Step(
                    chat_completions=[
                        {"role": "user", "content": verify_user},
                        {"role": "assistant", "content": v_msg_public},
                    ],
                    reward=float(reward_obj.reward),
                    info={
                        "turn_index": t,
                        "student_response": v_msg_public,
                        "student_response_raw": v_msg_raw,
                        "generator_message": g_msg_public,
                        "generator_message_raw": g_msg_raw,
                        "teacher_status": (v_teacher or {}).get("status"),
                        "teacher_justification": (v_teacher or {}).get("justification"),
                        "teacher_target": (v_teacher or {}).get("v_next_message"),
                    },
                )
            )

            judgment = _extract_judgment(v_msg_raw)
            is_correct = judgment == "correct"

            if is_correct:
                break

            regen_user = (
                f"{PAG_REGENERATE_PROMPT}\n\nQuestion:\n{question_text}\n\nVerifier feedback:\n{v_msg_public}"
            )
            gen_hist.append({"role": "user", "content": regen_user})

        return Episode(
            id=uid,
            task=task,
            trajectories=[gen_traj, ver_traj],
            is_correct=bool(is_correct),
            metrics={},
            termination_reason=TerminationReason.ENV_DONE,
        )


async def rollout_with_pag_workflow_engine(
    tasks: List[Dict[str, Any]],
    gen_engine,
    ver_engine,
    teacher_engine,
    max_turns: int,
    collect_for: str,
    parallel: int = 64,
    retry: int = 5,
) -> List[Episode]:
    """Helper to run the PAG-style workflow with the existing AgentWorkflowEngine."""
    total_tasks = len(tasks)
    # A dedicated tqdm showing how many questions have been processed (1 bar = 1 question)
    pbar = tqdm(total=total_tasks, desc="PAG questions", position=1, leave=True, unit="q")
    last_done = 0

    def progress_cb(done: int, total: int):
        nonlocal last_done
        if total <= 0:
            return
        delta = done - last_done
        if delta > 0:
            pbar.update(delta)
            last_done = done

    engine = AgentWorkflowEngine(
        workflow_cls=PAGStyleWorkflow,
        workflow_args={
            "gen_engine": gen_engine,
            "ver_engine": ver_engine,
            "teacher_engine": teacher_engine,
            "max_turns": max_turns,
            "collect_for": collect_for,
        },
        rollout_engine=None,
        n_parallel_tasks=parallel,
        retry_limit=retry,
    )
    try:
        return await engine.execute_tasks(tasks, progress_callback=progress_cb)
    finally:
        pbar.close()


def pag_episodes_to_sft_rows(
    episodes: Sequence[Episode],
    collect_for: str = "both",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert PAG workflow episodes into generator/verifier SFT rows.

    We keep the exact chat context seen by the students (teacher targets replace
    the student assistant message). Rows without a teacher target are skipped.
    """
    gen_rows: List[Dict[str, Any]] = []
    ver_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        raw_q = ep.task.get("question") if isinstance(ep.task, dict) else getattr(ep.task, "question", None)
        question = _question_text_from_raw(raw_q)
        for traj in ep.trajectories:
            for step in traj.steps:
                info = step.info or {}
                teacher_target = info.get("teacher_target")
                if not teacher_target:
                    want_label = (
                        (traj.name == "generator" and collect_for in {"both", "gen"})
                        or (traj.name == "verifier" and collect_for in {"both", "ver"})
                    )
                    if want_label:
                        print(
                            f"[warn] Missing teacher_target for {traj.name} "
                            f"(ep={getattr(ep, 'id', '?')}, turn={info.get('turn_index')})"
                        )
                    continue
                ctx = step.chat_completions or []
                cleaned_ctx = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in ctx
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content") is not None
                ]
                if cleaned_ctx and cleaned_ctx[-1]["role"] == "assistant":
                    cleaned_ctx = cleaned_ctx[:-1]
                if traj.name == "generator":
                    if not question:
                        continue  # skip unusable rows with missing question
                    if not cleaned_ctx:
                        cleaned_ctx = [
                            {"role": "system", "content": GEN_SYSTEM},
                            {"role": "user", "content": question},
                        ]
                    elif cleaned_ctx[-1]["role"] != "user":
                        cleaned_ctx.append({"role": "user", "content": question})
                    gen_rows.append({"messages": cleaned_ctx + [{"role": "assistant", "content": teacher_target}]})
                elif traj.name == "verifier":
                    generator_msg = info.get("generator_message", "")
                    if not question or not generator_msg:
                        continue  # skip broken rows
                    verify_user = f"{PAG_VERIFY_PROMPT}\n\nQuestion:\n{question}\n\nPrevious answer:\n{generator_msg}"
                    ver_rows.append(
                        {
                            "messages": [
                                {"role": "system", "content": PAG_VERIFY_SYSTEM},
                                {"role": "user", "content": verify_user},
                                {"role": "assistant", "content": teacher_target},
                            ]
                        }
                    )
    return gen_rows, ver_rows


def dump_pag_transcripts_with_raw(
    episodes: Sequence[Episode],
    path: str,
):
    """Dump raw/public messages for generator/verifier for debugging."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for ep in episodes:
            rec: Dict[str, Any] = {
                "episode_id": getattr(ep, "id", None),
                "task": ep.task if hasattr(ep, "task") else None,
                "trajectories": [],
            }
            for traj in getattr(ep, "trajectories", []) or []:
                traj_rec: Dict[str, Any] = {"name": getattr(traj, "name", None), "steps": []}
                for step in getattr(traj, "steps", []) or []:
                    info = step.info or {}
                    traj_rec["steps"].append(
                        {
                            "turn_index": info.get("turn_index"),
                            "student_response": info.get("student_response"),
                            "student_response_raw": info.get("student_response_raw"),
                            "generator_message": info.get("generator_message"),
                            "generator_message_raw": info.get("generator_message_raw"),
                            "teacher_target": info.get("teacher_target"),
                            "chat_completions": getattr(step, "chat_completions", None),
                        }
                    )
                rec["trajectories"].append(traj_rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
