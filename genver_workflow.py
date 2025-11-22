from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import TerminationReason, Workflow

from genver_teacher import (
    BOXED,
    GEN_SYSTEM,
    MAX_ROLLOUT_HISTORY,
    MAX_SFT_HISTORY,
    VER_SYSTEM,
    VERIFIER_USER_TEMPLATE,
    teacher_label_for_generator,
    teacher_label_for_verifier,
)
from vllm_engine import ChatEngineProtocol, VLLMChatEngine, call_engine, trim_history


class GenVerDaggerWorkflow(Workflow):
    """rLLM AgentWorkflowEngine-compatible workflow for joint Gen/Ver rollout."""

    def __init__(
        self,
        rollout_engine,  # unused
        executor,
        gen_engine: VLLMChatEngine,
        ver_engine: VLLMChatEngine,
        teacher_engine: ChatEngineProtocol,
        max_turns: int = 4,
        stop_on_verifier_fix: bool = False,
        collect_for: str = "both",
        reward_function=math_reward_fn,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)
        self.gen_engine = gen_engine
        self.ver_engine = ver_engine
        self.teacher_engine = teacher_engine
        self.max_turns = max_turns
        self.stop_on_verifier_fix = stop_on_verifier_fix
        self.collect_for = collect_for
        self.reward_fn = reward_function

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)
        question = task["question"]
        gt = task.get("ground_truth")
        gen_hist: List[Dict[str, str]] = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": question},
        ]
        ver_hist: List[Dict[str, str]] = [{"role": "system", "content": VER_SYSTEM}]

        gen_traj = Trajectory(name="generator", steps=[])
        ver_traj = Trajectory(name="verifier", steps=[])
        fixed_answer: Optional[str] = None
        is_correct: Optional[bool] = None

        for t in range(self.max_turns):
            gen_prompt = trim_history(gen_hist, limit=MAX_ROLLOUT_HISTORY)
            gen_prompt_for_teacher = [dict(m) for m in gen_prompt]
            g_out = await call_engine(self.gen_engine, gen_prompt)
            g_msg = g_out["content"].strip()
            gen_hist.append({"role": "assistant", "content": g_msg})
            gen_ctx = [dict(m) for m in trim_history(gen_hist, limit=MAX_SFT_HISTORY)]

            g_teacher = None
            if self.collect_for in ("both", "gen"):
                g_teacher = await teacher_label_for_generator(
                    self.teacher_engine,
                    question,
                    gen_prompt_for_teacher,
                    g_msg,
                    gt,
                )

            gen_traj.steps.append(
                Step(
                    chat_completions=gen_ctx,
                    reward=None,
                    info={
                        "turn_index": t,
                        "student_response": g_msg,
                        "teacher_status": (g_teacher or {}).get("status"),
                        "teacher_justification": (g_teacher or {}).get("justification"),
                        "teacher_target": (g_teacher or {}).get("g_next_message"),
                    },
                )
            )

            ver_user = VERIFIER_USER_TEMPLATE.format(question=question, generator=g_msg)
            ver_hist.append({"role": "user", "content": ver_user})
            ver_prompt = trim_history(ver_hist, limit=MAX_ROLLOUT_HISTORY)
            ver_prompt_for_teacher = [dict(m) for m in ver_prompt]
            v_out = await call_engine(self.ver_engine, ver_prompt)
            v_msg = v_out["content"].strip()
            ver_hist.append({"role": "assistant", "content": v_msg})

            fixed_answer = None
            m = re.search(r"<fixed_answer>(.*?)</fixed_answer>", v_msg, re.IGNORECASE | re.DOTALL)
            if m:
                fixed_answer = m.group(1).strip()

            v_teacher = None
            if self.collect_for in ("both", "ver"):
                v_teacher = await teacher_label_for_verifier(
                    self.teacher_engine,
                    question,
                    ver_prompt_for_teacher,
                    g_msg,
                )

            pred = None
            if fixed_answer and BOXED.search(fixed_answer):
                pred = BOXED.search(fixed_answer).group(1)
            elif BOXED.search(g_msg):
                pred = BOXED.search(g_msg).group(1)
            reward_obj = self.reward_fn({"question": question, "ground_truth": gt}, pred or "")

            ver_traj.steps.append(
                Step(
                    chat_completions=[
                        {"role": "user", "content": ver_user},
                        {"role": "assistant", "content": v_msg},
                    ],
                    reward=float(reward_obj.reward),
                    info={
                        "turn_index": t,
                        "student_response": v_msg,
                        "generator_message": g_msg,
                        "teacher_status": (v_teacher or {}).get("status"),
                        "teacher_justification": (v_teacher or {}).get("justification"),
                        "teacher_target": (v_teacher or {}).get("v_next_message"),
                    },
                )
            )

            # Feed verifier feedback back to the generator as the next user turn
            feedback_for_gen = v_msg or ver_user
            gen_hist.append({"role": "user", "content": feedback_for_gen})

            if self.stop_on_verifier_fix and fixed_answer and BOXED.search(fixed_answer):
                is_correct = bool(reward_obj.is_correct)
                break

        if is_correct is None:
            is_correct = any(step.reward and step.reward > 0.5 for step in ver_traj.steps)

        return Episode(
            id=uid,
            task=task,
            trajectories=[gen_traj, ver_traj],
            is_correct=bool(is_correct),
            metrics={},
            termination_reason=TerminationReason.ENV_DONE,
        )


async def rollout_with_workflow_engine(
    tasks: List[Dict[str, Any]],
    gen_engine: VLLMChatEngine,
    ver_engine: VLLMChatEngine,
    teacher_engine: ChatEngineProtocol,
    max_turns: int,
    stop_on_verifier_fix: bool,
    collect_for: str,
    parallel: int = 64,
    retry: int = 5,
) -> List[Episode]:
    engine = AgentWorkflowEngine(
        workflow_cls=GenVerDaggerWorkflow,
        workflow_args={
            "gen_engine": gen_engine,
            "ver_engine": ver_engine,
            "teacher_engine": teacher_engine,
            "max_turns": max_turns,
            "stop_on_verifier_fix": stop_on_verifier_fix,
            "collect_for": collect_for,
        },
        rollout_engine=None,
        n_parallel_tasks=parallel,
        retry_limit=retry,
    )
    return await engine.execute_tasks(tasks)


def episodes_to_sft_rows(episodes: Sequence[Episode]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen_rows, ver_rows = [], []
    for ep in episodes:
        q = ep.task["question"]
        for traj in ep.trajectories:
            for step in traj.steps:
                info = step.info or {}
                teacher_target = info.get("teacher_target")
                if not teacher_target:
                    continue
                if traj.name == "generator":
                    context_msgs = step.chat_completions or []
                    cleaned_ctx = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in context_msgs
                        if isinstance(msg, dict) and msg.get("role") and msg.get("content") is not None
                    ]
                    if cleaned_ctx and cleaned_ctx[-1]["role"] == "assistant":
                        cleaned_ctx = cleaned_ctx[:-1]
                    if not cleaned_ctx:
                        cleaned_ctx = [
                            {"role": "system", "content": GEN_SYSTEM},
                            {"role": "user", "content": q},
                        ]
                    elif cleaned_ctx[-1]["role"] != "user":
                        cleaned_ctx.append({"role": "user", "content": q})
                    row_msgs = cleaned_ctx + [{"role": "assistant", "content": teacher_target}]
                    gen_rows.append({"messages": row_msgs})
                elif traj.name == "verifier":
                    ver_user = VERIFIER_USER_TEMPLATE.format(
                        question=q,
                        generator=info.get("generator_message", ""),
                    )
                    ver_rows.append(
                        {
                            "messages": [
                                {"role": "system", "content": VER_SYSTEM},
                                {"role": "user", "content": ver_user},
                                {"role": "assistant", "content": teacher_target},
                            ]
                        }
                    )
    return gen_rows, ver_rows
