"""FastAPI server for Adaptive Multi-Agent System (RL-AMAS).

Connects frontend/backend chat requests to the dynamic multi-agent orchestrator.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure workspace root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.settings import settings
from app.agents.finalizer import format_final_response
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator
from app.evaluation.metrics_logger import MetricsLogger
from app.rl.q_learning import QLearningPolicy
from app.memory.mongo_client import get_mongo_client
from app.evaluation.reward import DualObjectiveRewardCalculator
from app.architecture.models import MASArchitecture
from app.rl.state import ArchitectureStateEncoder
from app.rl.meta_task import MetaTaskContext
from app.graph.visualizer import get_graph_png_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rl_amas_api")

# Global runtime state
state: Dict[str, Any] = {
    "orchestrator": None,
    "q_policy": None,
    "metrics_logger": None,
    "mongo_client": None,
    "step_counter": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system dependencies, RL policy, and orchestrator on startup."""
    logger.info("Initializing RL_Based_AMAS runtime...")
    
    # 1. MongoDB Atlas
    mongo_client = get_mongo_client(required=False)
    state["mongo_client"] = mongo_client

    # 2. Metrics Logger
    metrics_logger = MetricsLogger(log_base_dir="runs")
    state["metrics_logger"] = metrics_logger

    # 3. Persistent Q-learning policy
    q_policy = QLearningPolicy(task_aware=True, epsilon=0.1)
    if mongo_client is not None:
        loaded = q_policy.load_from_mongodb()
        if loaded and q_policy.q_table.num_state_action_pairs() > 0:
            logger.info("Loaded Q-table (%d entries) from MongoDB Atlas", q_policy.q_table.num_state_action_pairs())
        else:
            logger.info("Initialized fresh Q-learning policy in MongoDB Atlas")
    else:
        q_table_path = Path("data/q_table.json")
        if q_table_path.exists():
            q_policy.load_q_table(str(q_table_path))
            logger.info("Loaded Q-table (%d entries) from %s", q_policy.q_table.num_state_action_pairs(), q_table_path)
        else:
            logger.info("Initialized in-memory Q-learning policy")
    state["q_policy"] = q_policy

    # 4. Orchestrator
    orchestrator = AdaptiveRuntimeOrchestrator.from_settings(q_policy=q_policy)
    state["orchestrator"] = orchestrator
    logger.info("RL_Based_AMAS orchestrator initialized successfully.")

    yield

    # Teardown
    if state.get("metrics_logger"):
        state["metrics_logger"].close()
    logger.info("RL_Based_AMAS runtime stopped.")


app = FastAPI(
    title="RL_Based_AMAS API",
    description="Adaptive Multi-Agent System HTTP API for Astrix Chat and Topology Visualization",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat query or task")
    conversationId: Optional[str] = Field(None, description="Conversation / thread ID")
    thread_id: Optional[str] = Field(None, description="Alias for conversation ID")
    user_id: Optional[str] = Field(None, description="Authenticated user ID")


class ChatResponse(BaseModel):
    content: str
    conversationId: str
    architecture_version: int = 0
    agents_invoked: List[str] = Field(default_factory=list)
    tools_executed: List[str] = Field(default_factory=list)
    latest_graph: Optional[str] = Field(None, description="Base64 data URL for the latest architecture graph PNG")
    status: str = "success"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RL_Based_AMAS",
        "version": "1.0.0",
        "architecture_version": state["orchestrator"].current_version if state.get("orchestrator") else 0,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "orchestrator_ready": state.get("orchestrator") is not None,
    }


@app.get("/architecture")
def get_architecture():
    orchestrator: AdaptiveRuntimeOrchestrator = state.get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return {
        "architecture": orchestrator.current_architecture,
        "version": orchestrator.current_version,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    orchestrator: AdaptiveRuntimeOrchestrator = state.get("orchestrator")
    q_policy: QLearningPolicy = state.get("q_policy")
    metrics_logger: MetricsLogger = state.get("metrics_logger")

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator is not ready")

    task = (request.message or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="A message is required")

    thread_id = request.conversationId or request.thread_id or "thread-1"
    start_time = time.perf_counter()
    state["step_counter"] += 1
    step = state["step_counter"]

    logger.info("[Task #%d | Thread: %s | User: %s] Processing: '%s'", step, thread_id, request.user_id, task[:80])

    try:
        result = orchestrator.run_dynamic(task, thread_id=thread_id, user_id=request.user_id)
    except Exception as exc:
        logger.error("Error executing dynamic runtime: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"AMAS execution error: {str(exc)}")

    final_resp = result.final_response
    content = format_final_response(final_resp)
    final_arch = result.final_architecture
    invoked = result.agents_actually_invoked or ["planner", "finalizer"]
    tools_executed = getattr(result, "tools_executed", [])

    # Extract required capabilities
    plan_dict = result.planner_output or {}
    req_caps = list(plan_dict.get("required_capabilities") or [])
    tools_needed = list(plan_dict.get("tools_needed") or [])

    tool_to_cap = {
        "web_search": "web_search",
        "arxiv_search": "research",
        "calculator": "math",
        "code_interpreter": "coding",
        "retriever": "research",
    }
    for tool in tools_needed:
        cap = tool_to_cap.get(tool, tool)
        if cap not in req_caps:
            req_caps.append(cap)

    for key, val in plan_dict.items():
        if key.startswith("requires_") and val:
            cap_name = key[len("requires_"):]
            if cap_name == "tools":
                cap_name = "web_search"
            if cap_name not in req_caps:
                req_caps.append(cap_name)

    # Log metrics & RL reward in background / non-blocking
    try:
        logged = {}
        if metrics_logger:
            logged = metrics_logger.log_task_run(
                step=step,
                task=task,
                result=result,
                architecture=final_arch,
                required_capabilities=req_caps,
                user_id=request.user_id,
                thread_id=thread_id,
            )

        dual_calc = DualObjectiveRewardCalculator()
        reward_info = dual_calc.calculate(
            required_capabilities=req_caps,
            active_agents=[a["agent_id"] for a in final_arch.get("agents", []) if a.get("active")],
            invoked_agents=invoked,
            tools_executed=tools_executed,
            coverage_score=logged.get("coverage_score", 1.0),
            missing_capabilities=logged.get("missing_capabilities", []),
            surplus_agents=logged.get("surplus_agents", []),
            user_feedback="",
            final_response=result.final_response,
        )

        step_reward = reward_info["total_reward"]
        comps = reward_info["components"]

        if metrics_logger:
            metrics_logger.log_reward(
                step=step,
                r_arch=reward_info["r_arch"],
                r_resp=reward_info["r_resp"],
                total_reward=step_reward,
                components=comps,
            )
            if metrics_logger.runs_collection is not None:
                try:
                    metrics_logger.runs_collection.update_one(
                        {"step": step, "session_id": metrics_logger.session_id},
                        {"$set": {"total_reward": step_reward, "r_arch": reward_info["r_arch"], "r_resp": reward_info["r_resp"]}}
                    )
                except Exception:
                    pass

        # Update Q-policy
        if q_policy:
            init_arch_raw = getattr(result, "initial_architecture", final_arch)
            init_arch_obj = MASArchitecture.model_validate(init_arch_raw)
            final_arch_obj = MASArchitecture.model_validate(final_arch)

            known_categories = ["research", "coding", "math", "analysis", "data", "tool_use", "verification"]
            task_category = next((c for c in req_caps if c in known_categories), "general")
            task_ctx = MetaTaskContext(
                task_category=task_category,
                required_capabilities=sorted(req_caps),
                difficulty=1,
            )

            obs_init = ArchitectureStateEncoder(init_arch_obj).encode()
            state_key = q_policy.get_state_key(obs_init, task_ctx)

            obs_final = ArchitectureStateEncoder(final_arch_obj).encode()
            next_state_key = q_policy.get_state_key(obs_final, task_ctx)

            chosen_action_id = result.action_ids[0] if getattr(result, "action_ids", None) else 0

            q_policy.update(
                state_key=state_key,
                action_id=chosen_action_id,
                reward=step_reward,
                next_state_key=next_state_key,
                next_valid_actions=getattr(result, "action_ids", [0]) or [0],
                terminated=True,
                truncated=False,
            )
            q_policy.save_to_mongodb()
    except Exception as log_err:
        logger.warning("RL feedback/logging step error (non-fatal): %s", log_err)

    duration = time.perf_counter() - start_time
    logger.info("[Task #%d completed in %.2fs] Response length: %d chars", step, duration, len(content))

    # Generate and store ONLY latest graph PNG in MongoDB Atlas for this thread/user
    latest_graph_url = None
    try:
        target_graph_input = invoked if invoked else final_arch
        png_bytes = get_graph_png_bytes(target_graph_input, tools_executed=tools_executed)
        if png_bytes:
            b64_str = base64.b64encode(png_bytes).decode("utf-8")
            orchestrator.thread_store.save_latest_graph(
                thread_id=thread_id,
                graph_png_base64=b64_str,
                user_id=request.user_id,
            )
            latest_graph_url = f"data:image/png;base64,{b64_str}"
    except Exception as graph_err:
        logger.warning("Graph PNG generation/mongo save error (non-fatal): %s", graph_err)

    return ChatResponse(
        content=content,
        conversationId=thread_id,
        architecture_version=getattr(result, "architecture_version", 0),
        agents_invoked=invoked,
        tools_executed=tools_executed,
        latest_graph=latest_graph_url,
        status="success",
    )


class FeedbackRequest(BaseModel):
    thread_id: str
    feedback: str = Field(..., description="'p'/'y' for positive, 'n' for negative, 'over' or 'under'")
    comment: Optional[str] = None


@app.get("/threads")
def list_threads(user_id: Optional[str] = None):
    orchestrator: AdaptiveRuntimeOrchestrator = state.get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    threads = orchestrator.thread_store.list_threads(user_id=user_id)
    thread_list = []
    for tid in threads:
        stats = orchestrator.thread_store.get_thread_stats(tid, user_id=user_id)
        msgs = orchestrator.thread_store.get_messages(tid, user_id=user_id)
        snippet = ""
        for m in msgs:
            content = getattr(m, "content", "")
            if getattr(m, "type", "") in {"human", "user"} and content:
                snippet = str(content)[:60]
                break
        if not snippet and msgs:
            snippet = str(getattr(msgs[-1], "content", ""))[:60]
        thread_list.append({
            "thread_id": tid,
            "message_count": stats.get("message_count", len(msgs)),
            "snippet": snippet or "New Thread",
        })
    return {"threads": thread_list}


@app.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str, user_id: Optional[str] = None):
    orchestrator: AdaptiveRuntimeOrchestrator = state.get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    msgs = orchestrator.thread_store.get_serialized_messages(thread_id, user_id=user_id)
    return {"thread_id": thread_id, "messages": msgs}


@app.post("/chat/feedback")
def submit_feedback(req: FeedbackRequest):
    logger.info("Feedback received for thread '%s': %s", req.thread_id, req.feedback)
    clean_fb = req.feedback.lower().strip()
    if clean_fb in {"like", "thumbs_up", "up", "y", "p"}:
        code = "p"
        reward_delta = 0.5
    elif clean_fb in {"over", "verbose"}:
        code = "over"
        reward_delta = -0.4
    elif clean_fb in {"under", "missing"}:
        code = "under"
        reward_delta = -0.5
    else:
        code = "n"
        reward_delta = -0.6

    return {
        "status": "success",
        "thread_id": req.thread_id,
        "recorded_feedback": code,
        "reward_delta": reward_delta,
    }


@app.get("/threads/{thread_id}/graph")
def get_thread_graph(thread_id: str, user_id: Optional[str] = None):
    orchestrator: AdaptiveRuntimeOrchestrator = state.get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    graph_b64 = orchestrator.thread_store.get_latest_graph(thread_id, user_id=user_id)
    if not graph_b64:
        return {"thread_id": thread_id, "graph": None}
    return {
        "thread_id": thread_id,
        "graph": f"data:image/png;base64,{graph_b64}",
    }


@app.get("/metrics")
def get_metrics(user_id: Optional[str] = None, thread_id: Optional[str] = None):
    """Retrieve multi-agent execution and RL adaptation metrics strictly scoped to user_id."""
    metrics_logger: MetricsLogger = state.get("metrics_logger")
    if not metrics_logger:
        raise HTTPException(status_code=503, detail="Metrics logger not initialized")
    if not user_id:
        return {
            "user_id": None,
            "thread_id": thread_id,
            "summary": {
                "total_runs": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_coverage": 0.0,
                "avg_net_utility": 0.0,
                "avg_critical_path": 0.0,
                "avg_reward": 0.0,
            },
            "agent_invocations": {},
            "complexity_breakdown": {},
            "runs": [],
        }
    return metrics_logger.get_user_metrics(user_id=user_id, thread_id=thread_id)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port)

