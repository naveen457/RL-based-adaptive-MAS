import os
from dotenv import load_dotenv
load_dotenv()

from app.agents.planner import Planner
from app.agents.researcher import Researcher
from app.agents.coder import Coder
from app.agents.critic import Critic
from app.agents.finalizer import Finalizer

print("Testing Planner...")
try:
    p = Planner.from_settings()
    res = p.plan("Say hello")
    print("Planner success:", res)
except Exception as e:
    print("Planner error:", type(e), e)

print("\nTesting Researcher...")
try:
    r = Researcher.from_settings()
    res = r.research("Explain AI")
    print("Researcher success:", res)
except Exception as e:
    print("Researcher error:", type(e), e)

