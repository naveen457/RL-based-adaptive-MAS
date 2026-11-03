"""Test single live LLM call to NVIDIA NIM."""
import time
from app.config.settings import settings
from app.agents.planner import Planner

print(f"Testing live NVIDIA NIM model: {settings.model}")
print(f"Endpoint: {settings.base_url}")

start = time.perf_counter()
planner = Planner.from_settings()
output = planner.plan("Hello! How can you help me today?")
dur = time.perf_counter() - start

print(f"Success! Response received in {dur:.2f}s:")
print(f"Task understanding: {output.task_understanding}")
print(f"Complexity: {output.estimated_complexity}")
print(f"Required capabilities: {output.required_capabilities}")

