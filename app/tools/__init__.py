"""Tools Package for Adaptive Multi-Agent Systems.

Contains individual external tool implementations (search, calculators, etc.)
that can be registered to tool-calling nodes such as ToolExecutor.
"""

from app.tools.web_search import (
    duckduckgo_html_search,
    multi_query_search,
    tavily_search,
    web_search,
)
from app.tools.calculator import basic_calculator
from app.tools.date_time import get_current_date

__all__ = [
    "web_search",
    "multi_query_search",
    "tavily_search",
    "duckduckgo_html_search",
    "basic_calculator",
    "get_current_date",
]

