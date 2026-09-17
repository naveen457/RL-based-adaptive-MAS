"""Date & Time Tools.

Provides live system date, time, weekday, and ISO timestamps for temporal awareness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def get_current_date(
    tz_name: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve the current live system date, time, weekday, and ISO timestamp.
    
    Args:
        tz_name: Optional timezone name (e.g., 'UTC'). Defaults to local system time.
        date_format: Optional strftime formatting string. Defaults to '%Y-%m-%d'.
    
    Returns:
        Structured dict containing formatted date, time, day of the week, and timestamp.
    """
    now = datetime.now() if not tz_name or tz_name.upper() != "UTC" else datetime.now(timezone.utc)
    fmt = date_format or "%Y-%m-%d"

    return {
        "current_date": now.strftime(fmt),
        "current_time": now.strftime("%H:%M:%S"),
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "iso_timestamp": now.isoformat(),
        "timezone": tz_name or "local",
        "status": "success",
    }
