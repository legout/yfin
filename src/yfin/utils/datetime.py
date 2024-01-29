import datetime as dt
import re

import pandas as pd
import pendulum as pdl


def datetime_from_timestamp(timestamp: int | float, tz: str = "UTC"):
    return pdl.from_timestamp(timestamp, tz=tz)


def to_timestamp(datetime: dt.datetime, timezone: str = "UTC"):
    if isinstance(datetime, str):
        datetime = datetime_from_string(
            datetime, tz=timezone, exact=False
        ).int_timestamp

    elif isinstance(datetime, pd.Timestamp):
        datetime = int(datetime.timestamp())
    elif isinstance(datetime, dt.datetime | dt.date):
        datetime = datetime_from_string(
            datetime.isoformat(), tz=timezone, exact=False
        ).int_timestamp
    elif isinstance(datetime, pdl.DateTime | pdl.Date):
        datetime = int(datetime.timestamp())
    return datetime


def datetime_from_string(
    timestamp: str,
    tz: str | None = None,
    exact: bool = True,
    strict: bool = False,
    naive: bool = False,
) -> pdl.datetime:
    tz = extract_timezone(timestamp) if tz is None else tz
    timestamp = timestamp.replace(tz, "").strip() if tz else timestamp

    timestamp = pdl.parse(timestamp, exact=exact, strict=strict)

    if isinstance(timestamp, pdl.DateTime):
        if tz is not None:
            timestamp = timestamp.naive().set(tz=tz)
        if naive or tz is None:
            timestamp = timestamp.naive()
        if exact:
            if timestamp.time() == pdl.Time(0, 0, 0):
                timestamp = timestamp.date()
    return timestamp


def timedelta_from_string(
    timedelta_string: str, as_timedelta
) -> pdl.Duration | dt.timedelta:
    """
    Converts a string like "2d10s" into a datetime.timedelta object.

    Args:
        string (str): The string representation of the timedelta, e.g. "2d10s".

    Returns:
        datetime.timedelta: The timedelta object.
    """
    # Extract the numeric value and the unit from the string
    matches = re.findall(r"(\d+)([a-zA-Z]+)", timedelta_string)
    if not matches:
        raise ValueError("Invalid timedelta string")

    # Initialize the timedelta object
    delta = pdl.duration()

    # Iterate over each match and accumulate the timedelta values
    for value, unit in matches:
        # Map the unit to the corresponding timedelta attribute
        unit_mapping = {
            "us": "microseconds",
            "ms": "milliseconds",
            "s": "seconds",
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "w": "weeks",
            "mo": "months",
            "y": "years",
        }
        if unit not in unit_mapping:
            raise ValueError("Invalid timedelta unit")

        # Update the timedelta object
        kwargs = {unit_mapping[unit]: int(value)}
        delta += pdl.duration(**kwargs)

    return delta.as_timedelta if as_timedelta else delta


def extract_timezone(timestamp_string):
    """
    Extracts the timezone from a timestamp string.

    Args:
        timestamp_string (str): The input timestamp string.

    Returns:
        str: The extracted timezone.
    """
    pattern = r"\b([a-zA-Z]+/{0,1}[a-zA-Z_ ]*)\b"  # Matches the timezone portion
    match = re.search(pattern, timestamp_string)
    if match:
        timezone = match.group(0)
        return timezone
    else:
        return None
