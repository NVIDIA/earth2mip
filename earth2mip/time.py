import datetime
import numpy as np


def convert_to_datetime(time) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(np.datetime_as_string(time, "s"))
    return dt
