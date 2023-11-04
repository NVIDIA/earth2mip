import io
import datetime
import pandas as pd
import os
from earth2mip import forecast_metrics_io


def test_read_metric_empty(tmpdir):
    series = forecast_metrics_io.read_metrics(tmpdir)
    assert series.empty


def test_write_and_read_metric(tmpdir):
    # Write a metric to the file object
    initial_time = datetime.datetime(2022, 1, 1, 0, 0, 0)
    lead_time = datetime.timedelta(hours=24)
    channel = "t2m"
    metric = "rmse"
    value = 25.6

    # Read the metric from the directory
    directory = tmpdir.mkdir("metrics")
    filename = "test_metric.csv"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        forecast_metrics_io.write_metric(
            f, initial_time, lead_time, channel, metric, value
        )
    metrics = forecast_metrics_io.read_metrics(directory)

    # Check that the metric was written correctly
    expected_index = pd.MultiIndex.from_tuples(
        [(initial_time, lead_time, channel, metric)],
        names=["initial_time", "lead_time_hours", "channel", "metric"],
    )
    expected_value = pd.Series([value], index=expected_index, name="value")
    pd.testing.assert_series_equal(metrics, expected_value)
