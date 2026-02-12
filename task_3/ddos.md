# Task 3 — Web Server Log Analysis (DDoS Detection using Regression)

## Objective

The goal of this task is to analyze a web server log file and identify time interval(s) corresponding to a Distributed Denial-of-Service (DDoS) attack. The detection is performed using statistical analysis and regression-based anomaly detection.

---

## Input Log File

- Original source:
  http://max.ge/aiml_final/e_bakuradze25_23456_server.log

-- I couldnt upload log file in Guthub so I left here only the initial link provided. But Ive added the log file in pychatrm to work on it

<img width="1895" height="937" alt="image" src="https://github.com/user-attachments/assets/84284b81-81c0-4cf0-8611-583ea3f567c4" />

please see the error it gives me when trying to uploade the event log file
<img width="1085" height="691" alt="image" src="https://github.com/user-attachments/assets/5108bddc-f13c-473f-9390-659eb790be35" />



The log contains IP addresses, timestamps, HTTP methods, status codes, and user-agent strings.

---

## Methodology

### 1) Timestamp extraction

Each log entry contains a timestamp such as:

```
[2024-03-22 18:00:53+04:00]
```

A regular expression parser extracts timestamps and converts them to timezone-aware datetime objects.

---

### 2) Traffic aggregation

To analyze traffic intensity:

- Requests are aggregated into **30-second intervals**
- Each interval contains a request count
- This produces a time-series of traffic volume

This step converts raw logs into analyzable statistical data.

---

### 3) Regression baseline (normal traffic model)

A regression model is used to estimate normal behavior:

- Linear Regression models expected request volume
- RANSAC (robust regression) ignores extreme spikes
- Cyclical time features (sin/cos) capture daily patterns

The regression output represents expected traffic under normal conditions.

---

### 4) DDoS detection via residual analysis

An anomaly is detected when:

```
Residual = Actual − Predicted
```

The residuals are standardized using a z-score:

```
z = residual / standard deviation
```

If:

```
z ≥ 2.5
```

the interval is flagged as abnormal.

Consecutive abnormal intervals are merged into attack windows.

---

## Results — Detected DDoS Interval(s)

The regression-based anomaly detection identified:

**DDoS Attack Interval**

- **2024-03-22 18:31:00+04:00 → 2024-03-22 18:33:00+04:00**

During this period, request volume significantly exceeded the regression baseline, indicating a probable DDoS burst.

---

## Visualizations

### Requests per 30-second interval

Shows raw traffic intensity.

[![Requests](images/req_per_min.png)](https://github.com/EleneBa/aimlFin2026_e_bakuradze25/blob/main/task_3/req_per_min.png)

---

### Regression baseline vs actual traffic

Highlighted regions indicate detected attack intervals.

[![Regression](images/regression_fit.png)](https://github.com/EleneBa/aimlFin2026_e_bakuradze25/blob/main/task_3/regression_fit.png) 

---

### Residual z-score

Values above threshold indicate anomalies.

[![Residuals](images/residuals.png)](https://github.com/EleneBa/aimlFin2026_e_bakuradze25/blob/main/task_3/residuals.png) 

---

## Main Code Fragments

### Aggregation

```python
counts = df.resample("30s").size().rename("requests").to_frame()
```

---

### Regression baseline

```python
model = RANSACRegressor(
    estimator=LinearRegression(),
    random_state=42
)
model.fit(X, y)
y_pred = model.predict(X)
```

---

### Anomaly detection

```python
residual = y - y_pred
z = residual / residual.std(ddof=1)
flagged = counts[z >= 2.5]
```

---

## Reproducibility

To reproduce this analysis:

### Step 1 — Install dependencies

```
pip install numpy pandas matplotlib scikit-learn
```

---

### Step 2 — Place log file

Ensure the log file is inside:

```
task_3/
```

---

### Step 3 — Run analysis

```
python detect_ddos_regression.py
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RANSACRegressor


# ----------------------------
# Config
# ----------------------------
LOG_PATH = Path("e_bakuradze25_23456_server.log")  # must be in task_3/
OUT_DIR = Path("images")
OUT_DIR.mkdir(exist_ok=True)

AGG_FREQ = "30s"         # requests per minute
Z_THRESHOLD = 2.5         # residual z-score threshold
MIN_CONSECUTIVE = 3       # minimum consecutive minutes for an interval


# ----------------------------
# Helpers
# ----------------------------
# Your log timestamp format example: [2024-03-22 18:00:53+04:00]
TS_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:\.\d+)?([+\-]\d{2}:\d{2})\]"
)

def parse_timestamp(line: str):
    """
    Parses timestamps like:
      [2024-03-22 18:00:53+04:00]
      [2024-03-22T18:00:53+04:00]
      optionally with microseconds: [2024-03-22 18:00:53.123+04:00]
    Returns timezone-aware datetime or None.
    """
    m = TS_RE.search(line)
    if not m:
        return None

    dt_part = m.group(1).replace("T", " ")  # normalize
    tz_part = m.group(2)

    # If microseconds were present, regex stripped them, so format is stable
    return datetime.strptime(dt_part + tz_part, "%Y-%m-%d %H:%M:%S%z")


def merge_intervals(times: pd.DatetimeIndex, step: pd.Timedelta):
    """
    Merge consecutive timestamps spaced by <= step into intervals [start, end).
    """
    if len(times) == 0:
        return []

    times = times.sort_values()
    intervals = []
    start = times[0]
    prev = times[0]

    for t in times[1:]:
        if (t - prev) <= step:
            prev = t
        else:
            intervals.append((start, prev + step))
            start = t
            prev = t

    intervals.append((start, prev + step))
    return intervals


# ----------------------------
# Main
# ----------------------------
def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH.resolve()}")

    # 1) Parse timestamps
    timestamps = []
    with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = parse_timestamp(line)
            if ts is not None:
                timestamps.append(ts)

    if not timestamps:
        raise ValueError(
            "No timestamps parsed. Check the timestamp format in the log. "
            "Expected like: [YYYY-MM-DD HH:MM:SS+TZ]"
        )

    df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
    df = df.set_index("ts").sort_index()

    # 2) Aggregate requests per minute
    counts = df.resample(AGG_FREQ).size().rename("requests").to_frame()
    counts = counts[counts["requests"] > 0]

    if len(counts) < 10:
        raise ValueError(
            f"Not enough aggregated points for regression (got {len(counts)}). "
            "Try a larger time window (e.g., AGG_FREQ='5min') or confirm log content."
        )

    # 3) Build regression features
    t0 = counts.index.min()
    seconds = (counts.index - t0).total_seconds().values.reshape(-1, 1)

    minute_of_day = (counts.index.hour * 60 + counts.index.minute).values
    sin_day = np.sin(2 * np.pi * minute_of_day / 1440.0).reshape(-1, 1)
    cos_day = np.cos(2 * np.pi * minute_of_day / 1440.0).reshape(-1, 1)

    X = np.hstack([seconds, sin_day, cos_day])
    y = counts["requests"].values

    # 4) Robust regression baseline (RANSAC)
    base = LinearRegression()
    model = RANSACRegressor(estimator=base, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    residual = y - y_pred

    # 5) Thresholding on residuals (z-score)
    resid_std = float(np.std(residual, ddof=1))
    if resid_std == 0 or np.isnan(resid_std):
        resid_std = 1.0

    z = residual / resid_std

    counts["predicted"] = y_pred
    counts["residual"] = residual
    counts["z"] = z

    flagged = counts[counts["z"] >= Z_THRESHOLD]
    step = pd.to_timedelta(AGG_FREQ)

    # Merge consecutive flagged minutes into intervals
    intervals = merge_intervals(flagged.index, step)

    # Keep only intervals with enough consecutive bins
    strong_intervals = []
    for a, b in intervals:
        bins = int((b - a) / step)
        if bins >= MIN_CONSECUTIVE:
            strong_intervals.append((a, b))

    # 6) Visualizations
    # (a) request series
    plt.figure()
    plt.plot(counts.index, counts["requests"])
    plt.title(f"Requests per {AGG_FREQ}")
    plt.xlabel("Time")
    plt.ylabel(f"Requests / {AGG_FREQ}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "req_per_min.png")
    plt.close()

    # (b) regression fit + highlighted intervals
    plt.figure()
    plt.plot(counts.index, counts["requests"], label="actual")
    plt.plot(counts.index, counts["predicted"], label="predicted")
    for a, b in strong_intervals:
        plt.axvspan(a, b, alpha=0.2)
    plt.title("Regression baseline and detected DDoS intervals")
    plt.xlabel("Time")
    plt.ylabel(f"Requests / {AGG_FREQ}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regression_fit.png")
    plt.close()

    # (c) z-score residuals
    plt.figure()
    plt.plot(counts.index, counts["z"])
    plt.axhline(Z_THRESHOLD)
    plt.title("Residual z-score (attack if above threshold)")
    plt.xlabel("Time")
    plt.ylabel("z-score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "residuals.png")
    plt.close()

    # 7) Print results for ddos.md
    print("\nDetected DDoS time interval(s):")
    if not strong_intervals:
        print("  None detected with current thresholds.")
        print(f"  Tip: try Z_THRESHOLD=2.5 or AGG_FREQ='30s' if the log is very spiky.")
    else:
        for a, b in strong_intervals:
            print(f"  {a}  ->  {b}")

    # Save table for convenience
    counts.to_csv("ddos_regression_output.csv", index=True)
    print("\nSaved: images/*.png and ddos_regression_output.csv")


if __name__ == "__main__":
    main()


```

---

### Step 4 — Output

The script will:

- Print detected intervals
- Generate visualizations in `/images`
- Save regression output CSV

---

## Conclusion

Regression-based anomaly detection successfully identified a short DDoS burst. By modeling expected traffic and flagging deviations, it is possible to detect abnormal activity in server logs.

This method is reproducible, scalable, and applicable to real-world intrusion detection scenarios.
