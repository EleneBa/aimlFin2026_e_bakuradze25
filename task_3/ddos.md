# Task 3 — Web Server Log Analysis (DDoS Detection using Regression)

## Objective

The goal of this task is to analyze a web server log file and identify time interval(s) corresponding to a Distributed Denial-of-Service (DDoS) attack. The detection is performed using statistical analysis and regression-based anomaly detection.

---

## Input Log File

- Original source:
  http://max.ge/aiml_final/e_bakuradze25_23456_server.log

- Local copy (uploaded in this folder):
  `e_bakuradze25_23456_server.log`

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

![Requests](images/req_per_min.png)

---

### Regression baseline vs actual traffic

Highlighted regions indicate detected attack intervals.

![Regression](images/regression_fit.png)

---

### Residual z-score

Values above threshold indicate anomalies.

![Residuals](images/residuals.png)

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
