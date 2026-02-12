# Task 2 — Transformer Networks in Cybersecurity

## What is a Transformer Network?

A transformer network is a deep learning architecture designed for processing sequential data. Unlike traditional recurrent neural networks (RNNs), transformers do not process data step-by-step. Instead, they use an attention mechanism that allows the model to look at all parts of the input sequence at the same time. This makes transformers highly parallelizable and efficient for large datasets.

The core idea behind transformers is the **self-attention mechanism**. Self-attention allows each element in a sequence to evaluate its relationship with every other element. This helps the model understand context and dependencies, even when elements are far apart in the sequence.

Transformers also use **positional encoding**. Since transformers do not process data sequentially, they need positional information to understand order. Positional encoding injects information about token position using mathematical functions, typically sine and cosine waves.

## Why Transformers Are Powerful

Transformers excel at capturing long-range dependencies. They can model relationships across an entire sequence without the memory limitations of RNNs. They also scale well and are the foundation of modern AI systems such as large language models.

## Applications in Cybersecurity

In cybersecurity, transformers are valuable for:

- **Intrusion detection**: analyzing sequences of network events  
- **Malware detection**: modeling behavior patterns over time  
- **Anomaly detection**: spotting unusual user or system activity  
- **Threat intelligence analysis**: processing large volumes of security logs  
- **Phishing detection**: analyzing email text patterns

Transformers can learn complex relationships in logs, network flows, and user behavior, making them suitable for advanced threat detection systems.

---

## Attention Mechanism Visualization

[![Attention](images/attention.png)](https://github.com/EleneBa/aimlFin2026_e_bakuradze25/blob/main/task_2/attention.png) 

The heatmap shows how each token (query) attends to others (keys). Brighter values indicate stronger attention.

---

## Positional Encoding Visualization

[![Positional Encoding](images/positional_encoding.png)](https://github.com/EleneBa/aimlFin2026_e_bakuradze25/blob/main/task_2/positional_encoding.png)

The sinusoidal pattern represents how positional information is encoded so the model understands order.

---

## Reproducibility

To regenerate visuals:

```bash
pip install matplotlib numpy
python make_transformer_images.py
```
