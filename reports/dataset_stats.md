# MT-Bench human judgments: dataset statistics

| Quantity | Value |
|---|---|
| Human votes | 3355 |
| Human judges | 65 |
| Questions | 80 |
| Keys (question, turn, model pair) with human votes | 1814 |
| Mean human votes per key | 1.85 |
| Keys with 2+ human votes | 961 |
| GPT-4 votes | 2400 |
| GPT-4 votes inconsistent under A/B swap | 380 |
| Human keys that also have a GPT-4 vote | 1814 |

Models: alpaca-13b, claude-v1, gpt-3.5-turbo, gpt-4, llama-13b, vicuna-13b-v1.2.

Outcome counts (first / second / tie, canonical model order):

- human: 1342 / 1233 / 780
- GPT-4: 929 / 871 / 600 (inconsistent verdicts counted as ties)
