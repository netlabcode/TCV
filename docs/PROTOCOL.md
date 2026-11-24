# Protocol and code guide

## Records, edges and eligibility

`load_records(dataset)` normalizes packet shards or released network flows into
a DataFrame plus a content matrix. `record_id` indexes the original content
matrix, including when a subset of records is rebuilt. `aggregate` groups by
`capture, proto, src, dst, port` (destination port); source port is deliberately
not an edge-key field. At least **five records**, not five summarized packets,
are required. Majority labels use deterministic sorted-category tie breaking.
The corrected primary cohorts have 62,180 eligible 5G edges and 1,785 TON edges.

5G classes are Normal, Flooding, Scanning and Slow-rate. TON classes are Normal,
Flooding, Scanning and Exploit. TON DoS/DDoS map to Flooding; backdoor, injection,
password, XSS, ransomware and MITM map to Exploit. The fourth classes are not
equivalent across datasets. Each dataset is fitted and evaluated independently.

## Six features (column order is fixed)

`reference(aggregate, labeled_indices)` uses only annotated Normal edges. Its
content reference is record-weighted; timing uses positive within-edge intervals;
size/application distributions pool benign training records. No benign annotations
is an error. `features(aggregate, reference)` returns one row per edge:

| Column | Meaning and implemented definition |
|---|---|
| alpha | Maximum absolute difference between edge and reference mean content vectors |
| beta | Absolute positive-interval median deviation divided by training IQR, floored at 1e-6; zero when no positive interval exists |
| gamma | Sum of size and positive timing sample variances on paired positions; this is not covariance |
| delta | Mean absolute CDF gap over 32 size bins spanning 0–1500 bytes |
| epsilon | Jensen–Shannon divergence of application distributions, natural logarithms, union support |
| zeta | `log1p(mean record size)` |

5G content is the first 64 zero-padded payload bytes. TON content comprises
log-transformed HTTP request/response lengths, HTTP status/100, DNS query type,
DNS response code, SSL-established indicator, and log-transformed missed/source/
destination bytes. TON timing is duration divided by total packet count; its
mean record size is IP bytes divided by packet count, clipped to 0–1500.
5G lengths outside the histogram range are omitted only from that histogram,
not from other statistics. Empty histograms use the implemented protected
normalization. The six features therefore have dataset-specific input semantics.

## Splits, labels and references

`splits` fixes a stratified 20% random test set using seed 2026. Additional
splits hold out BS1/BS2 for 5G and complementary source groups A/B for TON.
TON's exact source allocation is explicit in `splits`; it retains four-class
support and is not a claim of random sampling of independent deployment sites.

`sample_labels` uses 1%, 3% and 5% of eligible training-pool edges (floor),
seeds 42, 123, 456, 789, 101112. A deterministic replacement guarantees at least
one annotation per class without increasing the exact budget. Representations
share labels and test membership. References use only annotated benign edges.

`run` evaluates TCV6, TCV3 (first three columns), adapted 16 flow statistics,
and hashed-identity features. TCV/flow-statistic comparisons use GradientBoosting
and 120-tree ExtraTrees; identity uses an 80-tree depth-10 RandomForest diagnostic.
It is not a learner-matched representation comparison. Ablations use GradientBoosting.

## Perturbation and additional comparisons

Source identity changes every 1,000/100/10 records per original source/capture.
Only test records are rotated; edge grouping, timing and eligibility are rebuilt.
Training and benign references stay fixed. Small fragments abstain: report both
retained-edge scores and record-level coverage/scores including missed records.

Native baselines use released numeric Zeek/Argus fields with endpoint and target
fields excluded. Argus uses a matched subset (6,403 edges), not the full primary
5G cohort. Withheld Goldeneye, Slowloris and TCPConnect captures preserve four
classes but leave few Flooding training examples; this is a difficult capture/tool
holdout, not unseen-class or cross-dataset transfer.

STEM uses RF/ET/LightGBM/XGBoost/HistGradientBoosting with logistic-regression
combination and standardized TCV passthrough. It refits benign references inside
every OOF training fold; final base fits use the outer annotated set. Up to three
stratified folds are limited by rare-class annotations. Report unavailable cases.

AfterImage uses at most 64 eligible edges per capture/class, the first five
records per edge, and 100 mean statistics. State resets for every capture and
train/test partition. Both representations use the same supervised ExtraTrees
protocol. It is a bounded cohort comparison, not a full KitNET evaluation.

Window experiments use absolute 1/10/60-second bins, BS1 training and BS2 test,
5% annotations. One-second windows do not retain four-class support. Delays
measure first malicious record to first correctly malicious window closure;
missed original attack edges are counted separately. These are not runtime latency.

The corpus-reference and five-packet TON settings are explicitly sensitivity
diagnostics. Corpus references include held-out information and are not the
leakage-aware primary protocol. Earlier historical diagnostics retained in the
paper are not regenerated by this focused release.

## Metrics and uncertainty

Macro-F1 uses the complete class list and zero_division=0. Five-seed means use
95% Student-t intervals with four degrees of freedom. Seeds vary annotation/model
sampling on a fixed split, not five independent data collections. Do not pool
the two complementary holdout directions as independent replicates. Runtime
depends on hardware and load; offline measurements do not establish edge-device
feasibility. Interpret results in the paper according to their stated cohort.
