# Dataset sources and preparation

The repository distributes reproducibility code and documentation; experimental
results are provided in the paper. Download data
directly from the providers; preserve their filenames and directory structure.
Provider links were checked on 24 November 2026. The landing pages are available;
the downstream DataPort download was not accessible to the checking tool and the
UNSW SharePoint link redirected to Microsoft sign-in. Access may require an account.

## 5G-NIDD

- Official source: https://netslab.ucd-cs.ie/netslab-datasets/5g-nidd-dataset/
- Provider-linked download: https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless
- Dataset paper: https://arxiv.org/abs/2212.01298

Obtain the **GTP-removed PCAPNG captures for BS1 and BS2**, together with the
corresponding **per-attack Argus CSVs**. A single encoded/combined CSV is not
sufficient for reproducing packet content, source rotation and window experiments.

```text
$FIVEG_RAW_ROOT/
  pcap/BS1_GTP_removed/*_nogtp.pcapng
  pcap/BS2_GTP_removed/*_nogtp.pcapng
  csv/BS1_each_attack/*.csv
  csv/BS2_each_attack/*.csv
```

The offline timing script specifically reads
`pcap/BS2_GTP_removed/Goldeneye_BS2_nogtp.pcapng`.
Do not rename captures: capture names identify sites and withheld tools.

After setting paths as in REPRODUCE.md, execute **both** steps:

```bash
python scripts/legacy_preparation/T2_0b_relabel_5gnidd.py run
python scripts/legacy_preparation/T2_0c_fix_labels.py
```

The first writes structured NumPy packet shards to
`$IOT_DATA_ROOT/features/payload_v2_4class/5G-NIDD/`. The second corrects labels
in those generated shards in place; it is essential because some Argus CSVs
encode malicious rows with attack names rather than the word `Malicious`.
The correction joins by capture, protocol and unordered endpoint pair, without
port or timestamp alignment; any non-Benign matched row makes that key malicious.
Unmatched packets remain benign. This coarse inherited label rule reproduces
the paper's protocol; it does not establish precise packet-level ground truth.
The first-stage docstring describes its initial join; the correction is authoritative.

Packet shards store 256 payload bytes; TCV uses the first 64, zero-padded where
necessary. The original per-packet timing field is not trusted by the corrected
pipeline: timing is recalculated inside each rebuilt edge.

## TON-IoT

- Official source and download entry: https://research.unsw.edu.au/projects/toniot-datasets
- Select **Train_Test_datasets**, network data, **train_test_network.csv**.
- Network dataset paper: N. Moustafa, “A new distributed architecture for
  evaluating AI-based security systems at the edge: Network TON_IoT datasets,”
  Sustainable Cities and Society, 2021, article 102994 (linked by UNSW).

Place the unmodified file at:

```text
$IOT_DATA_ROOT/data/v5_raw/TON_IoT/train_test_network.csv
```

Use network flows, not IoT telemetry or Windows/Linux audit files. The released
CSV supplies flow summaries, not packet payload or absolute timestamps. Its
content proxy has nine fields, and timing uses duration divided by packet count.
Do not synthesize timestamps or reuse the packet window experiment for these flows.

## Dataset permissions and provenance

Read the providers' terms and citation requirements before use. UNSW explicitly
describes academic-use permission and a separate request for commercial use;
it also lists required dataset citations. Our code license does not license
either dataset. Keep dataset citations in addition to the TCV paper citation.

Before running, save a manifest (hashing full PCAPs can take time):

```bash
python scripts/input_manifest.py "$IOT_DATA_ROOT" --output "$IOT_REVISION_ROOT/prepared_inputs.json"
python scripts/input_manifest.py "$FIVEG_RAW_ROOT" --output "$IOT_REVISION_ROOT/raw_inputs.json"
```

Manifests use relative filenames and SHA-256. Preserve these alongside the
environment, code commit, split indices and prediction files. Input differences
can change edge counts, splits and scores even when seeds are identical.
