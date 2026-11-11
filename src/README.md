# Poisoning Attacks to Local Differential Privacy Protocols for Dynamic Graphs

This code studies data-poisoning attacks against LDP protocols for dynamic graphs. It implements an absolute gain metric and a unified framework for target selection, fake-node scheduling, and fake-edge injection, evaluating TEA against ESA and RSA on real and synthetic data. Current experiments measure graph metrics such as degree centrality (DC) and local clustering coefficient (LCC). Attack nodes are selected from existing non-target nodes.

## Installation and Tests

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows, or `source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

## Datasets

Raw datasets are excluded. Download them from their sources, follow their citation and usage terms, and use these paths:

| Dataset | Source | Path and format |
| --- | --- | --- |
| AS-733 | [SNAP](https://snap.stanford.edu/data/as-733.html) | `dataset/as-733/`; one `.txt` edge list per time step, with two integer node IDs per edge. |
| Email-Eu-core temporal | [SNAP page](https://snap.stanford.edu/data/email-Eu-core-temporal.html), [download](https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz) | `dataset/email/email.txt`; `SRC DST UNIXTS`. The local file has 332,334 events. The reader makes undirected graphs in 20 windows and returns the first 12. |
| Linux kernel reply | [Network Data Repository](https://networkrepository.com/comm-linux-kernel-reply.php) | `dataset/reply/reply.txt`; source, target, edge attribute, timestamp. The reader ignores the edge attribute and uses 20 windows. |
| Synthetic | Generated data; generator not included | `dataset/Synthetic/`; one edge-list `.txt` file per time step. |

The email data represents anonymized communication within a European research institution (986 nodes, 332,334 temporal edges, 803 days); cite Paranjape, Benson, and Leskovec, *Motifs in Temporal Networks*, WSDM 2017. Synthetic data cannot be reproduced without its generator and parameters. Keep `dataset/as-733/` present because `main.py` reads it at import. Files are sorted lexicographically; zero-pad numeric time-step names.

## Running Experiments

Run from the repository root. Select the same dataset throughout; the attack scripts also prompt for an allocation strategy.

1. Reconstruct graphs and compute average-degree limits:

   ```bash
   python run_hmg_pipeline.py
   python calculate_hmg_average_degrees.py
   ```

2. Generate DC and LCC attacks:

   ```bash
   python attackcentre.py
   python attacklcc.py
   ```

3. Process and merge attack graphs:

   ```bash
   python hmg_attack_processor.py
   python fake_node_hmg_merger.py
   ```

4. Calculate gains:

   ```bash
   python attack_gain_calculator.py
   python original_vs_attack_gain_calculator.py
   ```

The gain scripts write CSV files to `attack_gain_results/` and `original_vs_attack_gain_results/`.

To run the configured parameter sweep, use `python automated_pipeline.py`. It edits constants in five Python files and restores them from sibling `.backup` files. Existing backups are not refreshed; run from a clean copy and check for stale backups first.
