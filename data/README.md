# PrimeKG (Precision Medicine Knowledge Graph)
We use the **Precise Medicine Knowledge Graph (PrimeKG)** [Github](https://github.com/mims-harvard/PrimeKG) [Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM) released by the MIMS-Harvard group.

PrimeKG integrates 20 curated biomedical resources into a disease-centric, heterogeneous biomedical
knowledge graph with ~129k nodes (across 10 types) and over 4 million edges (across 29 relation types).
Textual descriptors are provided for disease and drug nodes.

For the official schema and build pipeline, see the PrimeKG Github and Dataverse pages.

---

## Contents of this dataset mirror

```bash
kg.csv            # PrimeKG as released
kg_directed.csv   # our canonical directed view (similar to TxGNN)
edges.csv         # undirected index-based edges as released by PrimeKG
node.csv          # node table as released by PrimeKG

extra/HumanDO.obo                         # human disease ontology (from TxGNN)
extra/mondo_references.csv                # mondo cross-references (from TxGNN)
extra/kg_grouped_diseases_bert_map.tab    # grouped disease mapping as released by PrimeKG
extra/disease_features.tab                # disease text features as released by PrimeKG
extra/drug_features.tab                   # drug text features as released by PrimeKG

splits/
    complex_disease/
        seed_{42..46}/train.csv, valid.csv, test.csv, meta.json
    random/
        seed_{42.46}/train.csv, valid.csv, test.csv
```

**Schema of `kg_directed.csv` and split files**:
`relation, display_relation, x_id, x_type, y_id, y_type, x_idx, y_idx`
- `display_relation` preserves PrimeKG's display labels (which can be used as edge features).
- `x_idx` & `y_idx` are per-type contiguous indices (e.g., drugs: 0..N_drug-1).

---

## How `kg_directed.csv` is built
Our preprocessing is nearly identical to TxGNN's [GitHub](https://github.com/mims-harvard/TxGNN) with one small change, noted in step 4.

1. Normalize IDs: convert `x_id` and `y_id` to canonical strings.
2. Orient drug-disease edges: force drug->disease for `indication`, `contraindication`, and `off-label use`.
3. Canonical direction per relation:
    - Homogeneous (e.g., `protein_protein`): deduplicate by unordered pair (drop mirrored duplicates).
    - Heterogeneous (e.g., `drug_protein`): pick a canonical direction.
4. Drop exact duplicate triples on (`relation`, `x_id`, `y_id`). This is our only deviation from TxGNN.
5. Per-type reindex to get contiguous node indices `x_idx`, `y_idx`.

After these steps our `kg_directed.csv` has: 3,871,729 rows, 30 relation types, 10 node types, 0 exact duplicate triples,
and 42,631 drug->disease edges.


### What differs from TxGNN
TxGNN's script leaves a small set of true duplicate rows in the heterogeneous relation `drug_protein`
because it doesn't drop identical (`relation`, `x_id`, `y_id`) triples. PrimeKG coarsens fine-grained roles
(`target`, `enzyme`, `transporter`, `carrier`) under a single `drug_protein` relation, so the same 
(drug, protein) pair can appear twice with different `display_relation`.
- We found 185 unique duplicate triples (370 rows) in `drug_protein`.
- We remove exact duplicates to avoid doubel-counting.
- If you want to preserve role multiplicity as edge features, keep rows distinct by (`relation`, `x_id`, `y_id`, `display_relaion`)
instead of dropping exact duplicates.

We did not find duplication in any other relation types.

Everything else matches the TxGNN preprocessing logic.

---

## Splits
We provide five seeds: 42, 43, 44, 45, 46.

**A) `complex_disease` (TxGNN-style splits for zero-shot focus)**
- Randomly pick diseases until their drug->disease (DD) edges total ~5% of all DD edges.
- Test set = all DD edges for the selected diseases.
- Train/Val = all remaining edges, of all relations, with ~12% for validation.
- Each complex disease split folder includes a `meta.json` with counts.
- Typical sizes (seed 42): train=3,405,245, val=464,352, test=2,132.

**B) `random` (relation-agnostic splits)**
- These are uniform row-level splits over all edges: ~5% test, ~12% of the remainder for val, rest train. 
- The test set is larger for these splits as it covers all relations.
- Typical sizes (seed 42): train=3,236,766, val=441,377, test=193,586.

---

## Quick start (Colab)
```python
!pip -q install -U huggingface_hub

from huggingface_hub import login, list_repo_files, hf_hub_download
login() # paste read-only token interactively, DO NOT HARDCODE TOKEN

REPO_ID = "aekn/drveritas-primekg-splits"

# list files
print(*list_repo_files(REPO_ID, repo_type="dataset"), sep="\n")

# download a file and read its head
kgd = hf_hub_download(
    REPO_ID, repo_type="dataset",
    filename="kg_directed.csv"
)
import pandas as pd
print(pd.read_csv(kgd, nrows=5))
```

---

## Attribution & Citations

Please cite the original dataset and methods we build on:

PrimeKG (dataset): Chandak, P.; Huang, K.; Zitnik, M. Harvard Dataverse (2022).
DOI: [https://doi.org/10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM)

PrimeKG (paper): Chandak, P., Huang, K., Zitnik, M. “Building a knowledge graph to enable precision medicine.”
*Scientific Data (2023).* [https://doi.org/10.1038/s41597-023-01960-3](https://doi.org/10.1038/s41597-023-01960-3)

TxGNN (preprint): Huang, K. et al. “Zero-shot Prediction of Therapeutic Use with Geometric Deep Learning and Clinician Centered Design.”
*medRxiv (2023).* [https://doi.org/10.1101/2023.03.19.23287458](https://doi.org/10.1101/2023.03.19.23287458)

TxGNN (code): [https://github.com/mims-harvard/TxGNN](https://github.com/mims-harvard/TxGNN)
