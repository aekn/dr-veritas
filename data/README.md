# PrimeKG (Precision Medicine Knowledge Graph)
We use the **Precision Medicine Knowledge Graph (PrimeKG)** [\[Github\]](https://github.com/mims-harvard/PrimeKG) [\[Harvard Dataverse\]](https://doi.org/10.7910/DVN/IXA7BM) released by the MIMS-Harvard group.

PrimeKG integrates 20 curated biomedical resources into a disease-centric, heterogeneous biomedical
knowledge graph with ~129k nodes (across 10 types) and over 4 million edges (across 29 relation types).
Textual descriptors are provided for disease and drug nodes.

For the official schema and build pipeline, see the PrimeKG Github and Dataverse pages.

For more information about our dataset mirror [see the docs](../docs/data.md).

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

