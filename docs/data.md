# PrimeKG (Precision Medicine Knowledge Graph)
We use the **Precision Medicine Knowledge Graph (PrimeKG)** [\[Github\]](https://github.com/mims-harvard/PrimeKG) [\[Harvard Dataverse\]](https://doi.org/10.7910/DVN/IXA7BM) released by the MIMS-Harvard group.

PrimeKG integrates 20 curated biomedical resources into a disease-centric, heterogeneous biomedical
knowledge graph with ~129k nodes (across 10 types) and over 4 million edges (across 29 relation types).
Textual descriptors are provided for disease and drug nodes.

For the official schema and build pipeline, see the PrimeKG Github and Dataverse pages.

---

## Contents of this dataset moirror

The raw dataset and our processed files are located on a private Hugging Face dataset repository.

```bash
data/
    kg.csv                                # PrimeKG as released
    disease_features.tab                  # disease text features as released by PrimeKG
    drug_features.tab                     # drug text features as released by PrimeKG

processed/
    kg_directed.parquet                   # our processed directed KG
    drug_disease_edges.parquet            # all drug-disease edges
    
    mappings/
        node_table.parquet                # idx mappings for all nodes
        node_type_table.parquet           # idx mappings for node types
        relation_table.parquet            # idx mappings for edge types
    
    features/
        {drug, disease}_text.parquet      # drug and disease "label" and "descriptive" text
        {drug, disease}_sapbert_emb.pt    # SapBERT label text embeddings
        {drug, disease}_medembed_emb.pt   # MedEmbed descriptive text embeddings
        {drug, disease}_text_init_emb.pt  # concattenated label and descriptive text embeddings
    
    splits/
        dd_edges_zero_shot_{test/val/train}.parquet   # zero-shot train/val/test splits for drug disease rels
        dd_edges_random_splits.parquet                # random train/val/test splits for drug disease rels
        disease_zero_shot_splits.parquet              # train/val/test disease splits

```

---

## Quick start (Colab)
```python
!pip -q install -U huggingface_hub

from huggingface_hub import login, list_repo_files, hf_hub_download

from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))

REPO_ID = # our repo id

# list files
print(*list_repo_files(REPO_ID, repo_type="dataset"), sep="\n")

# download a file and read its head
kgd = hf_hub_download(
    REPO_ID, repo_type="dataset",
    filename="processed/kg_directed.parquet"
)
import pandas as pd
print(pd.read_parquet(kgd, nrows=5))
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
