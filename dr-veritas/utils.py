"""
A large portion of the code within this file is adapted from TxGNN's repository.
https://github.com/mims-harvard/TxGNN/tree/main
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from huggingface_hub import login, HfApi, hf_hub_download

# from google.colab import userdata
# TOKEN = userdata.get("<colab secret key>")
# login(token=Token)

REPO_ID = "dr-veritas/dr-dataset"
API = HfApi()
SEED = 42


def data_download(path_in_repo, save_dir, repo_id=REPO_ID):
    """Download file from HF dataset repo to local filespace."""
    local_path = os.path.join(save_dir, path_in_repo)

    if os.path.exists(local_path):
        print(f"Local copy found at '{local_path}'")
        return local_path

    url = f"{repo_id}/{path_in_repo}"
    print(f"Downloading file from '{url}'")
    os.makedirs(save_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        filename=path_in_repo,
    )


def data_upload(
    file_path, repo_id=REPO_ID, path_in_repo=None, commit_message=None, api=API
):
    """Upload file from local filespace to HF dataset repo."""

    if path_in_repo is None:
        path_in_repo = file_path

    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        commit_message=commit_message or f"Add {path_in_repo}",
    )


def convert2str(x):
    """
    TODO:
        FIX THIS, I THINK ITS A LITTLE TOO MESSY AND UNECESSARY
    """

    if pd.isna(x):
        return ""
    if isinstance(x, str):
        s = x.strip().strip('"')
        try:
            f = float(s)
            return str(int(f)) if f.is_integer() else s
        except Exception:
            return s
    try:
        f = float(x)
        return str(int(f)) if f.is_integer() else str(x)
    except Exception:
        return str(x)


def normalize_string(text):
    text = str(text).strip().lower()
    text = text.replace("/", "_").replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text


def build_directed_kg(
    kg_csv_path: str, out_dir: str, save_nodes: bool = True, add_edge_id: bool = True
):
    """Adapted from TxGNN: https://github.com/mims-harvard/TxGNN/blob/main/txgnn/utils.py"""

    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(kg_csv_path, low_memory=False)

    # nodes_path = None
    if save_nodes:
        nx = raw[["x_type", "x_id", "x_name", "x_source"]].rename(
            columns={
                "x_type": "node_type",
                "x_id": "node_id",
                "x_name": "node_name",
                "x_source": "node_source",
            }
        )
        ny = raw[["y_type", "y_id", "y_name", "y_source"]].rename(
            columns={
                "y_type": "node_type",
                "y_id": "node_id",
                "y_name": "node_name",
                "y_source": "node_source",
            }
        )
        nodes = pd.concat([nx, ny], ignore_index=True)
        nodes["node_id"] = nodes["node_id"].map(convert2str)
        nodes["node_type_raw"] = nodes["node_type"].astype(str)
        nodes["node_type"] = nodes["node_type_raw"].map(normalize_string)

        nodes = nodes.drop_duplicates(
            ["node_type", "node_id"], keep="first"
        ).reset_index(drop=True)
    else:
        nodes = None

    kg_df = raw[["x_type", "x_id", "relation", "y_type", "y_id"]].copy()

    kg_df["x_id"] = kg_df["x_id"].map(convert2str)
    kg_df["y_id"] = kg_df["y_id"].map(convert2str)

    kg_df["x_type_raw"] = kg_df["x_type"].astype(str)
    kg_df["x_type"] = kg_df["x_type"].map(normalize_string)
    kg_df["y_type_raw"] = kg_df["y_type"].astype(str)
    kg_df["y_type"] = kg_df["y_type"].map(normalize_string)

    kg_df["relation_raw"] = kg_df["relation"].astype(str)
    kg_df["relation"] = kg_df["relation_raw"].map(normalize_string)

    unique_relations = kg_df["relation"].unique()
    assert unique_relations.size == 30

    keep_idxs = []

    # deduplicate edges
    print("Deduplicating edges per relation...")
    for r in tqdm(unique_relations):
        temp = kg_df[kg_df.relation == r].copy()
        is_hom = (temp["x_type"] == temp["y_type"]).all()

        if is_hom:
            x = temp["x_id"].astype(str).values
            y = temp["y_id"].astype(str).values
            a = np.minimum(x, y)
            b = np.maximum(x, y)
            temp["check_key"] = temp["x_type"].astype(str) + "_" + a + "_" + b
            keep_idxs.extend(temp.drop_duplicates("check_key").index.tolist())
        else:
            canonical_x_type = temp["x_type"].iloc[0]
            keep_idxs.extend(temp[temp["x_type"] == canonical_x_type].index.tolist())

    kg_df = kg_df.loc[keep_idxs].copy().reset_index(drop=True)

    # build canonical key per edge based on (x_type, x_id, relation, y_type, y_id)
    if add_edge_id:
        is_hom = kg_df["x_type"] == kg_df["y_type"]
        x_id_s = kg_df["x_id"].astype(str)
        y_id_s = kg_df["y_id"].astype(str)
        swap = is_hom & (x_id_s > y_id_s)

        x_type_c = np.where(swap, kg_df["y_type"], kg_df["x_type"])
        x_id_c = np.where(swap, kg_df["y_id"], kg_df["x_id"])
        y_type_c = np.where(swap, kg_df["x_type"], kg_df["y_type"])
        y_id_c = np.where(swap, kg_df["x_id"], kg_df["y_id"])

        edge_key = (
            pd.Series(x_type_c).astype(str)
            + "_"
            + pd.Series(x_id_c).astype(str)
            + "_"
            + kg_df["relation"].astype(str)
            + "_"
            + pd.Series(y_type_c).astype(str)
            + "_"
            + pd.Series(y_id_c).astype(str)
        )

        kg_df["edge_id"] = pd.util.hash_pandas_object(edge_key, index=False).astype(
            "uint64"
        )

    kg_df["x_idx"] = -1
    kg_df["y_idx"] = -1

    print("Indexing nodes per type...")
    node_idx_map = {}

    node_types = np.unique(
        np.concatenate([kg_df["x_type"].unique(), kg_df["y_type"].unique()])
    )

    for ntype in tqdm(node_types):
        x_ids = kg_df.loc[kg_df.x_type == ntype, "x_id"].values
        y_ids = kg_df.loc[kg_df.y_type == ntype, "y_id"].values
        nids = np.unique(np.concatenate([x_ids, y_ids]).astype(str))

        nid2idx = dict(zip(nids, range(len(nids))))
        node_idx_map[ntype] = nid2idx

        kg_df.loc[kg_df.x_type == ntype, "x_idx"] = (
            kg_df.loc[kg_df.x_type == ntype, "x_id"]
            .astype(str)
            .map(nid2idx)
            .astype(int)
        )
        kg_df.loc[kg_df.y_type == ntype, "y_idx"] = (
            kg_df.loc[kg_df.y_type == ntype, "y_id"]
            .astype(str)
            .map(nid2idx)
            .astype(int)
        )

        if save_nodes:
            mask = nodes["node_type"] == ntype
            nodes.loc[mask, "node_idx"] = (
                nodes.loc[mask, "node_id"].astype(str).map(nid2idx)
            )

    edges_out = kg_df[
        [
            "x_type",
            "x_id",
            "x_idx",
            "relation",
            "y_type",
            "y_id",
            "y_idx",
            "relation_raw",
        ]
        + (["edge_id"] if add_edge_id else [])
    ].copy()

    kgd_path = os.path.join(out_dir, "kg_directed.parquet")
    edges_out.to_parquet(kgd_path, index=False)

    if save_nodes:
        nodes["node_idx"] = nodes["node_idx"].astype("Int64")
        nodes_path = os.path.join(out_dir, "nodes.parquet")
        nodes.to_parquet(nodes_path, index=False)

    return kgd_path, nodes_path


def add_reverse_edges(df_edges):
    df = df_edges.copy()
    df["is_reverse"] = 0

    rev = df.copy()
    rev = rev.rename(
        columns={
            "x_type": "y_type",
            "x_id": "y_id",
            "x_idx": "y_idx",
            "y_type": "x_type",
            "y_id": "x_id",
            "y_idx": "x_idx",
        }
    )

    hetero = rev["x_type"] != rev["y_type"]
    rev.loc[hetero, "relation"] = "rev_" + rev.loc[hetero, "relation"].astype(str)
    rev.loc[hetero, "relation_raw"] = "rev_" + rev.loc[hetero, "relation_raw"].astype(str)

    rev["is_reverse"] = 1
    out = pd.concat([df, rev], ignore_index=True)
    return out.reset_index(drop=True)


def random_edge_split(df_edges, frac=(0.8, 0.1, 0.1), seed=SEED):
    train_frac, val_frac, test_frac = frac
    rng = np.random.RandomState(seed)

    trains, vals, tests = [], [], []
    for rel in df_edges.relation.unique():
        temp = df_edges[df_edges.relation == rel]
        temp = temp.sample(frac=1.0, random_state=rng).reset_index(drop=True)

        n = len(temp)
        n_test = int(test_frac * n)
        n_val = int(val_frac * n)

        test = temp.iloc[:n_test]
        val = temp.iloc[n_test : n_test + n_val]
        train = temp.iloc[n_test + n_val :]

        trains.append(train)
        vals.append(val)
        tests.append(test)

    df_train = pd.concat(trains, ignore_index=True)
    df_val = pd.concat(vals, ignore_index=True)
    df_test = pd.concat(tests, ignore_index=True)
    return df_train, df_val, df_test


def dd_disease_holdout_split(
    df_edges, holdout_disease_idx, val_frac_of_train_dd=0.05, seed=SEED
):
    dd_rel_types_raw = ["indication", "contraindication", "off-label use"]
    rng = np.random.RandomState(seed)

    dd_rel_norm = [normalize_string(r) for r in dd_rel_types_raw]

    df_dd = df_edges[df_edges.relation.isin(dd_rel_norm)]
    df_not_dd = df_edges[~df_edges.relation.isin(dd_rel_norm)]

    df_dd_test = df_dd[df_dd.y_idx.isin(holdout_disease_idx)]
    df_dd_trainval = df_dd[~df_dd.y_idx.isin(holdout_disease_idx)]

    df_dd_val = df_dd_trainval.sample(frac=val_frac_of_train_dd, random_state=rng)
    df_dd_train = df_dd_trainval.drop(index=df_dd_val.index)

    df_train = pd.concat([df_not_dd, df_dd_train], ignore_index=True)
    df_val = df_dd_val.reset_index(drop=True)
    df_test = df_dd_test.reset_index(drop=True)

    return df_train.reset_index(drop=True), df_val, df_test


def full_graph_disease_holdout_split(
    df_edges, frac=(0.8, 0.1, 0.1), seed=SEED, split_non_dd="train"
):
    rng = np.random.RandomState(seed)

    dd_rel_types_raw = ["indication", "contraindication", "off-label use"]
    dd_rel_norm = [normalize_string(r) for r in dd_rel_types_raw]

    df_dd = df_edges[df_edges.relation.isin(dd_rel_norm)]
    df_not_dd = df_edges[~df_edges.relation.isin(dd_rel_norm)]

    diseases = df_dd["y_idx"].unique()
    rng.shuffle(diseases)

    n = len(diseases)
    n_train = int(frac[0] * n)
    n_val = int(frac[1] * n)

    train_dis = diseases[:n_train]
    val_dis = diseases[n_train : n_train + n_val]
    test_dis = diseases[n_train + n_val :]

    df_dd_train = df_dd[df_dd.y_idx.isin(train_dis)]
    df_dd_val = df_dd[df_dd.y_idx.isin(val_dis)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test_dis)]

    if split_non_dd == "train":
        df_train = pd.concat([df_not_dd, df_dd_train], ignore_index=True)
        df_val = df_dd_val.copy()
        df_test = df_dd_test.copy()
    else:
        nnd_train, nnd_val, nnd_test = standard_disease_fold(
            df_not_dd, frac=frac, seed=seed
        )
        df_train = pd.concat([nnd_train, df_dd_train], ignore_index=True)
        df_val = pd.concat([nnd_val, df_dd_val], ignore_index=True)
        df_test = pd.concat([nnd_test, df_dd_test], ignore_index=True)

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def create_split(
    df_edges, split_dir, frac=(0.8, 0.1, 0.1), method="standard", seed=SEED
):
    dd_rel_types_raw = ["indication", "contraindication", "off-label use"]
    dd_edges = df_edges[
        df_edges.relation.isin([normalize_string(r) for r in dd_rel_types_raw])
    ]
    all_diseases = dd_edges.y_idx.unique()
    np.random.seed(seed)
    holdout = np.random.choice(all_diseases, size=100, replace=False)

    match method:
        case "standard":
            train_df, val_df, test_df = dd_disease_holdout_split(df_edges, holdout)
        case _:
            train_df, val_df, test_df = dd_disease_holdout_split(df_edges, holdout)

    train_df = add_reverse_edges(train_df)
    val_df = add_reverse_edges(val_df)
    test_df = add_reverse_edges(test_df)

    train_df.to_parquet(os.path.join(split_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(split_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(split_dir, "test.parquet"), index=False)

    return train_df, val_df, test_df


# REMOVE LATER
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, required=True, help="Path to PrimeKG kg.csv")
    parser.add_argument("--out", type=str, default="./data/processed")
    parser.add_argument("--no_nodes", action="store_true")
    parser.add_argument("--no_edge_id", action="store_true")
    args = parser.parse_args()

    kgd_path, nodes_path = build_directed_kg(
        kg_csv_path=args.kg,
        out_dir=args.out,
        save_nodes=not args.no_nodes,
        add_edge_id=not args.no_edge_id
    )

    print("Wrote:", kgd_path)
    df_e = pd.read_parquet(kgd_path)

    print(df_e.head())
    print("Edges:", len(df_e))
    print("Relations:", df_e["relation"].nunique(), "sample:", df_e["relation"].unique()[:10])
    # sanity checks
    assert (df_e["x_idx"] >= 0).all(), "Some x_idx are -1"
    assert (df_e["y_idx"] >= 0).all(), "Some y_idx are -1"
    assert df_e["relation"].nunique() == 30, "Expected 30 relations after deduplication"
    if "edge_id" in df_e.columns:
        assert df_e["edge_id"].isna().sum() == 0, "Missing edge_id values"

    if nodes_path:
        df_n = pd.read_parquet(nodes_path)
        print("Nodes:", len(df_n), "Types:", df_n["node_type"].nunique())
        assert df_n["node_idx"].isna().sum() == 0, "Missing node_idx in nodes table"
        print(df_n.sample(5))
