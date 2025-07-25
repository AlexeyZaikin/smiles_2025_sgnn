from collections import defaultdict
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.data import Data
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")


def main():
    datasets = set()
    for p in Path("data/tabular/").glob("*.csv"):
        datasets.add(str(p).split("/")[-1].split(".")[0])

    all_data = {}
    for dataset in tqdm(datasets):
        all_data[dataset] = defaultdict(list)
        graph_data = pd.read_csv(f"data/tabular/{dataset}.graph.csv")
        node_features_data = pd.read_csv(
            f"data/tabular/{dataset}.node_features.csv", index_col=0
        )
        n_graphs = int(graph_data.columns[-1]) + 1
        # train_indices, _ = train_test_split(
        #     np.arange(n_graphs),
        #     train_size=0.9,
        #     stratify=node_features_data["target"].to_numpy(),
        # )
        for k in range(n_graphs):
            x = []
            y = []
            edge_index = []
            edge_attr = []
            for r in range(len(graph_data) - 1):
                i = int(graph_data["p1"].iloc[r].split("_")[-1])
                j = int(graph_data["p2"].iloc[r].split("_")[-1])
                v = graph_data[str(k)].iloc[r]
                edge_index.append((i, j))
                edge_attr.append((v,))
            x.append(node_features_data.iloc[k].to_numpy()[:-1])
            y = bool(node_features_data.iloc[k].to_numpy()[-1])
            is_test = bool(graph_data.iloc[len(graph_data) - 1][k])
            data = Data(
                x=torch.Tensor(x).T,
                edge_index=torch.LongTensor(edge_index).T,
                edge_attr=torch.Tensor(edge_attr),
                y=y,
            )
            if is_test:
                all_data[dataset]["test"].append(data)
            else:
                all_data[dataset]["train"].append(data)

    with open("data/tabular/processed_graphs.pkl", "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    main()
