import torch
import networkx as nx
from tqdm import tqdm


def add_node_features(graphs):
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []
        print("Adding node features for", data_type)
        for graph in tqdm(graphs[data_type]):
            if torch.isnan(graph.x).any():
                # shouldn't happen tho...
                raise ValueError(
                    "\n\n\nBEFORE NaN detected in node features for a graph."
                )

            graph = graph.clone()
            edge_index = graph.edge_index.cpu().numpy()
            edge_weights = graph.edge_attr.squeeze().cpu().numpy()
            num_nodes = graph.num_nodes

            G = nx.Graph()
            for (i, j), w in zip(edge_index.T, edge_weights):
                if w != 0:
                    G.add_edge(int(i), int(j), weight=float(w))
            G.add_nodes_from(range(num_nodes))

            deg = dict(G.degree())
            strength = dict(G.degree(weight="weight"))
            closeness = nx.closeness_centrality(G)
            betweenness = nx.betweenness_centrality(G, weight="weight")
            # ломается на 5ом графе
            # pagerank = nx.pagerank(G, weight="weight", max_iter=1000000)

            scalar_features = (
                graph.x.squeeze()
                if graph.x.dim() == 2 and graph.x.size(1) == 1
                else torch.zeros(num_nodes)
            )

            node_features = []
            for node in range(num_nodes):
                scalar = scalar_features[node].item()
                features = [
                    scalar,
                    deg.get(node, 0) / max(num_nodes - 1, 1),
                    strength.get(node, 0.0) / max(num_nodes - 1, 1),
                    closeness.get(node, 0.0),
                    betweenness.get(node, 0.0),
                    # pagerank.get(node, 0.0),
                ]
                node_features.append(features)

            graph.x = torch.tensor(node_features, dtype=torch.float)
            new_graphs.append(graph)

            if torch.isnan(graph.x).any():
                graph.x = torch.nan_to_num(graph.x) # TODO: proper fix it lol
                # raise ValueError(
                #     "\n\n\AFTER NaN detected in node features for a graph."
                # )

        graph_data[data_type] = new_graphs

    return graph_data
