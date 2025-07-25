import networkx as nx
import numpy as np
import torch
from functools import partial


def no_sparsify(graphs):
    graph_data = {}

    for split, graph_list in graphs.items():  # split \in {"train", "test"}
        new_graphs = []
        for g in graph_list:
            g = g.clone()
            # g.edge_attr = (g.edge_attr.squeeze() - 0.5).unsqueeze(1)
            new_graphs.append(g)
        graph_data[split] = new_graphs

    return graph_data


def sparsify_p(graphs, p):
    # p -- доля ребер, которые надо оставить
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []

        for graph in graphs[data_type]:
            graph = graph.clone()
            edge_attr = graph.edge_attr.squeeze()
            # edge_attr = edge_attr - 0.5
            num_edges_to_keep = int(np.ceil(len(edge_attr) * p))
            abs_weights = edge_attr.abs()
            threshold = torch.topk(
                abs_weights, num_edges_to_keep, largest=True
            ).values.min()
            edge_attr[abs_weights < threshold] = 0.0
            graph.edge_attr = edge_attr.unsqueeze(1)
            new_graphs.append(graph)

        graph_data[data_type] = new_graphs

    return graph_data


def sparsify_knn(graphs, p):
    # p -- доля соседей, ребра к которым оставляем
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []

        for graph in graphs[data_type]:
            graph = graph.clone()
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr.squeeze()
            # edge_attr = edge_attr - 0.5
            num_nodes = graph.x.shape[0]
            k = int((num_nodes - 1) * p)
            mask = torch.zeros(edge_attr.size(0), dtype=torch.bool)

            for node in range(num_nodes):
                idx = ((edge_index[0] == node) | (edge_index[1] == node)).nonzero(
                    as_tuple=True
                )[0]
                if len(idx) > k:
                    topk = edge_attr[idx].abs().topk(k).indices
                    mask[idx[topk]] = True
                else:
                    mask[idx] = True  # если рёбер меньше или равно k — оставить все

            edge_attr[~mask] = 1e-5
            graph.edge_attr = edge_attr.unsqueeze(1)
            new_graphs.append(graph)

        graph_data[data_type] = new_graphs

    return graph_data


def sparsify_min_connected(graphs):
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []

        for graph in graphs[data_type]:
            graph = graph.clone()
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr.squeeze()
            # edge_attr = edge_attr - 0.5
            abs_weights = edge_attr.abs()
            unique_weights = torch.unique(abs_weights)
            unique_weights, _ = torch.sort(unique_weights)
            low = 0
            high = len(unique_weights) - 1
            best_eps = unique_weights[-1]

            # бинпоиск по уникальным весам
            while low <= high:
                mid = (low + high) // 2
                eps = unique_weights[mid]

                # Строим граф из рёбер с abs(w) >= eps
                mask = abs_weights >= eps
                edges = edge_index[:, mask].T.cpu().numpy()
                G = nx.Graph()
                G.add_edges_from(edges)
                G.add_nodes_from(range(graph.num_nodes))

                if nx.is_connected(G):
                    best_eps = eps
                    low = mid + 1
                else:
                    high = mid - 1

            # Обнуляем слабые рёбра
            edge_attr[abs_weights < best_eps] = 0.0
            graph.edge_attr = edge_attr.unsqueeze(1)
            new_graphs.append(graph)

        graph_data[data_type] = new_graphs

    return graph_data


def get_sparsify_f_list(p_list=[0.3, 0.5, 0.8]):
    f_list = (
        [("no_sparsify", no_sparsify)]
        + [
            (f"sparsify_p_{p_val}".replace(".", "_"), partial(sparsify_p, p=p_val))
            for p_val in p_list
        ]
        + [
            (f"sparsify_knn_{p_val}".replace(".", "_"), partial(sparsify_knn, p=p_val))
            for p_val in p_list
        ]
        + [("sparsify_min_connected", sparsify_min_connected)]
    )
    return f_list
