import dgl
import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CATEGORICAL = [
    'proto_enum',
    'service_string',
    'conn_state_string',
]

COLUMNS = CATEGORICAL + [
    # 'id.resp_pport',
    'duration_interval',
    'missed_bytes_count',
    'bytes',
    'packet_count',
]


def categorical_variable(df: pd.DataFrame) -> pd.DataFrame:
    variables = list(df.unique())
    variables_map = dict(zip(variables, range(len(variables))))
    print(variables_map)
    df = df.apply(lambda x: variables_map[x])
    return df


def build_dataframe() -> pd.DataFrame:
    files = ["Only_Benign_7-1.csv", "Only_Benign_34-1.csv", "Only_Benign_60-1.csv",
             "Only_DDOS_7-1.csv", "Only_DDOS_34-1.csv"]
    df = [pd.read_csv(file) for file in files]
    df = pd.concat(df, ignore_index=True)
    df['bytes'] = df[["orig_ip_bytes_count", "resp_bytes"]].sum(axis=1)
    df['packet_count'] = df[["orig_pkts_count", "resp_pkts_count"]].sum(axis=1)

    # preprocess duration
    df['duration_interval'] = df['duration_interval'].replace("-", 0)
    df['duration_interval'] = df['duration_interval'].astype(float)
    df['Category'] = (df['Category'] == 'Malicious').astype(int)

    for category in CATEGORICAL:
        df[category] = categorical_variable(df[category])

    print(df.describe())
    print(df.head())
    print(df.columns)
    print(df.dtypes)
    return df


def build_networkx_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()

    for _, row in df.iterrows():
        g.add_node(row['id.orig_addr'])
        g.add_node(row['id.resp_haddr'])
        source = row['id.orig_addr']
        destination = row['id.resp_haddr']
        edge_attrs = {
            # 'id.orig_port': int(row['id.orig_port']),
            'id.resp_pport': int(row['id.resp_pport']),
            'proto_enum': int(row['proto_enum']),
            'conn_state_string': int(row['conn_state_string']),
            'service_string': int(row['service_string']),
            'duration_interval': float(row['duration_interval']),
            'missed_bytes_count': int(row['missed_bytes_count']),
            'bytes': int(row['bytes']),
            'packet_count': int(row['packet_count']),
            'Category': int(row['Category'])
        }
        g.add_edge(source, destination, **edge_attrs)
    return g


class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GNN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


def prepare_feats_labels(df):
    features = df[COLUMNS].values
    labels = df['Category'].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def train_model(model, graph, features, labels, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    for epoch in range(epochs):
        model.train()
        logits = model(graph, features)
        all_logits.append(logits)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def evaluate_model(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
    pred = torch.argmax(logits, dim=1).numpy()
    accuracy = accuracy_score(labels.numpy(), pred)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    df = build_dataframe()
    ben = sum(df['Category'] == 0)
    mal = sum(df['Category'] == 1)
    print(mal / (ben + mal))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])

    train_graph_data = build_networkx_graph(train_df)
    test_graph_data = build_networkx_graph(test_df)

    dgl_train_graph = dgl.from_networkx(train_graph_data, edge_attrs=COLUMNS + ["Category"])
    dgl_test_graph = dgl.from_networkx(test_graph_data, edge_attrs=COLUMNS + ["Category"])

    dgl_train_graph = dgl.add_self_loop(dgl_train_graph)
    dgl_test_graph = dgl.add_self_loop(dgl_test_graph)

    train_features, train_labels = prepare_feats_labels(train_df)
    test_features, test_labels = prepare_feats_labels(test_df)

    train_features = train_features[:dgl_train_graph.num_nodes()]
    train_labels = train_labels[:dgl_train_graph.num_nodes()]

    test_features = test_features[:dgl_test_graph.num_nodes()]
    test_labels = test_labels[:dgl_test_graph.num_nodes()]

    in_feats = train_features.shape[1]
    h_feats = 16
    num_classes = 2

    model = GNN(in_feats, h_feats, num_classes)
    train_model(model, dgl_train_graph, train_features, train_labels)
    evaluate_model(model, dgl_test_graph, test_features, test_labels)
