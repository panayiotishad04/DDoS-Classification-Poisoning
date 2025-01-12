# /bin/python3.11

# pyenv install 3.11
# pyenv global 3.11
# pip3.11 install tensorflow-gnn tensorflow tf-keras pandas matplotlib

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd


def build_dataframe() -> pd.DataFrame:
    files = ["Only_Benign_7-1.csv", "Only_Benign_34-1.csv", "Only_Benign_60-1.csv",
             "Only_DDOS_7-1.csv", "Only_DDOS_34-1.csv"]
    df = [pd.read_csv(file) for file in files]
    df = pd.concat(df, ignore_index=True)
    df['bytes'] = df[["orig_ip_bytes_count", "resp_bytes"]].sum(axis=1)
    df['packet_count'] = df[["orig_pkts_count", "resp_pkts_count"]].sum(axis=1)

    # preprocess duration
    df['duration_interval'] = df['duration_interval'].replace("-", 0, inplace=True)
    df['duration_interval'] = df['duration_interval'].astype(float)

    print(df.describe())
    print(df.head())
    print(df.columns)
    print(df.dtypes)
    return df


def build_tensor(df):
    nodes = {
        'source': tf.constant(df['id.orig_addr']),
        'destination': tf.constant(df['id.resp_haddr']),
    }

    ips = list(set(df['id.orig_addr'].tolist() + df['id.resp_haddr'].tolist()))
    ports = list(set(df['id.orig_port'].tolist() + df['id.resp_pport'].tolist()))

    node_features = {
        'ip_nodes': tf.constant(ips),
        'port_nodes': tf.constant(ports)
    }

    features = {
        'source_ports': tf.constant(df['id.orig_port']),
        'destination_ports': tf.constant(df['id.resp_pport']),
        'proto_enum': tf.constant(df['proto_enum']),
        'service_string': tf.constant(df['service_string']),
        'duration': tf.constant(df['duration_interval']),
        'miss_bytes_count': tf.constant(df['missed_bytes_count']),
        'bytes': tf.constant(df['bytes']),
        'packet_count': tf.constant(df['packet_count']),
        'label': tf.constant(df['Category']),
    }

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'ips': tfgnn.NodeSet.from_fields(
                sizes=[len(node_features['ip_nodes'])],
                features={'ip': node_features['ip_nodes']}
            ),
            'ports': tfgnn.NodeSet.from_fields(
                sizes=[len(node_features['port_nodes'])],
                features={'port': node_features['port_nodes']}
            )
        },
        edge_sets={
            'flows': tfgnn.EdgeSet.from_fields(
                sizes=[len(nodes['source'])],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('ips', tf.range(start=0, limit=len(nodes['source']))),
                    target=('ips', tf.range(start=0, limit=len(nodes['destination'])))
                ),
                features=features
            )
        }
    )
    return graph


if __name__ == "__main__":
    df = build_dataframe()
    ben = sum(df['Category'] == "Benign")
    mal = sum(df['Category'] == "Malicious")
    print(mal / (ben + mal))
    graph_tensor = build_tensor(df)
    print(graph_tensor)
