import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random


# Step 1: Load CSV and build PyG graph
def load_graph(nodes_path="nodes.csv", edges_path="edges.csv") -> Data:
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edges_path)

    # Index mapping
    node_ids = list(node_df['id'])
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    node_types = node_df['type'].fillna('unknown').values.reshape(-1, 1)

    # One-hot encode node types as features
    encoder = OneHotEncoder(sparse_output=False)
    x = encoder.fit_transform(node_types)
    x = torch.tensor(x, dtype=torch.float)

    # Edge index
    edges = edge_df[['source', 'target']].dropna()
    edge_index = torch.tensor(
        [[id2idx[s], id2idx[t]] for s, t in zip(edges['source'], edges['target'])],
        dtype=torch.long
    ).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# Step 2: Define GraphSAGE Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Step 3: Contrastive loss (InfoNCE-style)
def unsupervised_loss(z, pos_edge_index, neg_edge_index):
    def cosine_similarity(a, b):
        return F.cosine_similarity(a, b)

    pos_score = cosine_similarity(z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_score = cosine_similarity(z[neg_edge_index[0]], z[neg_edge_index[1]])

    loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    loss += -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
    return loss


# Step 4: Train unsupervised GraphSAGE
def train_graphsage(data: Data, epochs=200):
    model = GraphSAGE(data.num_node_features, 32, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Precompute negative samples (random pairs)
    N = data.num_nodes
    neg_edge_index = torch.randint(0, N, (2, data.edge_index.size(1)))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        loss = unsupervised_loss(z, data.edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, z.detach()


# Run everything
if __name__ == "__main__":
    data = load_graph()
    model, embeddings = train_graphsage(data)

    # Save or inspect embeddings
    print("Sample embeddings:", embeddings[:5])
    torch.save(embeddings, "node_embeddings.pt")
    print("Embeddings saved to node_embeddings.pt")
