import graph_generator
import visualize_graph
from train_GraphSAGE import load_graph, train_graphsage
import torch

# Generate graph
G, node_df, edge_df = graph_generator.generate_crm_graph(num_nodes=50, num_edges=2)

# Visualize graph
visualize_graph.visualize_graph(G)

# Train GraphSAGE
data = load_graph()
model, embeddings = train_graphsage(data)

# Save or inspect embeddings
print("Sample embeddings:", embeddings[:5])
torch.save(embeddings, "node_embeddings.pt")
print("Embeddings saved to node_embeddings.pt")
