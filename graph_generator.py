import networkx as nx
import random
import uuid
from typing import Tuple, Dict, Any
import pandas as pd


def generate_crm_graph(num_nodes: int, num_edges: int) -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame]:
    # Define possible node and edge types
    node_types = ['view', 'component', 'uielement', 'action', 'entity']
    edge_types = ['navigates_to', 'contains', 'triggers', 'performs', 'reads', 'writes', 'binds']

    G = nx.MultiDiGraph()
    node_registry = {nt: [] for nt in node_types}
    node_data = []

    # Generate nodes
    for _ in range(num_nodes):
        ntype = random.choice(node_types)
        nid = f"{ntype}_{uuid.uuid4().hex[:6]}"
        attrs: Dict[str, Any] = {'id': nid, 'type': ntype}

        if ntype == 'view':
            attrs['name'] = f"View {random.randint(1,100)}"
            attrs['role_visibility'] = random.sample(['admin', 'rep', 'sales', 'finance'], k=random.randint(1, 2))
        elif ntype == 'component':
            attrs['name'] = f"Component {random.randint(1,100)}"
            attrs['parent_view'] = random.choice(node_registry['view']) if node_registry['view'] else None
        elif ntype == 'uielement':
            attrs['subtype'] = random.choice(['button', 'input', 'dropdown'])
            attrs['label'] = f"{attrs['subtype'].capitalize()} {random.randint(1,100)}"
            attrs['parent_component'] = random.choice(node_registry['component']) if node_registry['component'] else None
        elif ntype == 'action':
            attrs['label'] = f"Action {random.randint(1,100)}"
            attrs['target_entity'] = random.choice(node_registry['entity']) if node_registry['entity'] else None
        elif ntype == 'entity':
            attrs['fields'] = [f"field_{i}" for i in range(random.randint(1, 3))]

        G.add_node(nid, **attrs)
        node_registry[ntype].append(nid)
        node_data.append(attrs)

    # Generate edges
    edge_data = []
    for _ in range(num_edges):
        etype = random.choice(edge_types)
        source, target = None, None

        if etype == 'navigates_to' and len(node_registry['view']) >= 2:
            source, target = random.sample(node_registry['view'], 2)
        elif etype == 'contains' and node_registry['view'] and node_registry['component']:
            source = random.choice(node_registry['view'])
            target = random.choice(node_registry['component'])
        elif etype == 'triggers' and node_registry['uielement'] and node_registry['action']:
            source = random.choice(node_registry['uielement'])
            target = random.choice(node_registry['action'])
        elif etype == 'performs' and node_registry['action'] and node_registry['entity']:
            source = random.choice(node_registry['action'])
            target = random.choice(node_registry['entity'])
        elif etype in ['reads', 'writes'] and (node_registry['view'] or node_registry['component']) and node_registry['entity']:
            source = random.choice(node_registry['view'] + node_registry['component'])
            target = random.choice(node_registry['entity'])
        elif etype == 'binds' and node_registry['uielement'] and node_registry['entity']:
            source = random.choice(node_registry['uielement'])
            target = random.choice(node_registry['entity'])

        if source and target:
            G.add_edge(source, target, type=etype)
            edge_data.append({'source': source, 'target': target, 'type': etype})

    node_df = pd.DataFrame(node_data)
    edge_df = pd.DataFrame(edge_data)

    # Print preview
    print("\nGenerated Nodes (preview):")
    print(node_df.head())
    print("\nGenerated Edges (preview):")
    print(edge_df.head())

    # Save to CSV
    node_df.to_csv("nodes.csv", index=False)
    edge_df.to_csv("edges.csv", index=False)
    print("\nData saved to 'nodes.csv' and 'edges.csv'.")

    return G, node_df, edge_df


# Run the generator
if __name__ == "__main__":
    generate_crm_graph(num_nodes=30, num_edges=40)
