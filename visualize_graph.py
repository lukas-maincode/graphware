import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import re


def load_graph_from_csv(nodes_path="nodes.csv", edges_path="edges.csv") -> nx.MultiDiGraph:
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edges_path)

    G = nx.MultiDiGraph()

    # Add nodes with attributes
    for _, row in node_df.iterrows():
        node_id = row['id']
        attrs = row.drop(labels=['id']).dropna().to_dict()
        G.add_node(node_id, **attrs)

    # Add edges
    for _, row in edge_df.iterrows():
        G.add_edge(row['source'], row['target'], type=row['type'])

    return G


def visualize_graph(G: nx.MultiDiGraph, output_html="crm_graph.html"):
    # Create a new Network instance with explicit parameters
    net = Network(
        height="100vh",
        width="100%",
        notebook=False,
        directed=True,
        bgcolor="#ffffff",
        font_color="black"
    )
    
    # Compute node positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)  # k controls spacing

    # Disable physics for a static layout
    net.set_options("""
    {
        "physics": {
            "enabled": false
        }
    }
    """)

    # Add nodes with improved spacing
    for node, attrs in G.nodes(data=True):
        label = attrs.get("label") or attrs.get("name") or node
        color = {
            'view': '#ffcc00',
            'component': '#00ccff',
            'uielement': '#66ff66',
            'action': '#ff6666',
            'entity': '#ccccff'
        }.get(attrs.get("type"), '#dddddd')
        x, y = pos[node]
        net.add_node(
            node,
            label=label,
            title=str(attrs),
            color=color,
            font={'size': 14},
            size=20,
            x=float(x)*1000,  # scale up for better spacing
            y=float(y)*1000,
            fixed=True
        )

    # Add edges with improved visibility
    for source, target, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "")
        net.add_edge(
            source,
            target,
            title=edge_type,
            label=edge_type,
            font={'size': 12},
            arrows={'to': {'enabled': True, 'scaleFactor': 1}},
            smooth={'type': 'continuous'}
        )

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_html)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the graph
    try:
        net.save_graph(output_html)
        remove_pyvis_footer(output_html)
        print(f"Graph successfully saved to: {output_html}")
    except Exception as e:
        print(f"Error saving graph: {str(e)}")
        net.write_html(output_html)
        remove_pyvis_footer(output_html)
        print(f"Graph saved using fallback method to: {output_html}")


def remove_pyvis_footer(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    # Remove everything after the closing </div> of the network
    cleaned_html = re.sub(r'(</div>\s*)<hr[\s\S]*', r'\1', html)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)


if __name__ == "__main__":
    try:
        G = load_graph_from_csv("nodes.csv", "edges.csv")
        visualize_graph(G)
    except Exception as e:
        print(f"Error: {str(e)}")
