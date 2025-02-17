import typer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Union


def load_and_preprocess_visual_system(
    subsystem_selection: Union[str, List[str]] = [
        "Dm1",
        "Dm4",
        "Dm6",
        "Dm10",
        "Dm12",
        "Dm13",
        "Dm14",
        "Dm15",
        "Dm16",
        "Dm17",
        "Dm18",
        "Dm19",
        "Dm20",
    ],
    connections_file: str = "./data/connections_no_threshold.csv",
    types_file: str = "./data/visual_neuron_types.csv",
    syn_threshold: int = 1,
):
    # Load raw data
    df_connections = pd.read_csv(connections_file)
    df_types = pd.read_csv(types_file)

    # generate mask for system types
    if isinstance(subsystem_selection, list):
        mask = df_types["type"].isin(subsystem_selection)
    else:
        mask = df_types["subsystem"] == subsystem_selection
    system_types = df_types[mask].reset_index(drop=True)

    # get all ids belonging to valid system types
    filtered_ids = system_types["root_id"].values  # cell ids
    df_filtered = df_connections[df_connections["post_root_id"].isin(filtered_ids)]
    df_merged = df_filtered.merge(
        df_types, left_on="pre_root_id", right_on="root_id", how="left"
    )

    dm_cell_map = df_types.set_index("root_id")["type"].to_dict()
    df_merged["Dm_type"] = df_merged["post_root_id"].map(dm_cell_map)

    connections_per_dm = (
        df_merged.groupby("Dm_type")
        .apply(
            lambda x: {
                "pre_root_ids": x["pre_root_id"].tolist(),
                "pre_types": x["type"].tolist(),
                "syn_counts": x["syn_count"].tolist(),
            }
        )
        .to_dict()
    )

    aggregated_synapse_counts = (
        df_merged.groupby(["Dm_type", "type"])["syn_count"].sum().unstack(fill_value=0)
    )
    num_dm_types = len(aggregated_synapse_counts.index)
    cols = 5  # Number of columns in the subplot grid
    rows = (num_dm_types // cols) + (
        num_dm_types % cols > 0
    )  # Compute rows dynamically

    # Create figure and axes for subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten in case of 2D array

    # Iterate over each Dm_type and create subplots
    weights, Hs = [], []
    for idx, (dm_type, ax) in enumerate(zip(aggregated_synapse_counts.index, axes)):
        # Get synapse counts for each pre_type input
        synapse_counts = aggregated_synapse_counts.loc[dm_type]
        p = synapse_counts / synapse_counts.sum()
        p = p[p != 0]  # Remove zero values
        H = -np.sum(p * np.log(p))  # Compute entropy
        weights.append(len(df_types[df_types["type"] == dm_type]))
        Hs.append(H)

        # Sort and select top 10 pre-synaptic cell types
        top_p = p.sort_values(ascending=False).iloc[:10]
        top_labels = top_p.index.tolist()  # Get pre-synaptic cell names

        # Plot histogram
        ax.bar(top_labels, top_p, alpha=0.7, edgecolor="black")
        ax.axhline(y=10 * 1 / len(p), alpha=0.7, ls="--", c="red")
        ax.set_title(f"{dm_type}")
        ax.set_ylabel("In Percentage")
        ax.set_xticklabels(top_labels, rotation=60)  # Rotate labels for readability

        ax.text(
            0.95,
            0.95,
            f"H[p] = {H:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"),
        )

    # Remove empty subplots if any
    for idx in range(len(aggregated_synapse_counts.index), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    weights, Hs = np.array(weights), np.array(Hs)
    weights = weights / weights.sum()


load_and_preprocess_visual_system()
