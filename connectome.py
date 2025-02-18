import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Union


def plot_synapse_distribution(aggregated_counts: pd.DataFrame, dm_counts: dict, direction: str = "input"):
    num_dm_types = len(aggregated_counts.index)
    cols = 5  # number of subplot columns
    rows = (num_dm_types // cols) + (num_dm_types % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()  # ensure we have a flat list of axes

    weights, entropies = [], []
    for idx, dm_type in enumerate(aggregated_counts.index):
        ax = axes[idx]
        # Get synapse counts for the given Dm cell type
        synapse_counts = aggregated_counts.loc[dm_type]
        total = synapse_counts.sum()
        if total == 0:
            continue

        # Compute normalized probabilities and entropy
        p = synapse_counts / total
        p = p[p != 0]  # remove zero entries
        H = -np.sum(p * np.log(p))
        entropies.append(H)
        weights.append(dm_counts.get(dm_type, 0))

        # Select and sort the top 10 contributing cell types
        top_p = p.sort_values(ascending=False).iloc[:10]
        top_labels = top_p.index.tolist()

        # Plot the histogram bars
        ax.bar(top_labels, top_p, alpha=0.7, edgecolor="black")
        if len(p) > 0:
            # Draw a horizontal line at 10/number-of-unique-cells as a reference threshold
            ax.axhline(y=10 / len(p), alpha=0.7, ls="--", c="red")
        ax.set_title(f"{dm_type}")
        ylabel = "In Percentage" if direction == "input" else "Out Percentage"
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(top_labels, rotation=60)
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

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return np.array(weights) / np.array(weights).sum(), np.array(entropies)


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

    # Select the system types (Dm cell types) you want to analyze
    if isinstance(subsystem_selection, list):
        mask = df_types["type"].isin(subsystem_selection)
    else:
        mask = df_types["subsystem"] == subsystem_selection
    system_types = df_types[mask].reset_index(drop=True)

    # Get all cell ids belonging to valid system types
    filtered_ids = system_types["root_id"].values  # Dm cell ids

    # Build a mapping from cell id to type for later use
    dm_cell_map = df_types.set_index("root_id")["type"].to_dict()
    # Create a dictionary with counts of each Dm cell type for weighting later
    dm_counts = system_types["type"].value_counts().to_dict()

    # ====================================
    # Compute inputs to Dm cells
    # ====================================
    # Filter connections where the Dm cell is the post-synaptic partner
    df_filtered_in = df_connections[df_connections["post_root_id"].isin(filtered_ids)]
    # Merge to get the pre-synaptic cell type information
    df_merged_in = df_filtered_in.merge(
        df_types, left_on="pre_root_id", right_on="root_id", how="left"
    )
    # Map the post-synaptic id to its Dm type
    df_merged_in["Dm_type"] = df_merged_in["post_root_id"].map(dm_cell_map)
    # Aggregate synapse counts by (Dm cell type, pre-synaptic cell type)
    aggregated_in = (
        df_merged_in.groupby(["Dm_type", "type"])["syn_count"].sum().unstack(fill_value=0)
    )

    # Plot input distributions using the helper function
    print("Plotting input distributions (synapses into Dm cells)...")
    in_weights, in_entropies = plot_synapse_distribution(aggregated_in, dm_counts, direction="input")

    # ====================================
    # Compute outputs from Dm cells
    # ====================================
    # Filter connections where the Dm cell is the pre-synaptic partner
    df_filtered_out = df_connections[df_connections["pre_root_id"].isin(filtered_ids)]
    # Merge to get the post-synaptic cell type information
    df_merged_out = df_filtered_out.merge(
        df_types, left_on="post_root_id", right_on="root_id", how="left"
    )
    # Map the pre-synaptic (Dm) cell id to its Dm type
    df_merged_out["Dm_type"] = df_filtered_out["pre_root_id"].map(dm_cell_map)
    # Aggregate synapse counts by (Dm cell type, post-synaptic cell type)
    aggregated_out = (
        df_merged_out.groupby(["Dm_type", "type"])["syn_count"].sum().unstack(fill_value=0)
    )

    # Plot output distributions using the helper function
    print("Plotting output distributions (synapses from Dm cells)...")
    out_weights, out_entropies = plot_synapse_distribution(aggregated_out, dm_counts, direction="output")

    print(np.sum(in_weights * in_entropies))
    print(np.sum(out_weights * out_entropies))
    # Optionally, you might want to return these values for further analysis
    return {
        "input": {"weights": in_weights, "entropies": in_entropies},
        "output": {"weights": out_weights, "entropies": out_entropies},
    }


if __name__ == "__main__":
    load_and_preprocess_visual_system()
