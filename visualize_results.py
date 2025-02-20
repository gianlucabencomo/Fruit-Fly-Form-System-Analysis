import typer
import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_results(
    file_path: str, save: bool = False
):
    
    # Load the saved data (dictionaries saved as numpy objects)
    data = np.load(file_path, allow_pickle=True)
    train_results = data["train_results"].item()  # dict with keys like 'agn', 'bn', etc.
    test_results  = data["test_results"].item()
    
    # Determine normalization methods and number of epochs from one of the norm methods.
    norm_methods = list(train_results.keys())
    num_epochs = len(train_results[norm_methods[0]][0])  # Each seed run is a list of (accuracy, loss) tuples.
    epochs = np.arange(1, num_epochs + 1)
    
    # Set a clean style for publication-quality figures.
    # (Note: text.usetex is False by default here; change to True if LaTeX is available)
    plt.rcParams.update({
        "font.size": 12,
        "text.usetex": False,
    })
    
    # Prepare color cycle for different normalization methods.
    colors = plt.cm.tab10(np.linspace(0, 1, len(norm_methods)))
    
    # Create figure for training metrics (accuracy and loss)
    fig_train, axes_train = plt.subplots(1, 2, figsize=(14, 6))
    ax_train_acc, ax_train_loss = axes_train
    
    # Create figure for testing metrics (accuracy and loss)
    fig_test, axes_test = plt.subplots(1, 2, figsize=(14, 6))
    ax_test_acc, ax_test_loss = axes_test
    
    # Loop over each normalization method and each seed run.
    for idx, norm in enumerate(norm_methods):
        # Convert list-of-seed results to numpy arrays.
        # Shape: (num_seeds, num_epochs, 2) where the last dimension is (accuracy, loss)
        train_arr = np.array(train_results[norm])
        test_arr  = np.array(test_results[norm])
        num_seeds = train_arr.shape[0]
        
        for seed in range(num_seeds):
            # Extract individual seed's accuracy and loss curves.
            train_acc = train_arr[seed, :, 0]
            train_loss = train_arr[seed, :, 1]
            test_acc = test_arr[seed, :, 0]
            test_loss = test_arr[seed, :, 1]
            
            # For the first seed of each norm method, add a legend label.
            label = norm if seed == 0 else None
            
            # Plot training metrics (using dashed lines)
            ax_train_acc.plot(epochs, train_acc, linestyle="--", color=colors[idx], alpha=0.7, label=label)
            ax_train_loss.plot(epochs, train_loss, linestyle="--", color=colors[idx], alpha=0.7, label=label)
            
            # Plot testing metrics (using solid lines)
            ax_test_acc.plot(epochs, test_acc, linestyle="-", color=colors[idx], alpha=0.7, label=label)
            ax_test_loss.plot(epochs, test_loss, linestyle="-", color=colors[idx], alpha=0.7, label=label)
    
    # Customize the training figure.
    ax_train_acc.set_title("Training Accuracy")
    ax_train_acc.set_xlabel("Epoch")
    ax_train_acc.set_ylabel("Accuracy")
    ax_train_acc.legend(fontsize=10, loc="best")
    ax_train_acc.grid(True)
    
    ax_train_loss.set_title("Training Loss")
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.legend(fontsize=10, loc="best")
    ax_train_loss.grid(True)
    
    fig_train.suptitle("Training Metrics", fontsize=14)
    fig_train.tight_layout(rect=[0, 0, 1, 0.95])
    save_path_train = file_path[:-4] + "_train.png"
    if save:
        fig_train.savefig(save_path_train, dpi=300)
    
    # Customize the testing figure.
    ax_test_acc.set_title("Testing Accuracy")
    ax_test_acc.set_xlabel("Epoch")
    ax_test_acc.set_ylabel("Accuracy")
    ax_test_acc.legend(fontsize=10, loc="best")
    ax_test_acc.grid(True)
    
    ax_test_loss.set_title("Testing Loss")
    ax_test_loss.set_xlabel("Epoch")
    ax_test_loss.set_ylabel("Loss")
    ax_test_loss.legend(fontsize=10, loc="best")
    ax_test_loss.grid(True)
    
    fig_test.suptitle("Testing Metrics", fontsize=14)
    fig_test.tight_layout(rect=[0, 0, 1, 0.95])
    save_path_test = file_path[:-4] + "_test.png"
    if save:
        fig_test.savefig(save_path_test, dpi=300)
    
    # Optionally display the figures.
    plt.show()

def main(path: str = "results/cifar100_64_0_adamw_200.npz", save: bool = False):
    print(path)
    load_and_visualize_results(file_path=path, save=save)

if __name__ == "__main__":
    typer.run(main)
