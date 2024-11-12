import torch
from gnn_model import GNN
from training_script import train
from data_setup import create_train_dataloader, create_val_dataloader

# Define file paths and directories
GRAPH_PATH = "../data/transformed/graph.pth"
LOG_DIR = "../data/logs/gnn_logs"
CHECKPOINTS_SAVE_PATH = "../data/checkpoints/gnn_checkpoints"

def main():
    """
    Main function to initialize data, model, and training parameters, then start the training process.
    Loads graph data, sets up the GNN model, loss function, optimizer, and configures training parameters.
    """
    # Load graph data
    data = torch.load(GRAPH_PATH, weights_only=False)

    # Set device for computation (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the GNN model with specified architecture parameters
    model = GNN(
        product_encoder_in_channels=770,
        product_encoder_hidden_channels=512,
        product_encoder_out_channels=128,
        gnn_encoder_hidden_channels=128,
        gnn_encoder_out_channels=64,
        graph_edge_dim=13,
        graph=data
    ).to(device)  # Move model to device

    # Set up the loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Training parameters
    epochs = 30
    batch_size = 8192

    # Start training
    train(
        model=model,
        train_batchsize=batch_size,
        train_dataloader=create_train_dataloader,
        val_batchsize=batch_size,
        val_dataloader=create_val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        checkpoints_save_path=CHECKPOINTS_SAVE_PATH,
        logging_directory=LOG_DIR
    )

if __name__ == "__main__":
    main()
