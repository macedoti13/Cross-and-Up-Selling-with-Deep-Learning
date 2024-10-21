from torch.utils.tensorboard.writer import SummaryWriter
from .model import GNN
import torch
import os

def prepare_batch_for_fitting(batch):
    products = batch["product"].x
    x_dict = batch.x_dict

    edge_index_dict = {
        ("customer", "bought", "product"): batch[("customer", "bought", "product")].edge_index,
        ("product", "rev_bought", "customer"): batch[("product", "rev_bought", "customer")].edge_index
    }

    edge_attr_dict = {
        ("customer", "bought", "product"): batch[("customer", "bought", "product")].edge_attr,
        ("product", "rev_bought", "customer"): batch[("product", "rev_bought", "customer")].edge_attr
    }

    edge_label_index = batch[("customer", "bought", "product")].edge_label_index
    labels = batch[("customer", "bought", "product")].edge_label

    return products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, labels

def train(
    model: GNN,
    train_batchsize: int,
    train_dataloader,
    val_batchsize: int,
    val_dataloader,
    optimizer: torch.optim.Optimizer,
    criterion,
    epochs: int,
    device: torch.device,
    checkpoints_save_path: str,
    logging_directory: str,
    validate_every_n_batches: int = 500  # Validate every N batches
) -> None:

    # Create logging directory and instantiate writer (tensorboard)
    os.makedirs(logging_directory, exist_ok=True)
    writer = SummaryWriter(logging_directory)

    # Create checkpoints directory
    os.makedirs(checkpoints_save_path, exist_ok=True)

    # Add model to device
    model.to(device)

    # Initialize table header
    print(f"{'Epoch':<10}{'Batch':<10}{'Train Loss':<15}{'Validation Loss':<15}")

    # Initialize val loss to infinity
    start_val_loss = float('inf')

    # Track steps for batch logging
    global_train_step = 0
    validation_step = 0  # Separate step count for validation

    # Loop through epochs
    for i in range(1, epochs+1):

        ### Train model
        model.train()

        # Create train batches
        train_loader = train_dataloader(batch_size=train_batchsize)

        # Iterate through batches
        for batch_idx, batch in enumerate(train_loader):

            # Add batch to device
            batch.to(device)

            # Prepare data for fitting
            products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, labels = prepare_batch_for_fitting(batch)

            # Prepare optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index)

            # Calculate loss
            train_loss = criterion(outputs, labels)

            # Backward pass
            train_loss.backward()

            # Update weights
            optimizer.step()

            # Log train loss per batch
            writer.add_scalar('train/train_batch', train_loss.item(), global_train_step)

            # Update gradients histogram after each batch
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, global_train_step)

            global_train_step += 1  # Increment global train step

            # Perform validation every N batches
            if (batch_idx + 1) % validate_every_n_batches == 0:
                model.eval()
                with torch.no_grad():
                    val_loader = val_dataloader(batch_size=val_batchsize)
                    val_batch_losses = []
                    for val_batch in val_loader:
                        val_batch.to(device)
                        products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, labels = prepare_batch_for_fitting(val_batch)
                        val_outputs = model(products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index)
                        val_loss = criterion(val_outputs, labels)
                        val_batch_losses.append(val_loss.item())

                    avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)

                    # Log validation loss after every N batches
                    writer.add_scalar('validation/validation_batch', avg_val_loss, validation_step)
                    validation_step += 1  # Increment validation step

                    # Print train and validation loss when validation is performed
                    print(f"{i:<10}{batch_idx+1:<10}{train_loss.item():<15.4f}{avg_val_loss:<15.4f}")

                    # Save checkpoint if validation loss improves
                    if avg_val_loss <= start_val_loss:
                        start_val_loss = avg_val_loss
                        torch.save(model.state_dict(), checkpoints_save_path + f"/model_checkpoint_batch_{global_train_step}.pth")

    writer.close()
