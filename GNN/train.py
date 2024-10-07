from data_setup import create_train_dataloader, create_val_dataloader, data
from model import GNN
from tqdm import tqdm
import torch
import os

def train(
    model,
    batch_size,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    epochs,
    device,    
    save_path="drive/MyDrive/SJ_PCD_24-2/gnn_checkpoints"
):
    # Add model to device
    model.to(device)

    # Create saving path directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Print header for output
    print(f"{'Epoch':^10}{'Train Loss':^15}{'Val Loss':^15}")
    print("-" * 45)

    # Loop through epochs
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        # Training loop
        batches = train_dataloader(batch_size=batch_size)
        with tqdm(batches, unit="batch", desc=f"Epoch {epoch}/{epochs}", leave=False) as pbar:
            for batch in pbar:
                # Move data to device
                batch.to(device)

                # Get data from batch
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

                # Zero the optimizer gradient
                optimizer.zero_grad()

                # Forward pass
                outputs = model(products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss and training accuracy
                running_loss += loss.item() * labels.size(0)
                total_train += labels.size(0)
                correct_train += ((outputs > 0.5).int() == labels).sum().item()

                pbar.set_postfix({"Batch Loss": loss.item()})

        # Calculate average training loss and accuracy
        avg_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_batches = val_dataloader(batch_size=batch_size)
        with torch.no_grad():
            for batch in val_batches:
                # Move data to device
                batch.to(device)

                # Get data from batch
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

                # Forward pass
                outputs = model(products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index)
                loss = criterion(outputs, labels)

                # Accumulate validation loss and accuracy
                val_loss += loss.item() * labels.size(0)
                total_val += labels.size(0)
                correct_val += (outputs.argmax(1) == labels).sum().item()

        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / total_val
        val_accuracy = correct_val / total_val

        # Save model checkpoint
        checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Print epoch summary
        print(f"{epoch:^10}{avg_train_loss:^15.4f}{avg_val_loss:^15.4f}")
        print(f"Model saved at: {checkpoint_path}")

    print("\nTraining Complete!")
    
    
def main():
    
    # Initialize the model
    model = GNN(
        product_encoder_in_channels=769,
        product_encoder_hidden_channels=512,
        product_encoder_out_channels=128,
        gnn_encoder_hidden_channels=128,
        gnn_encoder_out_channels=64,
        graph_edge_dim=15,
        graph=data  
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.BCELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10  
    batch_size = 1024

    # Now train the model using the train function
    train(
        model=model,
        batch_size=batch_size,
        train_dataloader=create_train_dataloader,
        val_dataloader=create_val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        save_path="drive/MyDrive/SJ_PCD_24-2/gnn_checkpoints"  # Path to save model checkpoints
    )    
        
        
if __name__ == "__main__":
    main()
