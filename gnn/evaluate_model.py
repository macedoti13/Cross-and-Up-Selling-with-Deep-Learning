import torch
from gnn_model import GNN
from data_setup import create_test_dataloader
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from joblib import load

# Define file paths
GRAPH_PATH = "../data/transformed/graph.pth"
PRODUCT_SCALER_PATH = "../data/transformed/product_scaler.pkl"
CUSTOMER_SCALER_PATH = "../data/transformed/customer_scaler.pkl"
EDGE_SCALER_PATH = "../data/transformed/edge_scaler.pkl"
REV_EDGE_SCALER_PATH = "../data/transformed/rev_edge_scaler.pkl"
TEST_DATA_PATH = "../data/transformed/test_data.pth"
MODEL_PATH = "../data/checkpoints/gnn_checkpoints/model_checkpoint_batch_28640.pth"

def load_model_and_data():
    """
    Loads the GNN model, test data, and scalers for evaluation.

    Returns:
        tuple: The GNN model, test data, and scalers (product, customer, edge, rev edge).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph data
    data = torch.load(GRAPH_PATH, weights_only=False)

    # Load scalers
    product_scaler = load(PRODUCT_SCALER_PATH)
    customer_scaler = load(CUSTOMER_SCALER_PATH)
    edge_scaler = load(EDGE_SCALER_PATH)
    rev_edge_scaler = load(REV_EDGE_SCALER_PATH)

    # Load test data
    test_data = torch.load(TEST_DATA_PATH)

    # Initialize and load model weights
    model = GNN(
        product_encoder_in_channels=770,
        product_encoder_hidden_channels=512,
        product_encoder_out_channels=128,
        gnn_encoder_hidden_channels=128,
        gnn_encoder_out_channels=64,
        graph_edge_dim=13,
        graph=data
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    return model, test_data, product_scaler, customer_scaler, edge_scaler, rev_edge_scaler

def prepare_batch_for_fitting(batch):
    """
    Prepares a batch of data for model input.

    Args:
        batch (torch_geometric.data.Data): Batch from the test dataloader.

    Returns:
        tuple: Prepared inputs for the model, including products, node features, edge indices,
               edge attributes, edge label indices, and labels.
    """
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

def evaluate(model, test_dataloader, criterion, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained GNN model.
        test_dataloader (DataLoader): Dataloader for the test data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device on which to perform computation.

    Returns:
        tuple: Metrics - average loss, accuracy, precision, recall, F1 score, labels, and predictions.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch.to(device)
            products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, labels = prepare_batch_for_fitting(batch)
            outputs = model(products, x_dict, edge_index_dict, edge_attr_dict, edge_label_index)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float().cpu()  # Binary predictions
            labels = labels.cpu()

            all_predictions.append(predictions)
            all_labels.append(labels)

            # Batch accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Concatenate all labels and predictions
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_samples
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions

def plot_evaluation_metrics(all_labels, all_predictions):
    """
    Plots the confusion matrix and ROC curve with AUC for model evaluation.

    Args:
        all_labels (np.array): True labels for the test data.
        all_predictions (np.array): Predicted labels for the test data.
    """
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

def main():
    """
    Main function to run the model evaluation pipeline.
    Loads data and model, performs evaluation, and plots evaluation metrics.
    """
    model, test_data, _, _, _, _ = load_model_and_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCELoss()

    # Initialize test dataloader
    test_batch_size = 8192
    test_dataloader = create_test_dataloader(batch_size=test_batch_size)

    # Run evaluation
    avg_loss, acc, precision, recall, f1, all_labels, all_predictions = evaluate(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )

    # Plot evaluation metrics
    plot_evaluation_metrics(all_labels, all_predictions)

if __name__ == "__main__":
    main()
