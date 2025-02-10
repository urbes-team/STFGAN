import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

class TemporalMemory(nn.Module):
    """Memory module for handling temporal lag"""
    def __init__(self, hidden_size, num_lags=3):
        super(TemporalMemory, self).__init__()
        self.hidden_size = hidden_size
        self.num_lags = num_lags

        # Memory processing
        self.memory_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # Memory attention
        self.memory_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, current_features, memory_bank):
        # memory_bank shape: [batch_size, num_lags, hidden_size]
        batch_size = current_features.size(0)

        if memory_bank is None:
            # Initialize empty memory if none exists
            return current_features, torch.zeros(
                batch_size, self.num_lags, self.hidden_size,
                device=current_features.device
            )

        # Process memory sequence
        memory_out, _ = self.memory_gru(memory_bank)

        # Calculate attention weights
        expanded_current = current_features.unsqueeze(1).expand(-1, self.num_lags, -1)
        attention_input = torch.cat([expanded_current, memory_out], dim=-1)
        attention_weights = self.memory_attention(attention_input)

        # Apply attention to memory
        weighted_memory = (memory_out * attention_weights).sum(dim=1)

        # Update memory bank - shift and add current features
        new_memory_bank = torch.cat([
            memory_bank[:, 1:, :],
            current_features.unsqueeze(1)
        ], dim=1)

        # Combine current features with weighted memory
        combined_features = current_features + weighted_memory

        return combined_features, new_memory_bank
    

class GCN_temporalmemory(nn.Module):
    def __init__(self,
                 num_node_features,
                 num_external_temporal_features,
                 n_lag_steps_ext=3,
                 n_lag_steps_target=3,
                 hidden_dim_gcn=64,
                 dropout_rate=0.1):
        """
        Initialize the GCN model with lagged temperature targets incorporated into node features.

        Args:
            num_node_features: Original number of input node features.
            num_external_temporal_features: Number of external temporal features.
            n_lag_steps_ext: Number of lag steps to incorporate (default: [3, 6, 12])
            hidden_dim_gcn: Hidden dimension size for GCN layers.
            dropout_rate: Dropout rate for regularization.
        """
        super(GCN_temporalmemory, self).__init__()

        # Total input features now include original node features plus lagged temperature targets
        total_input_features = num_node_features + n_lag_steps_target

        # Node embedding now accounts for total input features
        self.node_embedding = nn.Sequential(
            nn.Linear(total_input_features+num_external_temporal_features, hidden_dim_gcn),
            nn.LayerNorm(hidden_dim_gcn),
            nn.ELU()
        )

        # GAT(GCN) convolution layers
        #self.conv1 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)
        #self.conv2 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)
        self.conv1 = GATConv(hidden_dim_gcn, hidden_dim_gcn, heads=4, )
        self.conv2 = GATConv(4*hidden_dim_gcn, hidden_dim_gcn, heads=1)

        # Temporal feature processing
        self.temporal_net = nn.Sequential(
            nn.Linear(num_external_temporal_features, hidden_dim_gcn),
            nn.LayerNorm(hidden_dim_gcn),
            nn.ELU(),
            nn.Dropout(p=0.1)
        )

        # Memory module
        self.memory = TemporalMemory(hidden_dim_gcn, n_lag_steps_ext)

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim_gcn * 2, hidden_dim_gcn),
            nn.LayerNorm(hidden_dim_gcn),
            nn.ELU(),
            nn.Dropout(p=0.1)
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim_gcn, hidden_dim_gcn // 2),
            nn.LayerNorm(hidden_dim_gcn // 2),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim_gcn // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, temporal_features, lagged_targets):
      predictions = []
      temporal_memory_bank = None
      self.attention_weights_1=[]
      self.attention_weights_2=[]

      for t in range(temporal_features.size(0)):
          # Instead of using the same x, you might want to modify x based on temporal information
          # For example, you could add some temporal features to your node features
          temporal_t = temporal_features[t]
          x_with_temporal = torch.cat([x, temporal_t.unsqueeze(0).repeat(x.size(0), 1)], dim=-1)

          x_temp = torch.cat([x_with_temporal, lagged_targets[t]], dim=-1)

          # Transform node features
          x_temp = self.node_embedding(x_temp)

          # Process graph structure
          x_temp, self.attention_weights_1 = self.conv1(x_temp, edge_indexreturn_attention_weights=True)
          #x_temp= self.conv1(x_temp, edge_index)
          x_temp = torch.relu(x_temp)
          x_temp, self.attention_weights_2 = self.conv2(x_temp, edge_index, return_attention_weights=True)
          #x_temp = self.conv2(x_temp, edge_index)
          x_temp = torch.relu(x_temp)

          # Process temporal features
          temporal_features_processed = self.temporal_net(temporal_t.expand(x_temp.size(0), -1))

          # Apply memory mechanism to temporal features
          temporal_features_processed, temporal_memory_bank = self.memory(
              temporal_features_processed,
              temporal_memory_bank
          )

          # Fusion spatial and temporal features
          combined = torch.cat([x_temp, temporal_features_processed], dim=-1)
          fused_features = self.fusion_layer(combined)

          # Generate prediction
          out = self.output_layers(fused_features)
          predictions.append(out)

      # Stack predictions
      predictions = torch.stack(predictions, dim=0).squeeze(-1)
      return predictions

def train_model_with_lag(
    data,
    test_data,
    temporal_data,
    temporal_features_test,
    epochs=100,
    batch_size=64,
    num_lags_ext=12,
    lag_steps=[3, 6, 12],
    learning_rate=0.001,
    train_mask=None,
    test_mask=None,
    patience=10,
    save_best_model_path="best_model.pth"
):
    """
    Training function with early stopping and saving the best model.
    """
    model = GCN_temporalmemory(
        num_node_features=data.x.size(-1),
        num_external_temporal_features=temporal_data.size(1),
        n_lag_steps_ext=num_lags_ext,
        n_lag_steps_target=len(lag_steps),
        hidden_dim_gcn=128,
        dropout_rate=0.3
    )
    device = torch.device('cuda')
    model = model.to(device)
    data = data.to(device)
    temporal_data = temporal_data.to(device)
    lagged_targets = create_lagged_targets(data.y, lag_steps).to(device)

    if train_mask is not None:
        train_mask = torch.from_numpy(train_mask).type(torch.bool).to(device)

    if test_mask is not None:
        test_mask = torch.from_numpy(test_mask).type(torch.bool).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Early stopping variables
    best_rmse = float('inf')
    best_epoch = -1
    patience_counter = 0

    train_loss_list = []
    mae_pred_list = []
    rmse_pred_list = []
    mse_pred_list = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        for i in range(0, len(data.y) - max(lag_steps), batch_size):
            batch_indices = list(range(i, min(i + batch_size, temporal_data.size(0))))
            if max(batch_indices) > len(lagged_targets):
                continue

            batch_temporal = temporal_data[batch_indices]
            batch_y = data.y[batch_indices]

            # Verify no NaN values
            assert data.x.isnan().sum() == 0
            assert data.edge_index.isnan().sum() == 0
            assert data.y.isnan().sum() == 0
            assert temporal_data.isnan().sum() == 0

            optimizer.zero_grad()
            lagged_batch = lagged_targets[batch_indices]
            out = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                batch_temporal,
                lagged_batch
            )

            loss = criterion(out[train_mask[batch_indices]], batch_y[train_mask[batch_indices]])
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if torch.isnan(loss):
                print("Loss is NaN")
                print(out[train_mask[batch_indices]], batch_y[train_mask[batch_indices]])
                break

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches
        train_loss_list.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            test_data = test_data.to('cuda')
            temporal_features_test = temporal_features_test.to('cuda')

            lagged_targets_test = create_lagged_targets(test_data.y, lag_steps=[24, 48, 72])
            predictions = model(test_data.x, test_data.edge_index, test_data.edge_attr,
                                temporal_features_test[:-72], lagged_targets_test)
            pred_mask = test_mask[:-72].cpu().numpy() & ~np.isnan(predictions.cpu().numpy())

            # Compute mae for validation
            mae_pred = mean_absolute_error(
                predictions.cpu().numpy()[pred_mask],
                test_data.y[:-72].cpu().numpy()[pred_mask]
            )
            rmse_pred = root_mean_squared_error(
                predictions.cpu().numpy()[pred_mask],
                test_data.y[:-72].cpu().numpy()[pred_mask]
            )
            mse_pred = mean_squared_error(
                predictions.cpu().numpy()[pred_mask],
                test_data.y[:-72].cpu().numpy()[pred_mask]
            )
            mae_pred_list.append(mae_pred)
            rmse_pred_list.append(rmse_pred)
            mse_pred_list.append(mse_pred)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, MAE pred: {mae_pred:.4f}, RMSE_pred: {rmse_pred}, MSE pred : {mse_pred}')

        # Early stopping
        if rmse_pred < best_rmse:
            best_rmse = rmse_pred
            best_epoch = epoch
            patience_counter = 0

            # Save the best model
            best_model = model.state_dict()

        else:
            patience_counter += 1
            print(f"No improvement in MSE. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered. Best MSE: {best_rmse:.4f} at epoch {best_epoch+1}.")
            break

    # Load the best model before returning
    torch.save(model.state_dict(), save_best_model_path)
    model.load_state_dict(torch.load(save_best_model_path))
    print(f"Model loaded from epoch {best_epoch+1} with MSE: {best_rmse:.4f}")
    return model, lagged_targets, train_loss_list, mae_pred_list, rmse_pred_list, mse_pred_list
