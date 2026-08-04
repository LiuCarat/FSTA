"""Independent deterministic classifier for frozen BEC matrices."""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .brainnetcnn import DirectedBrainNetCNN
from .metrics import classification_metrics, select_youden_threshold
from Graph_BEC.data import set_seed
from Graph_BEC.normative_bec import to_directed_channels


def train_classifier(train_bec, train_labels, val_bec, val_labels,
                     test_bec, test_labels, device, seed,
                     max_epochs=80, patience=12, batch_size=32,
                     learning_rate=1e-3):
    """Use a stage-local seed and keep test labels out of model selection."""
    set_seed(seed)
    model = DirectedBrainNetCNN(nodes_num=train_bec.shape[-1], dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_x = to_directed_channels(torch.from_numpy(train_bec)).float()
    train_y = torch.from_numpy(train_labels).float()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size,
        shuffle=True, generator=generator, num_workers=0,
    )
    validation_x = to_directed_channels(torch.from_numpy(val_bec)).float().to(device)
    validation_y = torch.from_numpy(val_labels).float().to(device)
    best_state, best_loss, waiting = None, float("inf"), 0
    for _ in range(max_epochs):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_x), validation_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(model(validation_x)).cpu().numpy()
        test_probabilities = torch.sigmoid(
            model(to_directed_channels(torch.from_numpy(test_bec)).float().to(device))
        ).cpu().numpy()
    threshold = select_youden_threshold(val_labels, validation_probabilities)
    return classification_metrics(test_labels, test_probabilities, threshold), model
