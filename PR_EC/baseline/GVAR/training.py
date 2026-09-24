# Training procedures for GVAR
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import MSELoss

import numpy as np

from utils import construct_training_dataset

from models.senn import SENNGC



def run_epoch(epoch_num: int, model: nn.Module, optimiser: optim, predictors: np.ndarray, responses: np.ndarray,
              time_idx: np.ndarray, criterion: torch.nn.modules.loss, lmbd: float, gamma: float, batch_size: int,
              device: torch.device, alpha=0.5, verbose=True, train=True):
    """
    Runs one epoch through the dataset.

    @param epoch_num: number of the epoch (for bookkeeping only).
    @param model: model.
    @param optimiser: Torch optimiser.
    @param predictors: numpy array with predictor values of shape [N x K x p].
    @param responses: numpy array with response values of shape [N x p].
    @param time_idx: time indices of observations of shape [N].
    @param criterion: base loss criterion (e.g. MSE or CE).
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param batch_size: batch size.
    @param device: Torch device.
    @param alpha: alpha-parameter for the elastic-net (default: 0.5).
    @param verbose: print-outs enabled?
    @param train: training mode?
    @return: if train == False, returns generalised coefficient matrices and average losses incurred; otherwise, None.
    """
    if not train:
        coeffs_final = torch.zeros((predictors.shape[0], predictors.shape[1], predictors.shape[2],
                                    predictors.shape[2])).to(device)

    # Shuffle the data
    inds = np.arange(0, predictors.shape[0])
    if train:
        np.random.shuffle(inds)

    # Split into batches
    batch_split = np.arange(0, len(inds), batch_size)
    if len(inds) - batch_split[-1] < batch_size / 2:
        batch_split = batch_split[:-1]

    incurred_loss = 0
    incurred_base_loss = 0
    incurred_penalty = 0
    incurred_smoothness_penalty = 0
    for i in range(len(batch_split)):
        if i < len(batch_split) - 1:
            predictors_b = predictors[inds[batch_split[i]:batch_split[i + 1]], :, :]
            responses_b = responses[inds[batch_split[i]:batch_split[i + 1]], :]
            time_idx_b = time_idx[inds[batch_split[i]:batch_split[i + 1]]]
        else:
            predictors_b = predictors[inds[batch_split[i]:], :, :]
            responses_b = responses[inds[batch_split[i]:], :]
            time_idx_b = time_idx[inds[batch_split[i]:]]

        inputs = torch.tensor(predictors_b, dtype=torch.float, device=device)
        targets = torch.tensor(responses_b, dtype=torch.float, device=device)

        # Get the forecasts and generalised coefficients
        preds, coeffs = model(inputs=inputs)
        if not train:
            if i < len(batch_split) - 1:
                coeffs_final[inds[batch_split[i]:batch_split[i + 1]], :, :, :] = coeffs
            else:
                coeffs_final[inds[batch_split[i]:], :, :, :] = coeffs

        # Loss
        # Base loss
        base_loss = criterion(preds, targets)

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        penalty = (1 - alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                  alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

        # Smoothing penalty term
        next_time_points = time_idx_b + 1
        inputs_next = torch.tensor(
            predictors[np.where(np.isin(time_idx, next_time_points))[0], :, :],
            dtype=torch.float,
            device=device,
        )
        preds_next, coeffs_next = model(inputs=inputs_next)
        penalty_smooth = torch.norm(coeffs_next - coeffs[np.isin(next_time_points, time_idx), :, :, :], p=2)

        loss = base_loss + lmbd * penalty + gamma * penalty_smooth

        # Incur loss
        incurred_loss += loss.data.cpu().numpy()
        incurred_base_loss += base_loss.data.cpu().numpy()
        incurred_penalty += lmbd * penalty.data.cpu().numpy()
        incurred_smoothness_penalty += gamma * penalty_smooth.data.cpu().numpy()

        if train:
            # Make an optimisation step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    if verbose:
        print("Epoch " + str(epoch_num) + " : incurred loss " + str(incurred_loss) + "; incurred sparsity penalty " +
              str(incurred_penalty) + "; incurred smoothness penalty " + str(incurred_smoothness_penalty))

    if not train:
        return coeffs_final, incurred_loss / len(batch_split), incurred_base_loss / len(batch_split), \
               incurred_penalty / len(batch_split), incurred_smoothness_penalty / len(batch_split)


def training_procedure(data, order: int, hidden_layer_size: int, end_epoch: int, batch_size: int, lmbd: float,
                       gamma: float, seed=42, num_hidden_layers=1, initial_learning_rate=0.001, beta_1=0.9,
                       beta_2=0.999, use_cuda=True, verbose=True, test_data=None):
    """
    Standard training procedure for GVAR model.

    @param data: numpy array with time series of shape [T x p].
    @param order: GVAR model order.
    @param hidden_layer_size: number of units in a hidden layer.
    @param end_epoch: number of training epochs.
    @param batch_size: batch size.
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param seed: random generator seed.
    @param num_hidden_layers: number oh hidden layers.
    @param initial_learning_rate: learning rate.
    @param use_cuda: whether to use GPU?
    @param verbose: print-outs enabled?
    @param test_data: optional test data.
    @return: returns an estimate of the GC dependency structure, generalised coefficient matrices, and the test MSE,
    if test data provided.
    """
    # Set random generator seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Check for CUDA availability
    if use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
        device = torch.device("cpu")
    elif use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Number of variables, p
    if isinstance(data, list):
        num_vars = data[0].shape[1]
    else:
        num_vars = data.shape[1]

    # Construct a training dataset with lagged time series values as predictors and future values as responses
    predictors, responses, time_idx = construct_training_dataset(data=data, order=order)

    if test_data is not None:
        # Construct a test set with lagged time series values as predictors and future values as responses
        predictors_test, responses_test, time_idx_test = construct_training_dataset(data=test_data, order=order)

    # Model definition
    senn = SENNGC(num_vars=num_vars, order=order, hidden_layer_size=hidden_layer_size,
                  num_hidden_layers=num_hidden_layers, device=device)
    senn.to(device=device)

    # Optimiser
    optimiser = optim.Adam(params=senn.parameters(), lr=initial_learning_rate, betas=(beta_1, beta_2))

    # Loss criterion
    criterion = MSELoss()

    # Run the training and testing
    for epoch in range(end_epoch):
        if verbose:
            print()

        # Train
        run_epoch(epoch_num=epoch, model=senn, optimiser=optimiser, predictors=predictors, responses=responses,
                  time_idx=time_idx, criterion=criterion, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                  device=device, train=True, verbose=verbose)

    # Compute generalised coefficients & estimate causal structure
    with torch.no_grad():
        coeffs, l, mse, pen1, pen2 = run_epoch(epoch_num=end_epoch, model=senn, optimiser=optimiser,
                                               predictors=predictors, responses=responses, time_idx=time_idx,
                                               criterion=criterion, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                                               device=device, train=False, verbose=verbose)
        coefficients_abs = coeffs.abs().cpu().numpy()
        causal_struct_estimate = np.max(np.median(coefficients_abs, axis=0), axis=0)

        if test_data is not None:
            # Make predictions on the test data
            coeffs_test, l_test, mse_test, pen1_test, pen2_test = run_epoch(epoch_num=(end_epoch+1), model=senn,
                                                optimiser=optimiser, predictors=predictors_test,
                                                responses=responses_test,  time_idx=time_idx_test, criterion=criterion,
                                                lmbd=lmbd, gamma=gamma,  batch_size=batch_size, device=device,
                                                train=False, verbose=verbose)
            # Return test MSE in addition to the inference results
            return causal_struct_estimate, coeffs.cpu().numpy(), mse_test
        else:
            return causal_struct_estimate, coeffs.cpu().numpy()
