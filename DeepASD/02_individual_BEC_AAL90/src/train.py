from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import evaluation_windows, read_roi_1d
from src.model import IndividualBEC3Modes


def masked_smooth_l1(prediction, target, mask):
    loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    mask = mask[:, None, :].to(loss.dtype)
    denominator = (mask.sum() * prediction.size(1)).clamp_min(1.0)
    return (loss * mask).sum() / denominator


def _ramp(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def bec_regularizers(internal_bec):
    absolute_bec = internal_bec.abs()
    sparse = absolute_bec.sum(dim=-1).mean()

    reverse_bec = absolute_bec.transpose(-1, -2)
    reciprocal_overlap = (
        (2.0 * absolute_bec * reverse_bec).sum(dim=(-1, -2))
        / (
            absolute_bec.square().sum(dim=(-1, -2))
            + reverse_bec.square().sum(dim=(-1, -2))
            + 1e-8
        )
    ).mean()
    return sparse, reciprocal_overlap


def compute_losses(model, batch, args, epoch):
    timeseries = batch["timeseries"].to(args.device)
    mask = batch["time_mask"].to(args.device)
    phenotype = batch["phenotype"].to(args.device)
    qc = batch["qc"].to(args.device)

    context_length = max(
        8,
        min(
            timeseries.size(-1) - 2,
            round(timeseries.size(-1) * args.context_ratio),
        ),
    )
    context = timeseries[:, :, :context_length]
    previous = timeseries[:, :, context_length - 1 : -1]
    target = timeseries[:, :, context_length:]
    target_mask = mask[:, context_length:]

    ramp = _ramp(epoch, args.adversarial_warmup_epochs)
    output = model(
        context,
        phenotype if args.use_phenotype else None,
        args.modality_grl_strength * ramp,
        args.qc_grl_strength * ramp,
    )

    prediction = model.predict_next(
        previous,
        output["internal_bec"],
        output["self_coeff"],
    )

    reconstruction = masked_smooth_l1(
        prediction,
        target,
        target_mask,
    )
    sparsity, reciprocal = bec_regularizers(output["internal_bec"])

    modality = reconstruction.new_tensor(0.0)
    if args.use_phenotype:
        modality = F.cross_entropy(
            output["modality_logits"],
            output["modality_targets"],
        )

    qc_loss = reconstruction.new_tensor(0.0)
    if args.use_qc_adversary:
        qc_loss = F.smooth_l1_loss(
            output["qc_prediction"],
            qc,
        )

    consistency = reconstruction.new_tensor(0.0)
    if args.lambda_consistency > 0 and context_length >= 20:
        middle = context_length // 2
        first = model(
            context[:, :, :middle],
            phenotype if args.use_phenotype else None,
            0.0,
            0.0,
        )["internal_bec"]
        second = model(
            context[:, :, middle:],
            phenotype if args.use_phenotype else None,
            0.0,
            0.0,
        )["internal_bec"]
        consistency = F.mse_loss(first, second)

    total = (
        reconstruction
        + args.lambda_sparse * sparsity
        + args.lambda_directional * reciprocal
        + args.lambda_consistency * consistency
        + args.effective_lambda_modal * modality
        + args.effective_lambda_qc * qc_loss
    )
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "sparsity": sparsity,
        "reciprocal": reciprocal,
        "consistency": consistency,
        "modality": modality,
        "qc": qc_loss,
    }


def run_epoch(model, loader, optimizer, args, epoch, training):
    model.train(training)
    rows = []

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            metrics = compute_losses(model, batch, args, epoch)
            if training:
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
                optimizer.step()

        rows.append(
            {
                key: float(value.detach().cpu())
                for key, value in metrics.items()
            }
        )

    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def train_model(
    train_loader,
    valid_loader,
    phenotype_dim,
    qc_dim,
    args,
    fold_dir,
):
    model = IndividualBEC3Modes(
        num_rois=args.num_rois,
        phenotype_input_dim=max(1, phenotype_dim),
        qc_dim=max(1, qc_dim),
        use_phenotype=args.use_phenotype,
        use_qc_adversary=args.use_qc_adversary,
        phenotype_hidden=args.phenotype_hidden,
        phenotype_dim=args.phenotype_dim,
        temporal_hidden=args.temporal_hidden,
        roi_dim=args.roi_dim,
        common_dim=args.common_dim,
        edge_dim=args.edge_dim,
        max_incoming_edges=args.max_incoming_edges,
        adversary_hidden=args.adversary_hidden,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    best_reconstruction = float("inf")
    best_state = None
    history = []
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args,
            epoch,
            True,
        )
        with torch.no_grad():
            valid_metrics = run_epoch(
                model,
                valid_loader,
                None,
                args,
                epoch,
                False,
            )

        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                },
                **{
                    f"valid_{key}": value
                    for key, value in valid_metrics.items()
                },
            }
        )
        print(
            f"{args.experiment_mode_name} "
            f"epoch={epoch:03d} "
            f"train={train_metrics['reconstruction']:.5f} "
            f"valid={valid_metrics['reconstruction']:.5f} "
            f"sparse={valid_metrics['sparsity']:.5f} "
            f"reciprocal={valid_metrics['reciprocal']:.5f}"
        )

        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            best_reconstruction = valid_metrics["reconstruction"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    fold_dir = Path(fold_dir)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    pd.DataFrame(history).to_csv(
        fold_dir / "training_history.csv",
        index=False,
    )
    return model, {
        "loss": best_loss,
        "reconstruction": best_reconstruction,
    }


def _masked_metrics(prediction, target, mask):
    mask = mask[:, None, :]
    denominator = max(float(mask.sum() * prediction.shape[1]), 1.0)
    difference = prediction - target
    mae = float((np.abs(difference) * mask).sum() / denominator)
    rmse = float(
        np.sqrt(((difference ** 2) * mask).sum() / denominator)
    )
    return mae, rmse


@torch.no_grad()
def extract_bec(
    model,
    manifest,
    indices,
    phenotype_array,
    qc_array,
    args,
):
    model.eval()
    result = {
        "bec": [],
        "self_coeff": [],
        "reconstruction_mae": [],
        "reconstruction_rmse": [],
        "edge_density": [],
        "asymmetry": [],
        "directionality_index": [],
        "reciprocity": [],
        "qc_true_standardized": [],
        "qc_adversary_prediction": [],
        "label": [],
        "subject_id": [],
        "site_id": [],
    }

    for index in indices:
        index = int(index)
        row = manifest.iloc[index]
        timeseries = read_roi_1d(
            row["roi_path"],
            args.num_rois,
            args.allow_aal116_to_aal90,
        )
        windows, masks = evaluation_windows(
            timeseries,
            args.window_length,
            args.eval_windows,
        )

        x = torch.from_numpy(windows).to(args.device)
        phenotype = torch.from_numpy(
            phenotype_array[index]
        ).to(args.device)
        phenotype = phenotype.unsqueeze(0).expand(len(windows), -1)

        context_length = max(
            8,
            min(
                x.size(-1) - 2,
                round(x.size(-1) * args.context_ratio),
            ),
        )
        context = x[:, :, :context_length]
        previous = x[:, :, context_length - 1 : -1]
        target = x[:, :, context_length:]

        output = model(
            context,
            phenotype if args.use_phenotype else None,
            0.0,
            0.0,
        )
        prediction = model.predict_next(
            previous,
            output["internal_bec"],
            output["self_coeff"],
        )

        internal_bec = output["internal_bec"].mean(dim=0).cpu().numpy()

        # Export convention:
        # bec[source, target] = source ROI -> target ROI
        saved_bec = internal_bec.T.copy()

        mae, rmse = _masked_metrics(
            prediction.cpu().numpy(),
            target.cpu().numpy(),
            masks[:, context_length:],
        )
        off_diagonal = ~np.eye(args.num_rois, dtype=bool)

        result["bec"].append(saved_bec)
        result["self_coeff"].append(
            output["self_coeff"].mean(dim=0).cpu().numpy()
        )
        result["reconstruction_mae"].append(mae)
        result["reconstruction_rmse"].append(rmse)
        result["edge_density"].append(
            float(
                np.mean(
                    np.abs(saved_bec[off_diagonal])
                    > args.edge_presence_threshold
                )
            )
        )
        result["asymmetry"].append(
            float(np.mean(np.abs(saved_bec - saved_bec.T)))
        )
        edge_magnitude = np.abs(saved_bec)
        reverse_magnitude = edge_magnitude.T
        result["directionality_index"].append(
            float(
                np.sum(np.abs(edge_magnitude - reverse_magnitude))
                / (np.sum(edge_magnitude + reverse_magnitude) + 1e-8)
            )
        )
        result["reciprocity"].append(
            float(
                np.sum(2.0 * edge_magnitude * reverse_magnitude)
                / (
                    np.sum(edge_magnitude ** 2)
                    + np.sum(reverse_magnitude ** 2)
                    + 1e-8
                )
            )
        )

        if args.use_qc_adversary:
            result["qc_true_standardized"].append(qc_array[index])
            result["qc_adversary_prediction"].append(
                output["qc_prediction"].mean(dim=0).cpu().numpy()
            )

        result["label"].append(int(row["label"]))
        result["subject_id"].append(str(row["subject_id"]))
        result["site_id"].append(str(row.get("site_id", "")))

    output = {}
    for key, values in result.items():
        if key in {"subject_id", "site_id"}:
            output[key] = np.asarray(values)
        elif key in {
            "qc_true_standardized",
            "qc_adversary_prediction",
        } and not args.use_qc_adversary:
            output[key] = np.empty((len(indices), 0), dtype=np.float32)
        else:
            output[key] = np.asarray(values)
    output["manifest_index"] = np.asarray(indices, dtype=np.int64)
    return output
