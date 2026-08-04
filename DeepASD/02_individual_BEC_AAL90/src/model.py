from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(x, strength=1.0):
    return _GradientReverse.apply(x, strength)


class PhenotypeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ROITemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_dim, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, timeseries):
        batch, rois, time = timeseries.shape
        x = timeseries.reshape(batch * rois, 1, time)
        x = self.net(x).squeeze(-1)
        return x.reshape(batch, rois, self.output_dim)


class ModalityAlignment(nn.Module):
    """DeepASD-style PHENO-fMRI common-space alignment."""

    def __init__(self, phenotype_dim, fmri_dim, common_dim, hidden_dim):
        super().__init__()
        self.pheno_proj = nn.Linear(phenotype_dim, common_dim)
        self.fmri_proj = nn.Linear(fmri_dim, common_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pheno, fmri, grl_strength):
        pheno = F.normalize(self.pheno_proj(pheno), p=2, dim=-1)
        fmri = F.normalize(self.fmri_proj(fmri), p=2, dim=-1)

        features = torch.stack([pheno, fmri], dim=1)
        logits = self.discriminator(
            gradient_reverse(
                features.reshape(-1, features.size(-1)),
                grl_strength,
            )
        )
        targets = torch.tensor(
            [0, 1],
            device=features.device,
        ).repeat(features.size(0))

        return pheno, fmri, logits, targets


class QCAdversary(nn.Module):
    def __init__(self, input_dim, hidden_dim, qc_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, qc_dim),
        )

    def forward(self, x, grl_strength):
        return self.net(gradient_reverse(x, grl_strength))


class DirectedBECGenerator(nn.Module):
    """
    Internal matrix:
        internal_bec[target, source] = source -> target
    """

    def __init__(
        self,
        num_rois,
        roi_dim,
        condition_dim,
        edge_dim,
        max_incoming_edges=20,
        max_transition_norm=0.95,
    ):
        super().__init__()
        self.num_rois = num_rois
        self.max_incoming_edges = max_incoming_edges
        self.max_transition_norm = max_transition_norm

        self.film = nn.Linear(condition_dim, 2 * roi_dim)
        self.source_proj = nn.Linear(roi_dim, edge_dim)
        self.target_proj = nn.Linear(roi_dim, edge_dim)
        self.self_net = nn.Sequential(
            nn.Linear(roi_dim + condition_dim, roi_dim),
            nn.ReLU(),
            nn.Linear(roi_dim, 1),
        )
        self.register_buffer(
            "off_diagonal",
            1.0 - torch.eye(num_rois),
        )

    def forward(self, roi_features, condition):
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        features = (
            roi_features
            * (1.0 + 0.5 * torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )

        source = self.source_proj(features)
        target = self.target_proj(features)

        # Row=target, column=source for x_next = A @ x.
        scores = torch.einsum("bid,bjd->bij", target, source)
        scores = torch.tanh(scores / math.sqrt(source.size(-1)))
        scores = scores * self.off_diagonal

        if 0 < self.max_incoming_edges < self.num_rois - 1:
            strongest = scores.abs().topk(
                self.max_incoming_edges,
                dim=-1,
            ).indices
            sparse_mask = torch.zeros_like(scores).scatter_(
                -1,
                strongest,
                1.0,
            )
            scores = scores * sparse_mask

        threshold = scores.new_tensor(0.0)

        condition_roi = condition.unsqueeze(1).expand(
            -1,
            self.num_rois,
            -1,
        )
        self_coeff = 0.8 * torch.sigmoid(
            self.self_net(
                torch.cat([features, condition_roi], dim=-1)
            ).squeeze(-1)
        )

        row_l1 = scores.abs().sum(dim=-1, keepdim=True)
        limit = (
            self.max_transition_norm - self_coeff
        ).clamp(min=0.02).unsqueeze(-1)
        scale = torch.clamp(limit / (row_l1 + 1e-6), max=1.0)

        return scores * scale, self_coeff, threshold


class IndividualBEC3Modes(nn.Module):
    def __init__(
        self,
        num_rois,
        phenotype_input_dim,
        qc_dim,
        use_phenotype,
        use_qc_adversary,
        phenotype_hidden=64,
        phenotype_dim=32,
        temporal_hidden=32,
        roi_dim=64,
        common_dim=32,
        edge_dim=64,
        max_incoming_edges=20,
        adversary_hidden=64,
        dropout=0.1,
    ):
        super().__init__()
        self.use_phenotype = use_phenotype
        self.use_qc_adversary = use_qc_adversary

        self.temporal_encoder = ROITemporalEncoder(
            temporal_hidden,
            roi_dim,
            dropout,
        )
        self.fmri_projection = nn.Sequential(
            nn.Linear(roi_dim, common_dim),
            nn.LayerNorm(common_dim),
        )

        if use_phenotype:
            self.phenotype_encoder = PhenotypeEncoder(
                phenotype_input_dim,
                phenotype_hidden,
                phenotype_dim,
                dropout,
            )
            self.alignment = ModalityAlignment(
                phenotype_dim,
                roi_dim,
                common_dim,
                adversary_hidden,
            )
            self.condition_fusion = nn.Sequential(
                nn.Linear(2 * common_dim, common_dim),
                nn.LayerNorm(common_dim),
                nn.LeakyReLU(0.1),
            )

        if use_qc_adversary:
            self.qc_adversary = QCAdversary(
                common_dim,
                adversary_hidden,
                qc_dim,
            )

        self.bec_generator = DirectedBECGenerator(
            num_rois,
            roi_dim,
            common_dim,
            edge_dim,
            max_incoming_edges,
        )

    def forward(
        self,
        context,
        phenotype=None,
        modality_grl_strength=1.0,
        qc_grl_strength=1.0,
    ):
        roi_features = self.temporal_encoder(context)
        fmri_global = roi_features.mean(dim=1)

        modality_logits = None
        modality_targets = None

        if self.use_phenotype:
            if phenotype is None:
                raise ValueError("Phenotype is required.")
            pheno = self.phenotype_encoder(phenotype)
            pheno, fmri, modality_logits, modality_targets = (
                self.alignment(
                    pheno,
                    fmri_global,
                    modality_grl_strength,
                )
            )
            condition = self.condition_fusion(
                torch.cat([pheno, fmri], dim=-1)
            )
        else:
            fmri = F.normalize(
                self.fmri_projection(fmri_global),
                p=2,
                dim=-1,
            )
            condition = fmri

        qc_prediction = None
        if self.use_qc_adversary:
            qc_prediction = self.qc_adversary(
                fmri,
                qc_grl_strength,
            )

        internal_bec, self_coeff, threshold = self.bec_generator(
            roi_features,
            condition,
        )
        return {
            "internal_bec": internal_bec,
            "self_coeff": self_coeff,
            "threshold": threshold,
            "modality_logits": modality_logits,
            "modality_targets": modality_targets,
            "qc_prediction": qc_prediction,
        }

    @staticmethod
    def predict_next(previous, internal_bec, self_coeff):
        cross = torch.einsum(
            "bij,bjt->bit",
            internal_bec,
            previous,
        )
        return cross + self_coeff.unsqueeze(-1) * previous
