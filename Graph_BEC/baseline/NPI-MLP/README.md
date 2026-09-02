# NPI

[Mapping effective connectivity by virtually perturbing a surrogate brain](https://www.nature.com/articles/s41592-025-02654-x)

## **Abstract**

Effective connectivity (EC), which reflects the causal interactions between brain regions, is fundamental to understanding information processing in the brain. However, traditional methods for obtaining EC, which rely on neural responses to stimulation, are often invasive or limited in spatial coverage, making them unsuitable for whole-brain EC mapping in humans. To address this gap, we introduce Neural Perturbational Inference (NPI), a data-driven framework for mapping whole-brain EC. NPI employs an artificial neural network trained to model large-scale neural dynamics, serving as a computational surrogate of the brain. By systematically perturbing all regions in the surrogate brain and analyzing the resulting responses in other regions, NPI maps the directionality, strength, and excitatory/inhibitory properties of brain-wide EC. Validation of NPI on generative models with known ground-truth EC demonstrates its superiority over existing methods such as Granger causality and dynamic causal modeling. When applied to resting-state fMRI data across diverse datasets, NPI reveals consistent, structurally supported EC patterns. Furthermore, comparisons with cortico-cortical evoked potentials data show a strong resemblance between NPI-inferred EC and real stimulation propagation patterns. By transitioning from correlational to causal understandings of brain functionality, NPI marks a stride in decoding the brain's functional architecture and facilitating both neuroscience studies and clinical applications.

## **Introduction**

<img src=".\img\NPI_framework.jpg" alt="NPI_framework" style="zoom:80%;" />

This repository contains the code and documentation of the NPI framework, a tool designed for mapping whole-brain EC in the brain. NPI operates by first training an ANN to mimic the complex dynamics of the brain based on the observed neural data. Once trained, virtual perturbations are applied to specific regions of the ANN to simulate the effect of neural stimulation. By monitoring the responses in other brain regions, NPI constructs a comprehensive map of the causal relationships, detailing how each area influences others across the brain. This process allows for the assessment of not only the presence but also the direction and excitatory/inhibitory properties of neural connections, contributing to a deeper comprehension of the brain's functional architecture.

## **Methodology**

<img src=".\img\dynamics.gif" alt="dynamics" style="zoom:100%;" />

1. **Utilize an ANN to serve as a surrogate brain.**

   NPI utilizes ANN to learn the brain’s complex, nonlinear dynamics directly from data. This approach allows NPI to adapt to a wide range of data types and dynamics. The use of advanced AI techniques, such as pre-training (to train a group-level surrogate model) and fine-tuning (to obtain individual-level surrogate models), further enhances our model’s applicability to both group-level and individual-level analyses.

2. **Apply virtual perturbation to ANN for inferring EC.**

   NPI provides flexibility in the pattern of perturbations once the surrogate model is well-trained. It is not constrained to a fixed-size perturbation and can accommodate various forms and scales of perturbations, tailored to specific research needs. This adaptability enhances NPI’s applicability across diverse experimental settings and research questions.

## **Requirements**

**Operating Systems**: Windows 10, Ubuntu 20.04

**Python Version**: 3.12+

**Dependencies**: See `requirements.txt` file for details.

**Non-standard Hardware Requirements**: No non-standard hardware is required to run this software.

**Note on PyTorch**: We recommend installing the GPU version of PyTorch, especially if you plan to simulate large datasets. Running NPI can be computationally intensive and may take an extended period when running on CPU.

## **Installation**

1. Clone the repository: `git clone https://github.com/ncclab-sustech/NPI`.
2. Navigate to the project directory: `cd NPI`.
3. Install dependencies: `pip install -r requirements.txt`.

Typical installation time is approximately 5 minutes on a "normal" desktop computer.

## **How to run**

The Graph-BEC baseline entry point is `run_npi_classifier.py`. It independently
fits one MLP surrogate brain to each subject's standardized AAL90 ROI time
series, computes one individual NPI-BEC matrix, and evaluates it with the
Graph-BEC classification protocol.

The saved BEC convention is explicit:

```text
bec[source, target] = average target-output change after perturbing source
```

The convention follows `model_EC` in `NPI.py`: the perturbed node indexes the
row and the model-output response indexes the column. No transpose is applied
before saving.

Example generation and classification command:

```bash
python Graph_BEC/baseline/NPI/run_npi_classifier.py \
  --dataset abide \
  --gpu-id 0
```

Use `--generation-only` to create only the individual NPI-BEC archive, or
`--classification-only` to classify an existing archive.

This baseline intentionally uses subject-wise surrogate fitting. It does not
perform pooled group-surrogate pretraining or group-level fine-tuning.

## **How to cite**

If you use NPI in your research, please cite our paper:

- Luo, Z., Peng, K., Liang, Z. *et al.* Mapping effective connectivity by virtually perturbing a surrogate brain. *Nat Methods* (2025). [https://doi.org/10.1038/s41592-025-02654-x](https://doi.org/10.1038/s41592-025-02654-x)

## **Contact**

For any questions/comments, please contact [NCC Lab](https://www.sustech.edu.cn/en/faculties/liuquanying.html).

## **Copyright**

Copyright © 2024 NCC Lab, Southern University of Science and Technology, Shenzhen, China.

## ABIDE-II

The Graph-BEC baseline runner also supports ABIDE-II:

```bash
python Graph_BEC/baseline/NPI-MLP/run_npi_classifier.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --gpu-id auto
```

It uses the ABIDE-II profile, reading
`dataset/ABIDE-II/Phenotypic_Processing.csv` and the standardized ROI files
under `dataset/ABIDE-II/cpac/filt_noglobal/`. Results are written to
`Graph_BEC/baseline/NPI-MLP/outputs/`, with the archive named
`subject_npi_mlp_bec_abide_ii.npz`.

For a smoke test:

```bash
python Graph_BEC/baseline/NPI-MLP/run_npi_classifier.py \
  --dataset abide_ii \
  --max-subjects 2 \
  --npi-epochs 1 \
  --npi-batch-size 16 \
  --generation-only
```
