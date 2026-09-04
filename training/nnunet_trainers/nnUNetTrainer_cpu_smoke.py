import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_cpu_smoke(nnUNetTrainer):
    """Not a real nnU-Net benchmark - a drastically cut-down run (few
    iterations/epochs) added specifically because the default config's
    Epoch 0 took 76.6 minutes on a CPU-only machine (~64 hours for a
    reduced 50-epoch run), which is impractical here. This exists only to
    confirm the pipeline produces a plausible result end-to-end; its Dice
    numbers should NOT be compared against this project's own properly-
    trained models or against nnU-Net's own published results - it is
    heavily undertrained by design. See TECHNICAL_REPORT.md's nnU-Net
    section for the honest framing."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
