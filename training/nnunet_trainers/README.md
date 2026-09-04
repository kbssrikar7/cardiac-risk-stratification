# nnU-Net custom trainers (documentation copies)

nnU-Net discovers trainer classes by name within its own installed package
(`nnunetv2.training.nnUNetTrainer`), not via any project-side import path -
there's no supported way to point `-tr` at an external file. The files here
are kept for reproducibility/documentation only; to actually use one, copy
it into the installed package:

```bash
cp training/nnunet_trainers/nnUNetTrainer_cpu_smoke.py \
   .venv/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/training_length/
```

See `training/TECHNICAL_REPORT.md`'s nnU-Net section for why this trainer
exists (a CPU-only default run's Epoch 0 took 76.6 minutes - ~64 hours for a
full reduced run) and why its results are a pipeline-correctness check only,
not a real benchmark.
