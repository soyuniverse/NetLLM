# Remaining Team Input Requirements V3

The following provenance remains unresolved:

1. Immutable Llama2 base revision/checksum used during training.
2. Exact checkpoint epoch or training step represented by `try_llama2_7b`.
3. Validation metric and selection criterion used to choose the checkpoint.
4. Exact training PyTorch 2.1.0 CUDA build and complete training command/config.
5. Official checksum confirmation for the originally uploaded checkpoint/data archives.

These items do **not** block the recovered-artifact controlled comparison completed here. They block a defensible paper-reproduction claim and exact reconstruction of the original training run.

No additional dataset request is needed for this non-multimodal checkpoint: the 2,268 uploaded cooked Jin2022 CSV files were already confirmed checksum-identical to the Git-tracked upstream copies.

