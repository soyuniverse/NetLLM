# Remaining Team Input Requirements

## Blocking compatibility evidence

The supplied checkpoint stores the prediction head as
`4.task_head.0.{weight,bias}`, while the pinned upstream commit expects
`4.networking_head.0.{weight,bias}` and loads this state dict strictly. Before
load or inference can proceed, the team must provide one of:

1. The exact source commit or archived model implementation that generated the
   checkpoint and defines `task_head`; or
2. An authoritative checkpoint migration/key-mapping specification confirming
   that `task_head` and `networking_head` are semantically identical for this
   checkpoint.

This is a renamed-module mismatch, not a prefix-only mismatch. The current task
does not authorize an inferred rename.

## Unresolved run provenance

Please provide:

- The explicit `using_multimodal` value used for this run.
- The training command and resolved configuration.
- The checkpoint selection epoch/step and selection criterion.
- The source/upstream commit used during training.
- The exact Llama base repository revision used during training; the adapter
  records only a machine-local base path and no immutable revision.

## Conditional multimodal inputs

Only if `using_multimodal=True`, provide:

- The matching `saliencyMap` and precomputed `features` trees.
- The feature extractor name/version, configuration, and checksum.
- A checksum manifest and provenance for the image frames/features used by the
  checkpoint.

The current uploaded dataset contains no `saliencyMap` or precomputed feature
tree. Raw JPG frames must not be treated as a substitute.

## Already established; do not resend

No additional confirmation is needed for LoRA rank 32, alpha 32, dropout 0.05,
`q_proj`/`v_proj` targets, adapter tensor inventory, or the ten
`modules_except_plm.bin` tensors. These were verified directly.
