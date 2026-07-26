# Task-Head Source Archaeology

## Result

An exact checkpoint-specific source was found in the authoritative upstream:

- Repository: `https://github.com/duowuyms/NetLLM.git`
- Immutable commit: `ee4d8726898610e4ae7df08bdd26728cafb4701f`
- Commit date: 2024-07-30 23:01:55 +08:00
- Detached audit clone: `/root/NetLLM-source-checkpoint-era`
- Clone status/diff: clean / empty

No fetch or checkout was performed in `/root/NetLLM-source`.

## Search evidence

The project and upstream local branches, remote-tracking branches, tags,
reflogs, and all reachable commits were searched with pickaxe and Git grep.
Checkpoint/archive metadata, setup/manual documents, and training references
were also inspected. The uploaded archive contains weights and PEFT metadata but
no source.

The selected commit's `viewport_prediction/README.md` explicitly names the
published `try_llama2_7b` checkpoint, says it must use the old implementation,
and gives a `run_old.py` command for it.

## Source contract

The checkpoint-era implementation establishes the intended naming:

- `models/old/llama.py`: `task_head` is documented as the networking head.
- `models/old/networking_head.py`: `SimpleLinearTaskHead` is
  `Linear(4096, 3) -> Tanh` for Llama2-7B.
- `models/old/pipeline.py`: `modules_except_plm` index 4 is
  `self.plm.task_head`.
- `run_old.py`: LoRA is loaded with `load_adapter`; `modules_except_plm.bin`
  is loaded through the default strict `load_state_dict`.
- Dataset arguments: Jin2022, history 10, future 20, 5 Hz default, sample step
  15 default, test split from `config.py`.
- Published checkpoint command: Llama base, rank 32, scheduled sampling,
  batch size 1, 40 epochs, learning rate 0.0002, without `--multimodal`.

The old implementation's relevant Git blobs are identical between the selected
historical commit and the pinned current upstream commit, where they remain
under `models/old` and `run_old.py`.

## Package evidence

The checkpoint-era README specifies Python 3.8.10, torch 2.1.0,
NumPy 1.24.4, Munch 4.0.0, Transformers 4.34.1, and PEFT 0.6.2.
The supplied adapter model card reports PEFT 0.6.0. This version discrepancy is
carried into the environment plan rather than silently resolved here.

Machine-readable candidates:
`experiments/vp/llama_source_recovery/source_candidates.json`.
