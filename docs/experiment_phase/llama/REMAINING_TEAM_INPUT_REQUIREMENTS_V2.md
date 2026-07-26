# Remaining Team Input Requirements V2

Source archaeology resolved the model implementation, task-head semantics,
published command, dataset split, and non-multimodal mode. Do not request those
again.

The remaining inputs needed for a quality/full benchmark are:

1. The immutable Hugging Face revision or checksum of the exact Llama2-7B base
   used to train this adapter. `adapter_config.json` contains only the local
   path `/data/data1/wuduo/2023_prompt_learning/downloaded_plms/llama/base`.
2. The checkpoint selection provenance: selected epoch/step, whether this is
   `best_model` or a periodic checkpoint, and the validation criterion.
3. For byte-level published-checkpoint provenance, the official
   `try_llama2_7b` archive SHA-256 or team confirmation that the uploaded
   archive is the file linked by the upstream README.
4. For exact environment reproduction, the CUDA wheel/build paired with the
   README's torch 2.1.0 training environment.

None of these items blocks the validated technical inference or a controlled
latency-only benchmark. They do block quality metrics and a paper reproduction
claim.
