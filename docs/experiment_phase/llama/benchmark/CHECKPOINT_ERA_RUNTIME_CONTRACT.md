# Checkpoint-Era Runtime Tensor Contract

The non-invasive selector insertion point is:

```text
history [1,10,3]
-> original Conv1d/Linear temporal embedding [1,10,4096]
-> original LayerNorm
-> selector once
-> 20-step old Llama autoregressive loop
```

Each Llama forward invokes the unchanged checkpoint-era `task_head`. Its
`[1,1,3]` result is embedded by the original Conv1d/Linear path and appended as
one feedback token. Feedback is not selected and is not LayerNorm-normalized.
The final feedback embedding is computed but not consumed, preserving source
behavior.

Attention masks contain ones and match the selected/current sequence length.
No `past_key_values` is passed back, so cache is never reused. Batch size one is
required by the original `.view(1,256)` embedding contract.

The exact evaluation loss is `torch.nn.MSELoss` on normalized coordinates.
Normalization divisors are `(180,90,180)`; reporting metrics use denormalized
angles. Upstream RMSE ignores its rotation argument, so a separate corrected
rotation-aware RMSE is retained.
