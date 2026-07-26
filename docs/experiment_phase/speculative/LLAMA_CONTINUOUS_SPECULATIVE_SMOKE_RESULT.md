# Continuous VP Draft-and-Verify Prototype Smoke Result

## Scope and status

- prototype: **Continuous VP Draft-and-Verify Prototype**
- status: success
- target: checkpoint-era fine-tuned Llama2 NetLLM
- draft: deterministic recent-velocity extrapolation
- dataset/sample: Jin2022 test index 0, video 4, user 83, timestep 30
- history/future/output: `[1,10,3]` / `[1,20,3]` / `[1,20,3]`
- output finite: true

This is continuous-coordinate draft-and-verify control flow. It is not discrete-token exact speculative decoding.

## Acceptance

- policy: consecutive prefix whose maximum absolute normalized coordinate error is at most the threshold
- threshold: 0.1
- accepted prefix length: 8
- first rejected index: 8
- behavior after rejection: use target coordinates from the first rejected position onward

## Forward count and latency

- draft forward count: 1
- target forward count: 20
- baseline target forward count: 20
- target-only baseline latency: 616.881 ms
- prototype latency: 629.045 ms
- baseline/prototype latency ratio: 0.9807
- peak allocated/reserved: 12996.031 / 13086.000 MiB

The current implementation obtains the complete 20-step target trajectory before verification. It therefore does not reduce target forwards and was not faster in this smoke.

## Validity

- `control_flow_valid=True`
- `speedup_claim_valid=False`
- `quality_claim_valid=False`
- `technical_smoke_only=True`
- `target_block_verification_optimized=False`

No learned draft, training, backward, optimizer, full speculative benchmark, or quality superiority claim is included.

