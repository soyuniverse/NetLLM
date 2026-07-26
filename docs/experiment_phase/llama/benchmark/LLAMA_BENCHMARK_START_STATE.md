# Llama Selector Benchmark Start State

- Current upstream: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`, clean
- Checkpoint-era source: `ee4d8726898610e4ae7df08bdd26728cafb4701f`, clean
- Strict load: success; missing/unexpected keys `0/0`
- Technical smoke: success; prediction `[1,20,3]`, finite
- `using_multimodal=False`
- Random VP component: none
- Forward contract: 20 calls, lengths 10 through 29, cache not reused

Existing strict-load and technical-smoke runtimes were read and checksummed,
not rerun. Their result SHA-256 values at benchmark start were:

- strict load:
  `c9edebb9c7b991fc772e679b57cae5ec0884e25592413774ffd537d5c3d4f658`
- successful technical smoke:
  `4cad5e5942f3b21647b857b846db5577b49293c75252d6d2f9be8d97358e93d7`

This phase is a **recovered-artifact controlled comparison**. It is not an
official NetLLM benchmark or a paper reproduction.
