# Checkpoint Recovery Start State

## Audit policy

This recovery reuses the completed forensic, compatibility, and dataset audits.
Their tensor inventories and dataset comparisons were read, not recomputed.

## Existing result checksums

| Existing result | SHA-256 |
| --- | --- |
| `CHECKPOINT_FORENSIC_AUDIT.md` | `d2013ed14b089e02c3e18d77ee6b861ef5f5cc2e8695394d5efeb9a678116100` |
| `CHECKPOINT_UPSTREAM_COMPATIBILITY.md` | `a1c5aed089a702f791d94e1dcf6ca2655f34f34c97cb8e937cb203955b988f18` |
| `TEAM_UPSTREAM_DATASET_COMPARISON.md` | `b831c637f45549f5e5703a69ab3526744d314bf81bdab9d7db7e2d08b8c03756` |
| `compatibility_matrix.json` | `b138a5889d4b9ff51c2f11b8d2b8392ce1054368620d5a545f0429fd39385f68` |
| `adapter_key_manifest.json` | `e135b4fa23bb9459017aa863030f8205079796cf00593ff6d56b710ced75228e` |
| `modules_except_plm_key_manifest.json` | `22531a6db30a87bb3960f8632ecc27a930624682e5ee1255c72ce83be986b24f` |
| `REMAINING_TEAM_INPUT_REQUIREMENTS.md` | `954883ea81399c3882999890da25e86321e69648a4eaf57cbf45126dde48a6e3` |

## Fixed facts carried forward

- Current upstream: `/root/NetLLM-source`
- Current upstream commit:
  `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- Current upstream tracked status/diff: clean / empty
- Existing compatibility classification: `D. structurally-incompatible`
- Known mismatch: exactly the checkpoint-era prediction-head name requires
  source archaeology and full-key revalidation.
- Cooked Jin2022 relationship: all 2,268 uploaded files are byte-identical to
  their upstream Git-tracked counterparts.

The project working tree already contained staged and untracked user artifacts.
They are preserved and excluded from the recovery's mutation scope.
