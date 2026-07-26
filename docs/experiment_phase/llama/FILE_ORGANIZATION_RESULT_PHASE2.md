# File Organization Result — Compatibility Phase

## Outcome

All new files were created directly in the requested canonical directories.
No existing file was moved, copied, deleted, or overwritten. No asset content
was modified.

## Organization checks

- Generated-file inventory: complete
- Phase/category classification: complete
- Duplicate basename scan within compatibility/data outputs: none
- Hard-coded `/workspace` or `/venv/` scan in new phase outputs: none
- JSON syntax validation: passed for all four runtime JSON files
- Broken source imports: not applicable; Gate 5 source code was not created
- Empty canonical directories created by this task: none
- New `__pycache__`/`.pyc`: none
- Temporary files or figures created by this task: none
- Files moved: none; move checksum comparison was not applicable

## Gates intentionally left without artifacts

Gate 4–7 artifacts were not created. Gate 2 selected
`D. structurally-incompatible`, and the request prohibits proceeding after a
failed gate. In particular:

- `/root/venvs/vp_netllm_llama` was not created.
- No external key loader was written because Gate 5 is authorized only for
  classification B.
- No model/checkpoint load was attempted.
- No VP sample inference was attempted.

## Integrity snapshot

| Item | Result |
| --- | --- |
| Upstream commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` |
| Upstream status/diff | clean / empty diff |
| Upstream `__pycache__` or `.pyc` | 0 |
| GPT-2 pip-freeze SHA-256 | `731a5031a3fb94909d541db8b66a41299d0977716e09d26eaa682ec3154d0311` |
| GPT-2 artifact fingerprint | `f3fdcf85dd2a8d38b329048ebb0349bcc94e1c6a04aa08ec20a4c0334ed74f14` |
| Llama base checksum-manifest SHA-256 | `28afd48051cd8293a5744eba52d7b21955b273ceadf9209a64a6af227597d3a3` |
| Checkpoint manifest SHA-256 | `44cbaaa6a174207bd98c21030200ad4244f09b5273a3ec0355ece0830519c1c6` |
| Dataset manifest SHA-256 | `4cbc567ebc3783102c996b46fffe965b617815a6a9e487e0a33a59aa4fa17399` |
| `try_llama2_7b.zip` SHA-256 | `57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53` |
| `data.zip` SHA-256 | `9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d` |
| ZIPs staged | no |
| ZIPs locally ignored | yes, both |

The GPT-2 artifact value uses the existing Phase 3A fingerprint definition;
the GPT-2 environment and artifact were not changed or exercised.

## Existing user work

Pre-existing staged setup/manual files remain staged and untouched. Pre-existing
Llama setup/audit artifacts remain in place and were neither moved nor edited.
