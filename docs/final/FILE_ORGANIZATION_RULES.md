# Repository File Organization Rules

## 1. Canonical ownership

- Final manuals and cross-phase conclusions belong under `docs/final/`.
- Implementation explanations belong under `docs/implementation/`.
- Phase-specific evidence remains under `docs/experiment_phase/`.
- Executable experiment/setup files belong under `scripts/experiment_phase/`.
- Runtime results remain under `experiments/` and are never mixed with source.
- Machine-readable inventories and checksums belong under `manifests/`.
- Models, checkpoints, datasets, ZIP files, and environments stay outside the
  repository workflow and are never committed.

## 2. Immutability

Once a runtime, result, log, CSV, JSON, PNG, or phase report is referenced by a
document or manifest, its path is treated as immutable. It may be reorganized
only with an explicit reference-migration task.

File organization never changes file contents, benchmark values, code logic,
or external assets.

## 3. Move policy

A regular file may move only when all of the following hold:

1. Its purpose and canonical owner are clear.
2. The destination does not exist.
3. SHA-256 is recorded before the move.
4. SHA-256 is recorded after the move and is identical.
5. Existing references either remain valid or receive a compatibility symlink.
6. No duplicate regular-file copy is created.

Compatibility symlinks are preferred over duplicate copies when content edits
are prohibited.

## 4. Naming

- Primary files use stable descriptive names.
- Distinct uploaded or historical variants go under a `legacy/` directory.
- Zero-byte or suspicious variants are preserved under `legacy/`; they are not
  silently deleted.
- Generic runtime names such as `run_status.txt` are allowed only inside an
  isolated phase/configuration directory.

## 5. Git safety

- ZIP, model, checkpoint, adapter, and dataset paths must remain ignored and
  unstaged.
- Existing staged user work is preserved.
- Source-tree cache files may be removed only when they are newly generated and
  their scope is exact.
- External upstream trees and asset roots are never moved by repository
  organization tasks.

## 6. Stable historical artifacts

Historical Phase 0–4 reports and runtimes remain at their recorded paths when
existing manifests reference them. A cleaner-looking tree is not worth
invalidating reproducibility evidence.

