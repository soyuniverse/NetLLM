# NetLLM 원본 무결성

## 1. 식별 정보

| 항목 | 확인 결과 |
|---|---|
| 원본 repository path | upstream 기준 `.` / `/workspace/NetLLM-source` |
| remote URL | `https://github.com/duowuyms/NetLLM.git` |
| current commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` |
| commit subject | `finish cjs` |
| HEAD 상태 | detached HEAD (`HEAD (no branch)`) |
| 포함 branch | local `master`, `origin/master` |
| working tree | clean |
| modified file | 없음 |
| untracked file | 없음 |
| staged file | 없음 |
| Git object check | `git fsck --no-progress --no-dangling` 이상 없음 |

## 2. upstream 대비 상태

- local `origin/master`는 current commit과 동일했다.
- `git rev-list --left-right --count origin/master...HEAD` 결과는 `0 0`이었다.
- 원격을 변경하지 않는 `git ls-remote https://github.com/duowuyms/NetLLM.git refs/heads/master`로 확인한 remote master도 같은 commit이었다.
- 따라서 감사 시점의 원본은 remote master와 commit 및 tracked content가 일치한다.
- `git fetch`, `git checkout`, `git reset`, `git clean`은 이번 조사에서 실행하지 않았다.

## 3. 프로젝트 repository와의 관계

- 프로젝트 기준 `.` / `/workspace/NetLLM`과 upstream 기준 `.` / `/workspace/NetLLM-source`는 각각 독립된 `.git`을 가진 별도 repository다.
- 프로젝트 remote는 `https://github.com/soyuniverse/NetLLM.git`, 원본 remote는 `https://github.com/duowuyms/NetLLM.git`이다.
- 프로젝트 repository의 시작 상태에는 사용자 변경이 이미 존재했다.
  - `docs/MEETING_NOTES.md`: staged와 unstaged 변경이 동시에 존재
  - `docs/liteVLM.pdf`: untracked
  - `docs/논문리딩4NetLLM.pdf`: untracked
- 위 사용자 변경은 수정, stage, stash, 삭제하지 않았다.

## 4. 무결성 결론

`/workspace/NetLLM-source`는 현재 clean하며 실제 remote master와 일치한다. Phase 0 동안 원본 source, data, Git metadata를 변경하는 명령은 실행하지 않았다.

다만 현재 프로젝트의 실행 script는 그대로 실행할 경우 원본 아래에 output을 생성한다.

- 프로젝트 기준 `scripts/run_vp_regression_cpu.sh`는 upstream 기준 `viewport_prediction/logs/` 및 `viewport_prediction/data/results/`를 생성할 수 있다.
- 프로젝트 기준 `scripts/run_vp_gpt2_adapt_e1.sh`는 upstream 기준 `viewport_prediction/logs/`, `viewport_prediction/data/ft_plms/`, checkpoint를 생성할 수 있다.

따라서 이 script들을 현재 형태로 실행하는 것은 “원본 디렉터리를 read-only upstream으로 취급”한다는 연구 원칙과 충돌한다.

## 5. 향후 원본 보호 방법

1. 모든 실행 전후에 다음 두 명령이 빈 결과인지 확인한다.

   ```bash
   git -C /workspace/NetLLM-source status --porcelain=v2 --untracked-files=all
   git -C /workspace/NetLLM-source diff --name-status HEAD
   ```

2. baseline/PLM의 `models_dir`, `results_dir`, log를 프로젝트 기준 `experiments/vp/` / `/workspace/NetLLM/experiments/vp` 아래로 redirect하는 외부 runner만 사용한다.
3. 원본 `config.py`, entry point, model source에는 patch를 적용하지 않는다.
4. 현재 `setup.sh`는 원본에서 `git fetch`와 `git checkout`을 수행하므로, 원본에 변경이 감지되면 재실행하지 말고 먼저 사용자 검토를 받는다.
5. checkpoint, model weight, dataset 및 대용량 output은 프로젝트 Git에 추가하지 않는다.

