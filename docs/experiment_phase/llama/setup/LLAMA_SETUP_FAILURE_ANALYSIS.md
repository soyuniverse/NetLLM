# Llama setup failure 분석

- audit 시각: 2026-07-26 UTC
- 실패 log:
  `/root/NetLLM-assets/logs/setup_llama_20260726_101522.log`
- 실패 위치: `setup_netllm_llama.sh:249`
- Gate A 결과: 성공 — 원인 분류 완료

## 직접 확인한 오류

실패한 명령은 다음 `hf auth whoami` 호출이다.

```text
Cannot reach https://huggingface.co/api/whoami-v2:
offline mode is enabled.
```

Log에는 먼저 auth probe가 실패해 interactive login branch에 진입했고, CLI가
`User is already logged in`이라고 확인한 뒤 line 249의 두 번째 `auth whoami`에서 offline
오류로 종료된 흐름이 기록돼 있다. Token 값은 읽거나 기록하지 않았다.

## Offline 설정 출처 구분

### 1. 현재 shell에서 상속된 변수

Audit 시점에는 모두 unset이다.

```text
HF_HUB_OFFLINE=unset
TRANSFORMERS_OFFLINE=unset
HF_DATASETS_OFFLINE=unset
```

실패 프로세스 자체는 이미 종료되어 당시 environment를 다시 읽을 수 없다. 다만 HF CLI의
명시적 오류와 아래 세 검사를 함께 보면, 실패 실행을 시작한 shell/process에서
`HF_HUB_OFFLINE`이 일시적으로 상속됐다는 결론이다.

### 2. Startup file 설정

다음 파일에는 세 offline 변수 설정이 없다.

- `/root/.bashrc`
- `/root/.profile`
- `/etc/environment`

따라서 새 shell마다 재설정되는 persistent startup 원인은 아니다.

### 3. Setup script 설정

두 setup script 모두 auth/download 전에 offline 변수를 설정하지 않는다.

- `/root/NetLLM/setup_netllm_llama.sh`
- `/root/NetLLM/scripts/setup_netllm_llama.sh`

각 script의 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`은 마지막 local-only smoke
subprocess에만 붙으며 auth보다 뒤에 있다. 따라서 script 내부 설정이 line 249 실패를
일으킨 것은 아니다.

두 파일은 내용 checksum이 다르므로 동일 파일로 간주하지 않는다. 이번 작업에서는 어느
기존 script도 수정하지 않았다.

### 4. GPT-2 local-only 설정

Phase 2A/2B/3A script의 offline 설정은 GPT-2 artifact를 network fallback 없이 load하기 위한
command/process scoped 정책이다. 전역 startup 설정이 아니며 Llama auth 실패의 원인으로
오인하지 않는다.

## 확정 원인

분류: **실패 setup process가 상속한 transient offline environment**

근거:

1. 실패 log에서 HF CLI가 offline mode를 직접 보고했다.
2. setup script는 auth 전에 offline 변수를 설정하지 않는다.
3. startup file에도 설정이 없다.
4. 현재 shell에서는 세 변수가 unset이다.
5. 저장된 Hugging Face token 파일은 존재한다.

## 수정 정책

기존 setup script는 변경하지 않고 다음 wrapper를 신규 생성했다.

```text
scripts/experiment_phase/llama/setup/
  setup_llama_online_download_offline_run.sh
```

Wrapper는 setup process 전체에서 다음 변수를 명시적으로 제거한다.

```text
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
HF_DATASETS_OFFLINE
```

Setup script가 model download를 완료한 뒤 수행하는 local-only smoke subprocess는 기존
script가 자체적으로 offline 변수를 다시 지정하므로 정책이 유지된다.
