# Team dataset classification

- Gate D 결과: **실패 — NetLLM processed dataset incomplete**
- archive source: `/root/NetLLM/data.zip`
- canonical path: `/root/NetLLM-assets/datasets/team_data`
- extraction path safety: pass
- staging→canonical manifest checksum match: yes

## 전체 manifest

- file count: `48,957`
- total uncompressed size: `3,456,022,732 bytes`
- manifest:
  `/root/NetLLM-assets/manifests/dataset_manifest.sha256`
- manifest line count: `48,957`
- manifest size: `5,462,073 bytes`
- manifest SHA-256:
  `4cbc567ebc3783102c996b46fffe965b617815a6a9e487e0a33a59aa4fa17399`

Extension별 count:

| 확장자 | 파일 수 |
|---|---:|
| `.jpg` | 45,750 |
| `.csv` | 3,140 |
| `.bin` | 27 |
| `.json` | 13 |
| `.log` | 12 |
| `.md` | 14 |
| extension 없음 (`.DS_Store`) | 1 |

## 데이터 유형

Archive에는 다음이 있다.

- raw/extracted Jin2022 JPG frames
- Jin2022/Wu2017 cooked viewport CSV
- 기존 GPT-2 fine-tuned result/checkpoint
- regression/track/velocity model/result

MP4는 없다. JPG frame은 존재하지만 upstream이 직접 기대하는 processed
`Jin2022images/saliencyMap` 또는 precomputed `features` 구조는 없다.

```text
saliencyMap directories=0
features directories=0
.pth feature files=0
```

따라서 raw JPG를 saliency map 또는 feature tensor로 간주하지 않는다.

## Jin2022 image coverage

`images/Jin2022_images/video1_images`부터 `video27_images`까지 directory coverage는
27/27이다. 각 directory의 JPG 이름은 `1.jpg`부터 해당 max까지 연속이며 내부 gap은 없다.

대부분의 30-second video는 1,800장, video10–17은 1,500장이다. 다음 세 directory는
source의 nominal frame-count contract보다 짧다.

```text
video9_images:  1..1755  (nominal 1800)
video18_images: 1..1451  (nominal 1500)
video27_images: 1..1744  (nominal 1800)
```

전체 JPG count는 `45,750`, total JPG size는 `2,311,248,146 bytes`다.

## Cooked viewport

Jin2022 cooked CSV:

- file count: `2,268`
- total size: `27,396,178 bytes`
- structure: `viewports/Jin2022/video1..video27/5Hz/*.csv`

Cooked CSV는 실제 sample의 coordinate input으로 사용할 수 있지만 multimodal checkpoint가
필요로 하는 processed image/features를 대체하지 않는다.

## Completeness 판정

누락:

- `Jin2022images/saliencyMap/video1_images..video27_images`
- `Jin2022images/features/video1_images..video27_images`
- precomputed feature `.pth`
- raw JPG→saliency/features 전처리 config 및 checksum provenance

Checkpoint의 `using_multimodal` 값도 불명확하다. 따라서 현재 data asset을 NetLLM Llama
checkpoint의 완전한 evaluation dataset으로 확정할 수 없으며 **incomplete**로 판정한다.

Gate D 실패에 따라 Gate E base download, Gate F environment, Gate G smoke는 진행하지 않는다.
