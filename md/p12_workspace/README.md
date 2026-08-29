# Practice 12 작업 기록 — Object Detection and Segmentation

> **2차 개정으로 이 회차는 세 개로 쪼개졌다** (Practice12 추론 / Practice13 fine-tuning / HW PCB).
> 현재 상태의 기준은 `06_restructure_record.md` 이고, 그 진단은 `05_revision_plan.md` 에 있다.
> **아래 표의 `01_spec` ~ `04_consistency` 는 개정 전 35셀 노트북의 기록이다** — 지금 노트북과 대조하지 말 것.

하네스가 `_workspace/` 에 만든 산출물을 과목 레이아웃에 맞춰 `md/` 아래로 옮긴 것이다.
`CLAUDE.md` "디렉터리 레이아웃" 이 루트에는 `Practice*.ipynb` 와 `CLAUDE.md` 만 두고
작업 문서는 `md/` 에 두도록 정하고 있으며, 그 문서가 하네스 기본값보다 우선한다.

**이 회차는 강의 슬라이드가 없다.** `lecture_notes/` 에 detection/segmentation 자료가 없고
`md/practice_outline_ref.md` 에도 해당 항목이 없다. 그래서 명세서가 이 노트북의 유일한
재생성 근거이며, generator script 는 검증 후 규약대로 삭제했다.

| 파일 | 내용 |
|------|------|
| `00_env_probe.md` | ultralytics 경로 오염 6종과 해법, 학습 시간, API 속성 실측 |
| `01_spec_Practice12.md` | 셀 단위 명세 35셀. **향후 슬라이드 제작 시 대조 기준** |
| `02_author_report.md` | 저작 4라운드 이력, 명세 이탈 전수, 실측 수치 |
| `03_audit_Practice12.md` | 컨벤션 감사 (최종 PASS, NOTE 4건 미반영) |
| `04_consistency_Practice12.md` | 정합성 검증 (최종 PASS, **미검증 3건**) |

## 다음 개정 때 먼저 볼 것

- **강의자료 축은 미검증이다.** 이 주제의 슬라이드가 만들어지면 `04_` 의 축 B(ultralytics 공식 문서 대조)가
  축 A′(강의자료 대조)로 승격되므로 그 시점에 재검증이 필요하다
- **Roboflow `.download()` 왕복은 미검증이다.** API 키가 있는 환경에서 셀 31 을 한 번 돌리는 것 외에
  해소 방법이 없다. `model_format='yolov8'` 은 SDK 소스로 확정한 값이다
- `00_env_probe.md` 의 경로 정책은 ultralytics 를 쓰는 다른 회차에도 그대로 적용된다
