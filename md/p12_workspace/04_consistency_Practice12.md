# Consistency Verification — Practice12_Object_Detection_and_Segmentation.ipynb (라운드 6, 최종)

## 판정: **PASS**

- **치명 0 / 경미 2 (신규) / 미검증 4** — 라운드 5 의 치명 V-11 이 **해소**됐다
- 재검증 범위는 리더가 지정한 코드 5셀(19·20·22·23·29)이지만, **축 C 는 노트북 전체의 수치·방향 주장을 다시 돌렸다** — 수치가 미세하게 움직였으므로 변경 셀만 봐서는 새 모순을 잡을 수 없다
- 셀 번호는 **0-based**. 35셀, 에러 0 / 미실행 0 / `execute_result` 잔존 0 / `execution_count` [1..20] 연속

### 미검증 — 최종 보고에 반드시 포함할 것 (4건, 변동 없음)

| # | 항목 | 사유 | 성격 |
|---|---|---|---|
| 1 | **축 A′ 강의자료 ↔ 노트북 (전 항목)** | **대조할 원본이 존재하지 않는다.** `lecture_notes/` 에 `Ch1-ML 1_Linear Regression.pdf` 하나뿐이고, `md/` 두 문서는 학부 참조본으로 detection/segmentation/YOLO 키워드 0건 | **이 회차 최대 잔여 위험** |
| 2 | **Roboflow `.download()` 왕복** | `ROBOFLOW_API_KEY` 없음 → 셀 31 은 가드 경로만 실행 | 키 있는 환경에서 1회 실행으로만 닫힘 |
| 3 | **셀 2·6 의 미실행 분기** | 패키지·데이터가 이미 있어 최초 실행 경로가 이번에도 안 돌았다 | Colab 첫 실행 전용 경로 |
| 4 | **CPU 환경에서의 학습 시간** | GPU(TITAN RTX) 실측 **detection 0.019h(약 68초) / segmentation 0.023h(약 83초)**. `patience=0` 으로 120 epoch 전부 돌아 라운드 5(101 epoch, 0.008h/0.009h) 대비 **약 2.4배 늘었다.** 프로필 `cell_time_warning_sec: 300` 대비 GPU 는 여유가 있으나 **CPU 런타임에서는 측정 불가** | **이번 수정으로 위험이 커진 항목** |

---

## 라운드 6 재검증 — V-11 해소 확인

| 확인 | 라운드 5 (결함) | 라운드 6 (현재) | 판정 |
|---|---|---|---|
| 학습 종료 줄 (셀 19·29) | `101 epochs completed` ×2 | **`120 epochs completed`** ×2 (0.019h / 0.023h) | **해소** |
| `EarlyStopping` 문구 | detection·segmentation 각 1건 | **0건** | **해소** |
| 인자 로그 `patience` | `100` (기본값, 미지정) | **`0`** (명시) | **해소** |
| 마크다운·라벨의 `120` | 실제 101 과 충돌 | **실제 120 과 일치** | **해소** |
| 소스 내 `101` 문자열 | — | **0건** | — |
| `optimizer:` 실적용 줄 | `SGD(lr=0.01, momentum=0.937)` | **동일** | 유지 |

**저자의 파라미터화도 확인했다.** 셀 19 가 `finetune_epochs = 120` 을 정의하고 셀 20·22·23·29 의 라벨이 f-string 으로 그 값을 읽는다 (`정의 1 / 사용 8`). **코드 라벨은 이제 epoch 수를 하드코딩하지 않는다** — 다만 마크다운은 그렇지 않다 (V-14).

---

## 축 B 신규 확인 — `patience=0` 의 기전 (리더 지시)

**리더는 로그(관측)로만 확인했다. 기전을 소스로 보고 관측과 맞는지까지 확인하라는 지시였다. 수행했다.**

| 단계 | 소스 | 내용 |
|---|---|---|
| 1. 생성 | `engine/trainer.py:412` | `self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False` |
| 2. **치환** | `utils/torch_utils.py:1003` | `self.patience = patience or float("inf")` |
| 3. 호출 | `engine/trainer.py:605` | `self.stop \|= self.stopper(epoch + 1, self.fitness) or final_epoch` — **매 epoch 호출된다** |
| 4. 판정 | `utils/torch_utils.py:1024` | `stop = delta >= self.patience` |
| 기본값 | `cfg/default.yaml:14` | `patience: 100 # early stop after N epochs without val improvement` |

**기전:** `patience=0` 은 **"early stopping 을 끄는 전용 분기" 가 아니다.** `0` 이 Python 에서 falsy 라 `0 or float("inf")` 가 `inf` 로 평가되고(직접 실행 확인), 그 결과 `stop = delta >= inf` 가 **어떤 epoch 에서도 참이 될 수 없다**. stopper 객체는 계속 살아 있고 매 epoch 호출되지만 결코 `True` 를 반환하지 않는다.

**기전 ↔ 관측 대조:**

| 기전이 예측하는 것 | 실제 관측 | 일치 |
|---|---|---|
| `stop` 이 한 번도 참이 되지 않음 → 조기 종료 로그가 찍히지 않음 | `EarlyStopping` 문구 **0건** | ✓ |
| 요청한 epoch 전부 실행 | `120 epochs completed` | ✓ |
| **`best.pt` 선택은 별개 경로**(`trainer.py:766` `save_model` 의 `best_fitness` 비교)라 영향받지 않음 | `B best checkpoint` 행이 **0.68769 / 0.41834 로 라운드 5 와 완전히 동일** | ✓ |
| `last.pt` 만 epoch 101 → 120 으로 바뀜 | detection `0.36912/0.22711` → **`0.36998/0.23863`**, segmentation box `0.32123` → **`0.32036`**, mask `0.25789` → **`0.25711`** | ✓ |

**네 예측이 모두 관측과 맞는다.** 특히 `best.pt` 행이 한 자리도 안 움직인 것이 기전 확인의 결정적 증거다 — early stopping 과 checkpoint selection 이 서로 다른 코드 경로임을 보여 준다.

> **부수 발견 (노트북 무관, 기록용):** 치환이 falsy 검사이므로 `patience=None`·`patience=False` 도 같은 효과를 낸다. 반대로 **"1 epoch 만에 멈춘다" 를 `patience=0` 으로 표현할 수단은 없다.** 공식 문서의 안내 문구(`use patience=0 to disable EarlyStopping`)와 소스가 일치한다.

---

## 축 C 재검증 — 수치·방향 주장 전수 재대조

**현재 출력 (셀 20·29 print 블록 실물):**

```
                        mAP@50    mAP@50-95          box mAP@50  box mAP@50-95  mask mAP@50  mask mAP@50-95
A pretrained           0.68437    0.39986      after 120 epochs     0.32036       0.22290     0.25711        0.16023
B after 120 epochs     0.36998    0.23863      best checkpoint      0.76877       0.48625     0.73535        0.40272
B best checkpoint      0.68769    0.41834
```

| # | 해설 문장 (셀) | mAP@50 | mAP@50-95 | 판정 |
|---|---|---|---|---|
| C-1 | 셀 21 `it keeps about half of the pretrained **mAP@50**` | 0.36998/0.68437 = **0.5406** ✓ | (열 한정으로 해당 없음) | **일치** — 문장이 `mAP@50` 으로 **열을 명시**해 한정했다. 이 회차에서 이 한정이 유일하게 제 역할을 하는 자리다 |
| C-2 | 셀 32 `halved the score on the images it had not seen` (**열 한정 없음**) | 0.5406 (46% 손실) ✓ | 0.23863/0.39986 = **0.5968** (40% 손실) **△** | **V-13 — 경미** |
| C-3 | 셀 32 `The higher fine-tuned score came from a checkpoint picked on the split being reported` | 0.68769 > 0.68437 ✓ | 0.41834 > 0.39986 ✓ | **일치** — 양쪽 열 모두 참 |
| C-4 | 셀 21 `` `last.pt` is the weights after all 120 epochs `` | 실제 120 epoch 실행 ✓ | — | **일치** (라운드 5 에서는 거짓이었다) |
| C-5 | 셀 17 `+ 120 epochs on coco8`, `best of 120, chosen on images/val` | 실제 120 epoch, best 는 그중 선택 ✓ | — | **일치** |
| C-6 | 셀 17 `coco8 has no test split` / `The second row is not controlled.` / `Only the last-epoch row escapes that selection.` | 변동 없음 ✓ | — | **일치** |
| C-7 | 셀 28 `both the last epoch and the chosen checkpoint are shown` | 셀 29 2행 출력 ✓ | — | **일치** |
| C-8 | 셀 24 3문장 | 아래 표 | 아래 표 | **일치** |

### 셀 24 ↔ 셀 22·23 재대조 (샘플 예측이 바뀌었다)

| 확인 | 라운드 5 | **라운드 6** | 셀 24 문장 | 판정 |
|---|---|---|---|---|
| train `000000000009.jpg` pretrained | orange **0개** | orange **0개** (변동 없음) | — | — |
| 〃 fine-tuned | orange **5개** | **orange 6개** (`bowl, broccoli, bowl, bowl, orange ×6`) | `The training image gains the oranges the pretrained weights were missing.` | **일치** (0 → 6) |
| held-out `000000000049.jpg` pretrained | 8건 (`horse ×2, person ×6`) | **8건 (변동 없음)** | — | — |
| 〃 fine-tuned | `zebra` 1건 | **`zebra` 1건 (변동 없음)** | `The held-out image loses almost everything those same pretrained weights found.` | **일치** (8 → 1) |
| — | — | — | `That is overfitting, and the last.pt row above is its number.` (리더 허용) | **일치** |
| — | — | — | `Eight photographs are what this figure is about, not what fine-tuning does in general.` | **일치** |

- **훈련 이미지의 orange 가 5 → 6 으로 늘었다** (GT 는 4개). 셀 24 는 개수를 주장하지 않으므로 문장은 그대로 참이다. 다만 **개수를 언급하는 문장을 나중에 추가하면 즉시 깨진다** — 다음 개정자에게 남긴다
- 셀 22 는 여전히 `training image` / `held-out image` 를 파일명과 함께 출력하고, 셀 23 네 패널 제목이 그 구분을 유지한다

### 인과 단정 재확인 (소스 전수)

`due to` 0 / `caused by` 0 / `therefore` 0 / `hence` 0 / `leads to` 0 / **`mosaic` 0** / **`warmup` 0**. `because` 2건은 기존 오탐 2건(셀 10 검증된 인과, 셀 23 서술 의도) 그대로. `overfit` 1건은 리더 허용. **`patience` 3건은 전부 코드·주석**(셀 19 주석 1, 인자 2)이며 인과 주장이 아니다.

### 기계 검사

`markdown_budget.py` → **0 over cap**, soft 초과 6건(라운드 5 와 동일 구성). `audit_notebook.py` → **후보 없음** (16종 체크포인트).

---

## 불일치 상세 (신규 2건, 둘 다 경미)

### V-13. Summary 의 `halved` 는 열을 한정하지 않았는데, 이번 수정이 두 번째 열을 더 멀어지게 했다 — 경미

- **노트북:** 셀 32 — `Fine-tuning on four images fitted those four and **halved the score** on the images it had not seen.`
- **실제 출력 (셀 20):**

  | 열 | A pretrained | B after 120 epochs | 남은 비율 | 손실 |
  |---|---|---|---|---|
  | mAP@50 | 0.68437 | 0.36998 | **54.06%** | 45.94% |
  | mAP@50-95 | 0.39986 | 0.23863 | **59.68%** | **40.32%** |

- **무엇이 다른가:** **값(비율).** `halved` 는 열 한정이 없으므로 두 열 모두에서 참이어야 한다. `mAP@50` 은 54.06% 로 "반토막" 이 타당하지만, `mAP@50-95` 는 **59.68% 가 남아 손실이 40%** 다. 40% 손실을 "halved" 라고 부르는 것은 약 10%p 과장이다.
- **이 수정이 상황을 악화시켰다 (리더가 우려한 그대로):**

  | 열 | 라운드 5 남은 비율 | 라운드 6 남은 비율 | 방향 |
  |---|---|---|---|
  | mAP@50 | 53.94% | 54.06% | 거의 불변 |
  | mAP@50-95 | 56.80% | **59.68%** | **"half" 에서 2.88%p 더 멀어짐** |

- **학생에게 미치는 영향:** 치명은 아니다. **두 열 모두 실제로 크게 떨어졌고 방향도 같으므로 학생이 얻는 결론("fine-tuning 이 일반화를 해쳤다")은 옳다.** 문제는 정도의 과장이며, 바로 위 셀 21 이 `mAP@50` 으로 열을 한정해 정확히 쓴 것과 대비되어 **같은 노트북 안에서 기준이 두 개**가 된다.
- **수치 확인:** 수행함 (위 두 표).
- **수정 방향:** 셀 21 의 방식을 그대로 따라 **열을 한정**한다 — `halved the score on the images it had not seen` → `` kept about half of its `mAP@50` on the images it had not seen ``. 분량 영향 없음(셀 32 는 산문 61/80). 열을 한정하기 싫다면 `lost 40 to 46 percent of both scores` 처럼 범위로 쓰면 양쪽 열에서 참이 된다.

### V-14. 코드 라벨은 파라미터화됐는데 마크다운은 `120` 하드코딩이라, Exercise 2 를 풀면 둘이 갈라진다 — 경미

- **노트북 (파라미터화된 쪽):** 셀 19 `finetune_epochs = 120` + 주석 `# every epoch count printed below reads this one value`. 셀 20·22·23·29 라벨이 f-string 으로 이 값을 읽는다 (사용 8곳).
- **노트북 (하드코딩으로 남은 쪽) — 전부 마크다운:**
  - 셀 17 표 — `**+ 120 epochs on coco8 (SGD, lr0=0.01)**`, `**best of 120, chosen on images/val**`
  - 셀 21 — `` `last.pt` is the weights after all 120 epochs ``, `those same 120-epoch weights`
- **셀 33 Exercise 2:** `Change `epochs` in the detection fine-tuning call from 120 to 30.`
- **무엇이 다른가:** 학생이 지시대로 값을 30 으로 바꾸면 **셀 20·22·23 라벨은 `30` 으로 따라가고 셀 17·21 마크다운만 `120` 으로 남는다.** `auditor2` 가 라운드 5 에 올린 지적이 **코드 쪽만 해소되고 마크다운 쪽은 남았다.**
- **부수 문제:** Exercise 2 는 `` `epochs` `` 를 바꾸라고 하는데, 실제 호출은 `epochs=finetune_epochs` 이므로 **학생이 바꿔야 하는 것은 `finetune_epochs`** 다. 셀 19 첫 줄 주석이 안내하고 있어 못 찾을 정도는 아니다.
- **학생에게 미치는 영향:** 기본 상태(120)에서는 **모든 문장이 참이므로 결함이 발현되지 않는다.** 연습문제를 푸는 학생에게만 나타난다. 다만 이 회차가 반복해서 잡아 온 결함이 정확히 "본문과 출력이 어긋난다" 이므로, 학생이 스스로 그 상태를 만들게 두는 것은 일관되지 않는다.
- **수치 확인:** 해당 없음 (구조적 확인). 마크다운 `120` 잔존 5곳을 소스 파싱으로 전수 확인했다 (셀 17 ×2, 셀 21 ×2, 셀 33 ×1 — 셀 33 은 지시문이므로 정상).
- **수정 방향 (택일, 저자 재량):**
  1. **셀 17·21 마크다운을 epoch 비의존으로** — `+ 120 epochs` → `` fine-tuned on coco8 (SGD, `lr0=0.01`) ``, `best of 120` → `best epoch, chosen on images/val`, `after all 120 epochs` → `after the last epoch`. **셀 17 은 116/120 이라 단어를 늘리면 안 된다** — 위 치환은 전부 단어 수가 줄거나 같다
  2. Exercise 2 문구를 `` Change `finetune_epochs` from 120 to 30 `` 로 바꿔 대상만 정확히 가리킨다 (마크다운 불일치는 남음)
  - **1 을 권고한다.** 2 는 대상만 고칠 뿐 갈라짐 자체는 그대로다

---

## V-12 (`best.pt` = epoch 1) 미반영 — **리더 판단에 동의한다. 이의 없음**

리더 결정: *"사실이지만 epoch 1~11 지표가 동일해 노트북 출력만으로 epoch 1 을 특정할 수 없다."*

**동의하며, 한 가지를 덧붙여 기록한다 — 이 수정으로 내 근거 자체가 소멸했다.**

라운드 5 에서 내가 V-12 의 근거로 삼은 것은 셀 19 로그의 `Best results observed at epoch 1, best model saved as best.pt.` 한 줄이었다. **그 줄은 `EarlyStopping` 이 발동할 때만 출력된다.** `patience=0` 으로 조기 종료가 사라지면서 **그 줄도 함께 사라졌다** (현재 `EarlyStopping` 문구 0건). 즉 지금 노트북 출력에는 best 가 몇 번째 epoch 인지 알려 주는 정보가 **어디에도 없다.**

따라서 V-12 를 본문에 적으면 **출력이 뒷받침하지 못하는 주장**이 되며, 이는 이 회차 내내 걷어낸 결함과 정확히 같은 종류다. **리더 판단이 옳고, 미반영이 맞다.** 최종 보고에 미반영으로 명시한다.

> **다만 사실 자체는 명세에 남겨야 한다.** 노트북에 쓰지 않는 것과 명세가 모르는 것은 다르다. `01_spec_Practice12.md` 에 "`best.pt` 는 epoch 1~11 동률 중 tie-break 로 뽑히며, 학습이 출발점을 한 번도 넘지 못한다 — **따라서 `best` 행을 '학습 성과' 로 서술하지 마라**" 를 남기지 않으면, 다음 개정자가 `best of 120` 을 성능 향상으로 해석하는 문장을 새로 쓴다. **노트북 금지 / 명세 존치** 로 갈라 두는 것이 이 항목의 올바른 처리다 (아래 S-13).

---

## 명세 정정 목록 — `spec` 작업 지시 (라운드 5 목록 유지 + 이번 수정분 3건 추가)

라운드 5 의 **S-1 ~ S-12 는 그대로 유효하다** (본문 아래 라운드 5 절 참조). 이번 수정으로 추가되는 항목:

| # | 명세 위치 | 반영할 내용 |
|---|---|---|
| **S-13** | §3.7 (또는 새 절) | **`patience=0` 을 넘긴다.** 넘기지 않으면 ultralytics 기본값 `patience=100` 이 개입해 **101 epoch 에서 조기 종료**되고, 노트북의 `epochs=120` 서술이 전부 거짓이 된다 (라운드 5 치명 V-11). 기전: `utils/torch_utils.py:1003` 의 `patience or float("inf")` — falsy 치환이므로 `0` 이 "무한" 을 뜻한다. **`best.pt` 선택은 별개 경로라 영향받지 않는다.** 함께: **`best.pt` 는 epoch 1~11 동률의 tie-break 이고 학습이 출발점을 한 번도 넘지 못하므로, `best` 행을 "학습 성과" 로 서술하지 마라** (V-12 — 노트북에는 쓰지 않되 명세에는 남긴다) |
| **S-14** | §4 셀 19 / §5 변수표 | **`finetune_epochs = 120` 단일 정의**를 명세화한다. 셀 20·22·23·29 의 모든 epoch 라벨이 이 값을 읽는다(사용 8곳). 변수표에 `finetune_epochs`, `last_epoch_label`, `finetuned_label`, `segmentation_last_label` 추가. **다음 개정자가 "중복" 으로 보고 리터럴로 되돌리지 못하게 이유를 함께 적을 것** |
| **S-15** | §3.6 학습 시간 | 실측 갱신: **detection 0.019h(약 68초) / segmentation 0.023h(약 83초)** — GPU, 120 epoch 전량. 라운드 5(101 epoch) 대비 약 2.4배. 프로필 `cell_time_warning_sec: 300` 대비 GPU 여유 있음. **CPU 런타임 미측정을 미검증 항목으로 명시할 것** |

**추가로 §3.7·§4.2 의 실측 숫자를 라운드 6 값으로 교체해야 한다** (S-4·S-5 의 대상 값이 또 바뀌었다):

| arm | mAP@50 | mAP@50-95 |
|---|---|---|
| A pretrained | 0.68437 | 0.39986 |
| B after 120 epochs (`last.pt`) | **0.36998** | **0.23863** |
| B best checkpoint (`best.pt`) | 0.68769 | 0.41834 |

| segmentation arm | box mAP@50 | box mAP@50-95 | mask mAP@50 | mask mAP@50-95 |
|---|---|---|---|---|
| after 120 epochs (`last.pt`) | **0.32036** | **0.22290** | **0.25711** | **0.16023** |
| best checkpoint (`best.pt`) | 0.76877 | 0.48625 | 0.73535 | 0.40272 |

---

## 미검증 (라운드 6)

| 항목 | 사유 |
|------|------|
| **축 A′ — 강의자료 ↔ 노트북 전 항목** | **대조할 원본이 존재하지 않는다.** 다른 과목 자료로 대체하지 않았다 |
| **Roboflow `.download()` 왕복** | `ROBOFLOW_API_KEY` 없음 → 가드 경로만 실행 |
| **셀 2·6 의 미실행 분기** | 최초 실행 경로가 이 환경에서 돌지 않았다 |
| **CPU 환경 실행 시간** | GPU 0.019h/0.023h. `patience=0` 으로 약 2.4배 늘어 **위험이 커졌으나** CPU 측정 불가 |
| 그림 렌더링(PNG 육안) | **검증 범위 밖 — 리더 소관.** 단 셀 22 의 훈련 이미지 예측이 orange 5 → **6개**로 바뀌었으므로 **셀 23 오른쪽 위 패널이 이전 육안 확인본과 다르다.** 리더 재확인 권고 |

---

## 상위로 올릴 항목

### 리더에게

1. **판정 PASS.** 치명 0. V-11 해소를 로그·소스 양쪽으로 확인했다.
2. **`patience=0` 기전 확인 완료.** 전용 분기가 아니라 `patience or float("inf")` 의 falsy 치환이며, `best.pt` 행이 한 자리도 안 움직인 것이 기전과 관측의 일치를 증명한다.
3. **신규 경미 2건.** V-13(Summary `halved` 가 열 한정 없음 — 이번 수정이 `mAP@50-95` 를 59.68% 로 밀어 더 멀어졌다), V-14(마크다운 `120` 하드코딩 잔존 → Exercise 2 와 갈라짐). **둘 다 반영 여부는 리더 판단**이며, 반영한다면 셀 32 한 문장 + 셀 17·21 치환으로 끝난다 (분량 증가 없음).
4. **V-12 미반영 동의.** 근거였던 로그 줄이 이 수정으로 사라졌으므로 리더 판단이 옳다. **단 명세에는 남길 것** (S-13).
5. **그림 재확인 1건.** 셀 22 훈련 이미지 orange 5 → 6 이라 셀 23 오른쪽 위 패널이 이전 육안 확인본과 다르다.

### `lecture-spec-analyst` 에게

라운드 5 의 **S-1 ~ S-12** + 이번 **S-13·S-14·S-15** + 실측 숫자 교체.

---

## 하네스 갱신 제출 (라운드 6 추가)

| # | 트리거 | 제출 내용 | 대상 |
|---|---|---|---|
| H-5 | **이번 수정이 다른 셀의 주장을 약화시켰다** | **"수치를 바꾸는 수정 뒤에는 변경 셀이 아니라 그 수치를 인용하는 모든 셀을 다시 대조한다."** V-13 은 셀 19 수정(`patience=0`) 때문에 **손대지 않은 셀 32** 의 주장이 약해진 사례다. 재검증 범위를 "변경된 셀" 로 잡으면 구조적으로 못 잡는다 | `lecture-notebook-consistency` 스킬 보강 |
| H-6 | **부분 파라미터화가 새 불일치면을 만든다** | **"라벨을 변수로 뽑을 때는 같은 값을 말하는 마크다운도 함께 처리한다. 코드만 파라미터화하면 연습문제가 둘을 갈라놓는다."** V-14 | `CLAUDE.md` 또는 저작 스킬 |
| H-7 | **기전 확인과 원인 확인의 분리 (리더 지시가 규율로 정착)** | **"라이브러리 인자의 효과는 로그(관측)만이 아니라 소스(기전)로 확인하고, 기전이 예측하는 부수 효과가 관측과 맞는지까지 본다."** 이번에 `best.pt` 행 불변이 기전 확인의 결정적 증거가 됐다 | `lecture-notebook-consistency` 스킬 |

라운드 5 제출분(H-1 ~ H-4)은 그대로 유효하다. 특히 **H-1**(라이브러리가 인자 값을 바꾼다)은 이번에 `patience=0` 으로 해소된 것이 아니라 **회피된 것**이므로, 규약에는 여전히 필요하다.

**강의자료 자체의 오류로 보고할 항목: 없다** (대조할 자료가 존재하지 않는다).

---
---

# 이전 라운드 이력 (라운드 5) — **개정 전 / patience 미지정** 기준

> 최상단 라운드 6 판정이 최신이고 우선한다. 아래 V-11 은 **해소됐다.**

## 라운드 5 본문

### 판정: **FAIL**

- **불일치 2건 (치명 1 / 경미 1) / 미검증 4건**
- `auditor2` 도 독립적으로 같은 자리에 도달해 **PASS → FAIL 로 정정**했다 (자기 FIX 를 BLOCK 으로 승격). 양측 판정 일치. 저자 발신 창구는 **V-11 하나**로 합의됨
- 감사자가 보낸 논거 1건은 **채택하되 표현 하나를 반증해 정정**했다 (V-12 "인용 시 주의")
- 치명 1건은 **축 C** — 노트북이 8곳에서 `120 epochs` 라고 단정하는데, **같은 셀의 출력 로그가 101 epochs 라고 적혀 있다.** `EarlyStopping(patience=100)` 이 detection·segmentation 양쪽에서 발동했다
- 셀 번호는 **0-based**
- 축 A′(강의자료 ↔ 노트북)는 이번에도 **미검증 — 대조할 원본 없음.** 아래 미검증 표를 반드시 함께 읽을 것

#### 미검증 — 최종 보고에 반드시 포함할 것

| # | 항목 | 사유 | 성격 |
|---|---|---|---|
| 1 | **축 A′ 강의자료 ↔ 노트북 (전 항목)** | **대조할 원본이 존재하지 않는다.** `lecture_notes/` 에 `Ch1-ML 1_Linear Regression.pdf` 하나뿐이고, `md/` 두 문서는 학부 참조본으로 detection/segmentation/YOLO/mAP/IoU 키워드 **0건** (grep 재확인) | **이 회차 최대 잔여 위험** |
| 2 | **Roboflow `.download()` 왕복** | `ROBOFLOW_API_KEY` 없음 → 셀 31 은 가드 경로만 실행됨 (`ROBOFLOW_API_KEY is not set...`). `model_format='yolov8'` 과 `Dataset.location` 은 설치본 SDK 소스로만 확인 | 키 있는 환경에서 1회 실행으로만 닫힘 |
| 3 | **셀 2·6 의 미실행 분기** | 패키지·데이터가 이미 있어 `except ImportError` → `pip install` 경로와 다운로드 경로가 이번에도 안 돌았다 | Colab 첫 실행에서만 도는 경로 |
| 4 | **CPU 환경에서의 학습 시간 (신규)** | 이번 실측은 GPU(TITAN RTX)에서 101 epoch = **0.008시간(약 29초)**. `epochs` 가 20 → 120 으로 6배가 됐으므로 프로필 `cell_time_warning_sec: 300` 대비 여유를 **CPU 런타임에서 재확인해야 한다.** 이 환경에서는 측정할 수 없다 | 신규 — 개정으로 생긴 항목 |

> **강의자료 축을 다른 과목 자료로 대체하지 않았다.** 과목마다 논리 전개와 scope 가 달라, 남의 자료로 판정하면 이 과목에서 옳은 것이 틀린 것으로 잡힌다.

#### 검증 축

| 축 | 원본(왼쪽) | 수행 여부 |
|---|---|---|
| A′. 강의자료 ↔ 노트북 | `lecture_notes/*.pdf`, `md/lectures_and_formulas.md`, `md/practice_outline_ref.md` | **미검증 — 대조할 원본 없음** |
| A. 명세 ↔ 노트북 | `md/p12_workspace/01_spec_Practice12.md` | 수행 — **단, 명세가 개정 전 상태다.** 아래 "명세 정정 목록" 참조 |
| B. 외부 사실 ↔ 노트북 | `https://docs.ultralytics.com/` (WebFetch 직접 확인) + 설치본 소스 `ultralytics 8.4.120` | 수행 |
| C. 마크다운 단정문 ↔ 같은/다음 셀의 실제 출력 | 실행된 노트북의 셀 출력 (35셀, 에러 0 / 미실행 0 / `execution_count` [1..20] 연속) | 수행 |

---

### 대조표 — 축 B (외부 사실 정확성)

이번 개정으로 새로 생긴 주장을 중점 확인했다. **전부 공식 문서 WebFetch 또는 설치본 소스로 직접 확인했고, 기억으로 판정한 항목은 없다.**

| # | 노트북 주장 (셀) | 권위 있는 근거 | 판정 |
|---|---|---|---|
| B-1 | 셀 17 `+ 120 epochs on coco8 (SGD, lr0=0.01)`, 셀 19·29 `optimizer='SGD', lr0=0.01` 이 **실제로 적용되는가** | **셀 19 출력 실물:** `optimizer: SGD(lr=0.01, momentum=0.937) with parameter groups ...` / 셀 29 도 동일. `engine/trainer.py:1114-1120` — `name == "auto"` 일 때만 `"ignoring 'lr0=...' and 'momentum=...'"` 를 찍고 `AdamW`/`MuSGD` 로 덮어씀. `SGD` 명시 시 그 분기를 타지 않음 | **일치** — `optimizer=auto` 함정을 정확히 피했다 |
| B-2 | 셀 20 주석 `train() leaves the best checkpoint loaded, so the weights after all 120 epochs are opened separately` | `engine/model.py` `YOLO.train()` 말미: `ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last` → `self.model, self.ckpt = load_checkpoint(ckpt)`. **경험적 재확인:** 셀 19 학습 종료 시 `Validating .../best.pt` = `0.671 0.667 0.688 0.418`, 셀 20 의 `finetune_model.val()` = **동일한 `0.688/0.418`** → `finetune_model` 이 `best.pt` 를 들고 있음이 출력으로 증명됨 | **일치** (단 문장 안의 `120` 은 V-11) |
| B-3 | `.box.map50` / `.box.map` / `.box.mp` / `.box.mr` / `.seg.map50` / `.seg.map` 대응 | **같은 셀 로그와 자릿수 대조.** 셀 18 로그 `all 4 17 0.644 0.567 0.684 0.4` ↔ print `precision 0.64370 / recall 0.56667 / mAP@50 0.68437 / mAP@50-95 0.39986`. 셀 29 seg 로그 `Box(P R mAP50 mAP50-95) 0.733 0.35 0.321 0.224` + `Mask(...) 0.734 0.25 0.258 0.16` ↔ print `0.32123 / 0.22372 / 0.25789 / 0.16033` | **일치** — 4/4 열 모두 대응 |
| B-4 | 셀 17 `coco8` has no test split | 로컬 실측: `data/coco8/images/` = `train`, `val` 뿐. 공식 문서 `https://docs.ultralytics.com/datasets/segment/coco8-seg/` 의 YAML 인용 `test: # test images (optional)` — **키는 있으나 값이 비어 있다.** 패키지 내장 `cfg/datasets/coco8.yaml` 도 동일 | **일치** |
| B-5 | 셀 8 라벨 포맷 `class cx cy w h` 정규화 / `class x1 y1 ... xn yn`, `at least three vertices` | `https://docs.ultralytics.com/datasets/segment/` — `<class-index> <x1> <y1> ... <xn> <yn>`, "normalized polygon coordinates (values are in `[0,1]` relative to image width and height)", **"a minimum of 3 (x, y) points"** | **일치** — 삭제·재번호로 훼손되지 않았다 |
| B-6 | 셀 5 `data.yaml` 키표 (`path`/`train`/`val`/`names`, `0..N-1` no gaps) | 위 문서: 필수 키 `path`/`train`/`val`/`names`, `test` 는 optional. `0..N-1` 연속 요구는 리더가 `KeyError` 로 재현 (명세 §3.3) | **일치** |
| B-7 | 셀 25 `masks.data` = binary grid sized by **the model input**, `masks.xy` = 원본 픽셀 polygon | `https://docs.ultralytics.com/modes/predict/` — `data`: "torch.uint8 binary mask tensor with shape (N,H,W)", 추론 해상도 공간. `xy`: "mask polygons in pixel coordinates" (원본 크기). **출력이 본문을 증명:** 셀 26 `masks.data (7, 448, 640)` vs `orig_shape (1380, 2070)` | **일치** |
| B-8 | 셀 15 `boxes.xywhn : normalized, same format as the label file` | 위 문서 — `xywhn`: "boxes in xywh format normalized by original image size". 라벨 = `class cx cy w h` 정규화 | **일치** |
| B-9 | 셀 13 `conf=0.25` discards detections the model is less than 25 percent confident about | `predict` 인자 `conf` 정의와 일치. 셀 15 출력 최저 conf = 0.251 (0.25 초과) | **일치** |
| B-10 | 셀 0·6 "eight photographs each, four for training and four for validation" | 셀 6 출력 `coco8 : 4 train / 4 val`, `coco8-seg : 4 train / 4 val` + 공식 문서 "4 train / 4 val" | **일치** |

#### 축 B 신규 확인 — 삭제로 인해 **새로 위험해진** 자리

| 확인 | 결과 |
|---|---|
| Step 4 삭제로 mAP 정의가 노트북에 없다 → **틀린 mAP 주장이 남아 있는가** | **없다.** 소스 전수 확인 결과 노트북에 남은 mAP 언급은 (a) print 라벨 `mAP@50` / `mAP@50-95` / 열 머리글, (b) 셀 21 `it keeps about half of the pretrained mAP@50` 뿐이다. **정의를 시도하는 문장이 0건**이므로 라운드 1 의 V-2(`m` 을 IoU 평균으로 오설명)가 재발할 자리 자체가 사라졌다. **사용자 결정대로 결함 아님** |
| 인과 단정 / 없는 근거 | **소스 기준** `due to` 0 / `caused by` 0 / `therefore` 0 / `hence` 0 / `leads to` 0 / **`mosaic` 0** / **`warmup` 0**. `because` **2건** — 셀 10 `names must cover 0 .. 79 ... because the labels use COCO indices` (리더가 `KeyError` 로 재현한 **검증된 인과**, 오탐 처리) 와 셀 23 `# class name only, because this figure asks what was found and not how sure the model is` (**저자의 서술 의도 설명이지 현상의 인과 주장이 아님**, 오탐 처리) |
| `overfitting` (리더 허용) 의 범위를 넘는 주장 | **없다.** 셀 24 는 `zebra` 가 왜 나오는지 설명하지 않고, 학습률·epoch 수·augmentation 을 원인으로 지목하지 않는다. `Eight photographs are what this figure is about, not what fine-tuning does in general.` 로 일반화까지 차단했다 |

---

### 대조표 — 축 C (마크다운 단정문 ↔ 실제 출력) — **이번의 핵심**

리더 실측표와 노트북 출력을 자릿수까지 대조했다.

| arm | 리더 실측 mAP@50 | 노트북 셀 20 출력 | 리더 실측 mAP@50-95 | 노트북 셀 20 출력 |
|---|---|---|---|---|
| A pretrained | 0.68437 | `0.68437` | 0.39986 | `0.39986` |
| B after 120 epochs (`last.pt`) | 0.36912 | `0.36912` | 0.22711 | `0.22711` |
| B best checkpoint (`best.pt`) | 0.68769 | `0.68769` | 0.41834 | `0.41834` |

**6/6 자릿수 일치.**

#### 두 열의 방향이 다른 문제 — 이번 라운드는 통과했다

| # | 해설 문장 (셀) | 대조 대상 | mAP@50 | mAP@50-95 | 판정 |
|---|---|---|---|---|---|
| C-1 | 셀 21 `Only last.pt is scored without any selection, and it keeps about half of the pretrained mAP@50.` | 셀 20 | 0.36912/0.68437 = **0.539** ✓ | 0.22711/0.39986 = **0.568** ✓ | **일치** — 문장이 `mAP@50` 으로 **열을 한정**했고, 한정하지 않았어도 양쪽 열 모두 참이다 |
| C-2 | 셀 32 `Fine-tuning on four images fitted those four and halved the score on the images it had not seen.` | 셀 20 `last.pt` 행 | 0.539 ✓ | 0.568 ✓ | **일치** — 양쪽 열 모두 참 |
| C-3 | 셀 32 `The higher fine-tuned score came from a checkpoint picked on the split being reported, so the last-epoch number is the honest one.` | 셀 20 `best.pt` 행 | 0.68769 > 0.68437 ✓ | 0.41834 > 0.39986 ✓ | **일치** — `best.pt` 가 **양쪽 열 모두 baseline 보다 높다**는 사실과 어긋나지 않는다. 라운드 4 의 V-10 과 이번 초안의 오류가 **재발하지 않았다** |

> **저자가 초안에서 만들었다가 스스로 잡았다고 보고한 문장**(`The validation rows above are where generalization is measured, and they went down.`)이 **최종본에 남아 있지 않음을 소스 전수로 확인했다.** `went down` / `validation rows` 문자열 0건. 저자 보고가 사실이다.

#### 셀 22·23 ↔ 리더 샘플 실측

| 확인 | 리더 실측 | 노트북 출력 (셀 22) | 판정 |
|---|---|---|---|
| train `000000000009.jpg` pretrained | GT 4/8, **orange 0개** | `bowl, broccoli, bowl, bowl, broccoli, fork, bowl` — **orange 0개** | 일치 |
| 〃 fine-tuned (`last.pt`) | GT 8/8, **orange 5개** | `bowl, broccoli, bowl, bowl, orange, orange, orange, orange, orange` — **orange 5개** | 일치 |
| held-out `000000000049.jpg` pretrained | GT 6/9 | `horse, horse, person ×6` (8건) | 일치 |
| 〃 fine-tuned | GT 0/9, **`zebra` 1개** | `zebra` — **1건** | 일치 |
| 셀 22 해설 ↔ 셀 24 | — | `The training image gains the oranges the pretrained weights were missing.` (0→5 ✓) / `The held-out image loses almost everything those same pretrained weights found.` (8→1 ✓) | 일치 |
| **학습 이미지와 held-out 이미지 구분** | — | 셀 22 print 가 `training image 000000000009.jpg` / `held-out image 000000000049.jpg` 로 **파일명까지 명시**. 셀 23 네 패널 제목이 `Pretrained - training image` / `After 120 epochs - training image` / `Pretrained - held-out image` / `After 120 epochs - held-out image` | **명확히 구분됨** (단 `120` 은 V-11) |

- 셀 22 는 `detection_model`(사전학습)과 `last_epoch_model`(`last.pt`)만 쓴다. `conf=0.25` 가 셀 14 의 사전학습 추론과 동일 → **비교가 통제되어 있다**
- 그림에 `best.pt` 를 쓰지 않은 것은 리더 지시이며, `best.pt` 가 보고 split 에서 선택된 것이라는 논점 오염을 피한다. **정당하다**

---

### 불일치 상세

#### V-11. 노트북이 8곳에서 `120 epochs` 라고 단정하는데 같은 셀의 로그는 `101 epochs` 다 — **치명**

- **노트북 (단정문):**
  - 셀 17 표 — `**+ 120 epochs on coco8 (SGD, lr0=0.01)**`, `**best of 120, chosen on images/val**`
  - 셀 20 주석 — `# train() leaves the best checkpoint loaded, so the weights after all 120 epochs are opened separately`
  - 셀 20 print — `B after 120 epochs`
  - 셀 21 MD — `` `last.pt` is the weights after all 120 epochs ``, `those same 120-epoch weights`
  - 셀 22 print — `  after 120 epochs : `
  - 셀 23 패널 제목 ×2 — `After 120 epochs - training image`, `After 120 epochs - held-out image`
  - 셀 29 print — `after 120 epochs`
- **실제 출력 (셀 19, 학습 로그 원문):**

  ```
  EarlyStopping: Training stopped early as no improvement observed in last 100 epochs.
                 Best results observed at epoch 1, best model saved as best.pt.
  To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300`
                 or use `patience=0` to disable EarlyStopping.

  101 epochs completed in 0.008 hours.
  ```

  **셀 29(segmentation)도 동일** — `101 epochs completed in 0.009 hours.`, `Best results observed at epoch 1`.
- **무엇이 다른가:** **값(epoch 수).** 학습은 120 이 아니라 **101 epoch 에서 멈췄다.** ultralytics 의 기본값 `patience=100` 이 인자 로그에 그대로 찍혀 있고(`patience=100`), 노트북은 이 인자를 넘기지 않았다. best 가 epoch 1 이므로 `1 + 100 = 101` 에서 조기 종료됐다.
- **학생에게 미치는 영향:** 셀 19 의 로그를 읽은 학생이 셀 20 의 `B after 120 epochs` 라벨을 본다. **같은 화면에서 101 과 120 이 충돌한다.** 학생이 내릴 결론은 둘 중 하나다 — "내가 로그를 잘못 읽었다" 또는 "이 노트북의 라벨을 믿을 수 없다". 이 노트북이 라운드 내내 지켜 온 것이 **"본문이 주장하는 것을 그 셀의 출력이 반박하지 않는다"** 인데(라운드 1 V-1 과 정확히 같은 결함 유형), 그 원칙이 개정으로 깨졌다.
  - 더 나쁜 것은 **이 숫자가 재현되지 않는다**는 점이다. `patience=100` 때문에 종료 epoch 은 best epoch 에 의존한다 — best 가 epoch 1 이면 101, best 가 epoch 25 면 120. 학생 환경에서 값이 달라져도 라벨은 `120` 으로 고정돼 있다.
- **수치 확인:** 수행함. `grep "epochs completed in"` → 2건 전부 `101 epochs`. `grep "EarlyStopping"` → detection·segmentation 각 1건. 인자 로그에 `patience=100`, `epochs=120` 병기 확인.
- **수정 방향 (노트북을 고친다):**
  1. **필수** — 하드코딩된 `120` 을 **epoch 수에 의존하지 않는 표현**으로 바꾼다. `B after 120 epochs` → `B last epoch`, `After 120 epochs - training image` → `After fine-tuning - training image`, `` the weights after all 120 epochs `` → `` the weights at the last epoch ``, `best of 120` → `best epoch, chosen on images/val`. 셀 17 표의 `+ 120 epochs on coco8` 은 **호출 인자**를 말하므로 `` fine-tuned on coco8 (SGD, `lr0=0.01`, `epochs=120`) `` 처럼 인자임이 드러나면 참으로 유지된다.
  2. **부수 효과 — 감사자 `auditor2` 의 유일한 FIX 가 같이 닫힌다.** 셀 33 Exercise 2 가 학생에게 `epochs` 를 30 으로 바꾸라고 시키는데, 라벨이 `120` 으로 박혀 있으면 학생이 다시 돌린 순간 라벨 전부가 거짓이 된다. epoch 비의존 표현이면 30 으로 바꿔도 라벨이 참으로 남는다. **한 번의 수정으로 두 지적이 모두 해소된다 — 저자에게는 단일 지시로 간다.**
  3. **리더 판단 필요 (택일):**
     - **(a) 라벨만 고친다.** 재실행은 필요하나 숫자는 안 바뀐다(같은 seed·같은 조기종료). 비용 최소. 조기 종료 사실을 한 줄로 드러낼지는 선택 — 드러낸다면 셀 19 로그가 이미 근거이므로 축 C 위반이 아니다.
     - **(b) `patience=0` 을 넘겨 실제로 120 epoch 을 돌린다.** 노트북이 자기 문장과 일치하게 되고 종료 epoch 의 비결정성이 사라진다. 대신 `last.pt` 가 epoch 120 의 가중치가 되어 **셀 20·22·23·29 의 모든 숫자와 샘플 그림이 다시 측정돼야 하고, 리더 기준표도 갱신해야 한다.**
     - 어느 쪽이든 **1번(epoch 비의존 라벨)은 그대로 필요하다** — (b) 를 골라도 Exercise 2 충돌은 남기 때문이다.

#### V-12. `best.pt` 가 **epoch 1** 의 가중치라는 사실이 노트북 어디에도 없다 — 경미

> **라운드 5 보강.** `auditor2` 가 이 항목의 논거를 강화해 보내왔고, **채택하되 표현 하나를 반증해 정정했다** (아래 "인용 시 주의"). 학습 로그 101 epoch 을 전수 파싱해 근거를 다시 세웠다.

- **노트북:** 셀 17 `**best of 120, chosen on images/val**` / 셀 21 `` `best.pt` is whichever epoch scored highest on the split reported here. ``
- **실제 출력 (셀 19·29):** `Best results observed at epoch **1**, best model saved as best.pt.`
- **무엇이 다른가:** 두 문장 다 **거짓은 아니다** — 기계적으로 `best.pt` 는 가장 높은 점수의 epoch 이 맞다. 문제는 **그 epoch 이 1 이라는 것**, 즉 `B best checkpoint` 행(0.68769 / 0.41834)이 **한 번의 iteration 직후 상태**라는 점이다 (train 4장 / `batch=4` → epoch 당 1 iteration, 로그의 진행 표시줄이 `1/1`). 학생은 `best of 120` 을 읽고 "120번 학습한 것 중 최고" 로 이해한다.

- **학습 로그 101 epoch 전수 파싱 (detection·segmentation 각각, 본 검증자 수행):**

  | 확인 | detection | segmentation |
  |---|---|---|
  | epoch 1 의 val | `mAP50 0.688 / mAP50-95 0.418` | `0.769 / 0.486` |
  | **최댓값에 도달한 epoch** | **1–11 (11개가 동률)** | **1–11 (11개가 동률)** |
  | **epoch 1 을 어느 한 열에서라도 넘어선 epoch** | **0개** | **0개** |
  | 나머지 epoch (양쪽 열 모두 epoch 1 미만) | **90 / 101** | **90 / 101** |
  | 마지막 epoch | `0.369 / 0.227` | `0.321 / 0.224` |

  → **학습은 단 한 번도 출발점을 넘어서지 못했다.** `best.pt` 가 epoch 1 인 것은 11개 동률의 tie-break 결과다 (`engine/trainer.py:872` — `self.best_fitness < fitness` 가 strict 이라 동률이면 앞선 epoch 이 유지된다). 즉 `best of 120` 은 "120개 후보 중 고른 것" 이 아니라 **"고를 것이 없어 맨 앞이 남은 것"** 이다. 이 구조가 detection·segmentation 에서 동일하게 나온다.

- **인용 시 주의 — `auditor2` 표현 1건을 반증했다.** 감사자는 `best.pt(0.68769) 가 baseline(0.68437) 과 사실상 같다` 고 썼다. **mAP@50 에서는 참이지만(+0.00332, 상대 +0.49%) mAP@50-95 에서는 아니다(+0.01848, 상대 +4.62%).** 이것이 이 노트북이 라운드 4 부터 반복해 온 **"한 열만 보고 쓴 문장"** 과 정확히 같은 형태이므로, 감사자 리포트의 이 문구를 **노트북 본문에 그대로 옮기면 안 된다.** 두 열 모두에서 안전하게 참인 서술은 위 표의 **"어떤 epoch 도 epoch 1 을 어느 열에서도 넘지 못했다"** 쪽이다.
- **노트북에 넣지 말 것:** `warmup_epochs=3.0` 이 인자 로그에 있어 초반 11 epoch 의 동률을 설명할 수 있으나, **그것은 로그가 보여 주는 사실이 아니라 추론이다.** 리더가 금지한 인과 단정(`warmup`)에 해당하므로 본문에 쓰지 마라. 이 문단은 검증자 판단 근거일 뿐이다.
- **개정 이전 명세가 정확히 이것을 경고하고 있었다.** `01_spec_Practice12.md` §3.7: *"로그상 `best.pt` 는 **epoch 1** 의 가중치이고 ... 이는 사실상 학습 전 상태다. **'20 epoch 학습한 결과' 라고 쓰면 거짓이 된다.**"* Step 4 삭제와 해설 전면 교체 과정에서 **이 가드 문장이 사라졌다.**
- **학생에게 미치는 영향 — 왜 치명으로 올리지 않았는가:** 셀 32 가 `The higher fine-tuned score came from a checkpoint picked on the split being reported, so the last-epoch number is the honest one.` 으로 그 행의 신뢰도를 이미 깎아 놓았다. **학생이 도달하는 결론("best.pt 행을 믿지 마라")은 옳다.** 틀리는 것은 그 **메커니즘**이다 — 학생은 "120개 중 운 좋은 것이 뽑혔다" 로 이해하지만 실제로는 "학습이 출발점을 한 번도 못 넘었다" 이다. 결론이 보존되므로 경미로 둔다. **등급을 부풀리면 이 하네스의 치명/경미 구분이 무의미해진다.**
- **수치 확인:** 수행함. 셀 19 학습 종료 시 `Validating .../best.pt` = `0.671 0.667 0.688 0.418`, 이는 epoch 1 의 val 값이며 셀 20 의 `B best checkpoint` 와 동일. 위 101 epoch 전수 파싱 표도 본 검증자가 직접 수행했다.
- **수정 방향:** V-11 과 **한 문장으로 함께 처리 가능**하다. 셀 21 의 `best.pt` 문장을 `` `best.pt` is the epoch that scored highest on the split reported here, and the log above names it. `` 류로 바꾸면 학생의 눈이 로그로 간다. **분량 여유:** 셀 21 은 현재 산문 61/80(soft 60 초과, hard 80 이내)이므로 한 문장 교체는 가능하나 **추가는 어렵다.** 셀 17 은 116/120 으로 여유 4단어뿐이니 셀 17 에 넣지 마라.

---

### 명세 정정 목록 — `spec` 작업 지시

**`md/p12_workspace/01_spec_Practice12.md` 는 개정 전 상태다. 아래는 전부 "명세가 틀렸고 노트북이 옳다" 는 판정이며, 노트북 결함이 아니다.** (저자 리포트 §6 의 자기 보고 9건을 독립 검증했고, 저자가 보고하지 않은 3건을 추가했다.)

| # | 명세 위치 | 현재 명세 | 실물(개정본) | 판정 | 저자 보고 |
|---|---|---|---|---|---|
| S-1 | §4 셀표 17–19 (Step 4 전체 `How Detection Is Scored`, IoU 손계산, IoU 그림) | 필수 3셀 | **삭제됨** | **노트북이 옳다** — 사용자 결정 (강의가 다룸) | 보고함 |
| S-2 | §2.1·§2.2·§2.3 (IoU / P·R / AP·mAP 수식) | *"두 수식은 셀 17 에 반드시 들어간다"* | **대응 셀 없음** | **노트북이 옳다.** 단 §2.3 의 **금지 서술**(`mAP averages precision over a range of IoU thresholds`)과 **§2.3 지표 접근 표**(`.box.map50`/`.box.map`/`.seg.map50`/`.seg.map`)는 **삭제하지 말고 유지하라** — 전자는 mAP 오설명 재발 방지 가드이고, 후자는 이번 라운드에 B-3 로 재검증된 유효 사실이다 | 보고함(§2 폐기로) — **일부 존치를 내가 추가 지시** |
| S-3 | §3.6 학습 설정 | `epochs=20`, `optimizer` 미지정(=auto), 실측 10.2초/9.2초 | `epochs=120`, `optimizer='SGD'`, `lr0=0.01`. **실측 101 epoch, 0.008h/0.009h (GPU)** | **노트북이 옳다** (리더 확정). 명세에 **`optimizer=auto` 는 `lr0` 를 무시한다**는 근거(`engine/trainer.py`)와 **`patience=100` 기본값으로 조기 종료된다**는 사실을 함께 박을 것 | 보고함 (조기 종료는 **미보고**) |
| S-4 | §3.7 실측 3-arm 표 | `0.63026 / 0.41646`, `0.68777 / 0.42254` | `0.36912 / 0.22711`, `0.68769 / 0.41834` | **노트북이 옳다** — 전면 교체 | 보고함 |
| S-5 | §3.7 segmentation 실측 표 | `0.56476 / 0.40104 / 0.55504 / 0.35476`, `0.76875 / 0.48625 / 0.73537 / 0.41137` | `0.32123 / 0.22372 / 0.25789 / 0.16033`, `0.76877 / 0.48625 / 0.73535 / 0.40272` | **노트북이 옳다** — 전면 교체 | 보고함 |
| S-6 | §3.7 "확정 4문장" | `one iteration per epoch at lr=0.000119`, `mAP50 holds two values` | 새 설정에서 **전부 거짓** (`SGD(lr=0.01)`) | **노트북이 옳다** — 폐기. **단 "`best.pt` 는 epoch 1 이므로 'N epoch 학습한 결과' 라고 쓰지 마라" 는 가드는 반드시 살려서 새 §3.7 에 옮길 것** (V-12) | 보고함 — **가드 존치는 내가 추가 지시** |
| S-7 | §4.2 통제표 9행 | `weights before evaluation` / `checkpoint evaluated` / dataset / split / imgsz / batch / seed / amp / model size | **6행** (`weights evaluated` / `checkpoint` / dataset / evaluation split / model size / imgsz) | **노트북이 옳다** — `batch`·`seed`·`amp` 는 셀 18·19·20 인자에서 **실제로 동일함을 확인했다**(전부 `batch=4, seed=42, amp=False`). 표에서 뺀 것은 분량(셀 17 이 116/120)이지 통제 실패가 아니다. `The second row is not controlled.` 는 유지됨 | 보고함 |
| S-8 | §4 Step 번호 / 셀 번호 | Step 0~8, 35셀 | **Step 0~7, 35셀** (4→7 재번호) | **노트북이 옳다** — 리더 승인. §4 표 전면 재작성 | 보고함 |
| S-9 | §5 변수표 | 8종 누락 | `heldout_image_path`, `heldout_label_path`, `heldout_image`, `pretrained_heldout_result`, `last_epoch_result`, `last_epoch_heldout_result`, `image_rows`, `panels` (+ 루프 지역 `ground_truth_names`/`pretrained_names`/`finetuned_names`/`panel_*`) | **노트북이 옳다** — 추가. `fill_colors` → **`instance_colors`** 개명도 반영할 것 (§5 표가 아직 `fill_colors` 다) | 보고함 (개명은 **미보고**) |
| S-10 | §5 / §3.5.1 | `annotated_image` 미사용, `data/sample.jpg` 필수 | 변동 없음 | **명세가 옳다** — 유지. 셀 26 이 `data/sample.jpg` 를 그대로 쓰고 `masks.data (7,448,640)` vs `orig_shape (1380,2070)` 로 §3.5.1 을 계속 증명한다 | — |
| S-11 | §6 미검증 3건 | 슬라이드 부재 / Roboflow / 셀 2·6 분기 | **여전히 유효 + 신규 1건** | **명세에 4번째를 추가하라** — "CPU 환경에서 `epochs=120` 의 셀 실행 시간 미측정" (프로필 `cell_time_warning_sec: 300`) | **미보고 — 내가 추가** |
| S-12 | §3.9 | `train()` 이 val split 에서 `best.pt` 를 고른다 | 변동 없음 | **명세가 옳다** — 유지. 이번에도 `engine/model.py`·`engine/trainer.py` 로 재확인했고 셀 19↔20 출력으로 경험적으로도 확인됐다 | — |

**S-2·S-6·S-9(개명)·S-11 은 저자가 보고하지 않은 항목이다.** 특히 **S-6 의 가드 문장 존치**가 중요하다 — 명세를 그냥 "폐기" 로 처리하면 다음 개정자가 `best.pt` 를 다시 "N epoch 학습 결과" 로 부르게 된다.

---

### 미검증

| 항목 | 사유 |
|------|------|
| **축 A′ — 강의자료 ↔ 노트북 전 항목** | **대조할 원본이 존재하지 않는다.** `lecture_notes/` 에 이 주제의 슬라이드가 없고 `md/` 두 문서는 학부 참조본이다. 다른 과목 자료로 대체하지 않았다 |
| **Roboflow `.download()` 왕복** | `ROBOFLOW_API_KEY` 없음 → 셀 31 은 가드 경로만 실행. 반환 객체·`Dataset.location`·실제 폴더 배치(셀 30 표의 근거)를 확인하지 못했다 |
| **셀 2·6 의 미실행 분기** | `except ImportError` → `pip install` 경로, 데이터셋 다운로드 경로가 이 환경에서 돌지 않았다 |
| **CPU 환경 실행 시간 (신규)** | GPU 에서 101 epoch = 0.008h. `epochs` 6배 증가분이 CPU 런타임에서 `cell_time_warning_sec: 300` 을 넘는지 이 환경에서 측정 불가 |
| **`patience=0` 으로 바꿨을 때의 숫자** | **부분 해소.** 101 epoch 전수 파싱으로 **`best.pt` 는 epoch 1 로 그대로이고 `best` 행 숫자는 안 바뀐다**는 것까지는 확정했다. 다만 `last.pt`(epoch 120)의 실제 값과 셀 22·23 샘플 결과는 **돌려 보지 않았으므로 예측하지 않는다.** 채택 시 재측정 필요 |
| 그림 렌더링(PNG 육안) | **검증 범위 밖 — 리더가 직접 보고 통과 판정함.** 재확인하지 않았다 |

---

### 상위로 올릴 항목

#### `notebook-author` 에게 (치명 — 즉시)

**V-11 단일 지시.** 하드코딩된 `120 epochs` 라벨 8곳을 epoch 비의존 표현으로 교체. `auditor2` 의 Exercise 2 충돌 FIX 와 **같은 자리·같은 수정**이므로 지시를 하나로 합쳤다. V-12(셀 21 `best.pt` 문장)를 같은 수정에 포함. **리더가 `patience` 처리를 결정하기 전까지 착수하지 말 것** — (b) 를 고르면 전 숫자를 재측정해야 한다.

#### `lecture-spec-analyst` 에게

위 **명세 정정 목록 S-1 ~ S-12**. 특히 **S-2·S-6 의 "가드 문장 존치"** — 명세를 단순 폐기하면 다음 라운드가 같은 오류를 재생산한다.

#### 리더에게

1. **판정 FAIL.** 치명 1 (V-11), 경미 1 (V-12). 축 B·축 C 의 나머지는 전부 통과했고, **두 열의 방향이 다른 문제(라운드 4 V-10 유형)는 이번에 재발하지 않았다.**
2. **V-11 수정안 (a) 라벨만 / (b) `patience=0` 결정 필요.** (b) 는 노트북이 자기 문장과 일치하게 되지만 **리더 기준표를 포함한 전 숫자·셀 22 목록·셀 23 네 패널·셀 29 표의 재측정과 재렌더링**을 요구한다 (리더가 육안 통과 판정한 그림 4장도 다시 봐야 한다 — `auditor2` 지적).
   - **결정에 필요한 사실 하나를 추가로 확인했다: (b) 를 골라도 이야기는 바뀌지 않는다.** 101 epoch 전수 파싱 결과 **epoch 1 을 어느 열에서도 넘어선 epoch 이 0개**이고 최댓값은 epoch 1–11 동률이다 (detection·segmentation 동일). 120 까지 돌려도 `best.pt` 는 그대로 epoch 1 이고 `best` 행의 숫자도 바뀌지 않는다. **바뀌는 것은 `last.pt` 행(epoch 101 → 120)과 샘플 그림뿐**이며, 이미 90/101 epoch 이 양쪽 열에서 epoch 1 미만이므로 방향도 바뀌지 않는다.
   - → **비용 대비 얻는 것이 거의 없다. 본 검증자는 (a) 를 권고한다.** (b) 의 유일한 실익은 "`epochs=120` 이라고 썼으면 120 을 돌린다" 는 결벽인데, (a) 의 epoch 비의존 라벨이 그 요구 자체를 없앤다.
3. **강의자료 쪽이 틀렸다고 판단되는 항목: 없다** — 대조할 강의자료가 존재하지 않기 때문이다.
4. **`patience` 는 프로필/명세 어디에도 없다.** ultralytics 기본값이 조용히 개입해 학습 길이를 바꾼 것이므로, 이 과목이 ultralytics 를 쓰는 모든 회차에 해당하는 항목이다.

---

### 하네스 갱신 제출

| # | 트리거 | 제출 내용 | 대상 |
|---|---|---|---|
| H-1 | **신규 결함 유형** | **"라이브러리에 넘긴 하이퍼파라미터 값을 마크다운·print 라벨에 하드코딩하지 마라. 라이브러리가 그 값을 바꿀 수 있다."** `epochs=120` 을 넘겼지만 실제로 돈 것은 101 이었다(`patience=100` 조기 종료). 같은 성질: `optimizer=auto` 가 `lr0` 를 무시하는 것(이미 알려짐), `batch=-1` 자동 배치. **검증 절차: 학습 로그의 "완료" 줄을 라벨과 대조한다** | `CLAUDE.md` 또는 `notebook-convention-audit` 새 체크포인트 (승인 대상) |
| H-2 | **3라운드 연속 재발 → 저자도 같은 것을 제출함** | 저자 리포트 §9 의 *"여러 행·열을 출력하는 셀의 해설문은 모든 행과 열에 대해 참인지 확인한다"* 에 **동의하며 중복 제출한다.** 라운드 4 에서 제출됐는데 **규약 문서에 반영되지 않아** 저자가 이번 초안에서 또 만들었다(스스로 잡음). **제출이 아니라 반영이 병목이다** | `CLAUDE.md` 새 절 (승인 대상) |
| H-3 | **명세 개정 시 가드 유실** | **"명세의 한 절을 폐기할 때, 그 절 안의 '이렇게 쓰면 거짓이 된다' 형 금지 서술은 함께 버리지 말고 새 절로 옮긴다."** §3.7 폐기로 `best.pt` = epoch 1 가드가 사라져 V-12 가 생겼고, §2 폐기로 mAP 오설명 금지 서술이 사라질 뻔했다 | `lecture-spec-extraction` 스킬 보강 후보 |
| H-4 | **강의자료 부재의 반복 처리** | 이 회차는 5라운드 연속 축 A′ 를 `미검증` 으로 냈다. **프로필에 `lecture_sources` 부재를 회차 단위로 선언하는 필드가 없어** 매 라운드 검증자가 세 위치를 다시 grep 한다. 프로필에 회차별 `lecture_sources.absent: true` + 사유를 둘 수 있는지 | `notebook-profile.json` 스키마 (사용자 확인 필요) |

**강의자료 자체의 오류로 보고할 항목: 없다** (대조할 자료가 존재하지 않는다). 요약본↔슬라이드 불일치도 이 회차에서는 판정 불가.

---
---

## 이전 라운드 이력 (라운드 1~4) — 아래는 **개정 전** 노트북 기준이다

> **최상단 라운드 5 판정이 최신이고 우선한다.** 아래 본문의 셀 번호는 개정 전 35셀 기준이라
> 현재 노트북과 어긋난다 (Step 4 삭제·재번호). 판정 이력 보존용으로만 읽을 것.

### Consistency Verification — Practice12_Object_Detection_and_Segmentation.ipynb (라운드 4, 최종)

#### 판정: PASS — 단, 아래 미검증 3건을 반드시 함께 읽을 것

- **제기 10건 (치명 4 / 경미 6) → 노트북 결함 9건 전부 해소.** 남은 1건(V-8)은 노트북이 아니라
  `md/p12_workspace/02_author_report.md` 의 문구 문제이므로 학생 배포물과 무관하다
- **미검증 3건은 해소되지 않았고 해소될 수 없다** — PASS 는 "검증한 범위에서 일치" 라는 뜻이지
  "전부 확인됐다" 는 뜻이 아니다. 아래 요약과 본문 `미검증` 절을 함께 볼 것
- 셀 번호는 **0-based** (감사 스크립트·저자 리포트와 동일 기준)

##### 미검증 — 최종 보고에 반드시 포함할 것

| 항목 | 사유 | 성격 |
|---|---|---|
| **축 A′ 강의자료 ↔ 노트북 (전 항목)** | **대조할 원본이 존재하지 않는다.** 이 주제의 강의 슬라이드가 없고, `md/` 두 문서는 학부 참조본으로 detection/segmentation/YOLO 키워드 0건 | **이 회차 최대 잔여 위험** |
| Roboflow `.download()` 왕복 | `ROBOFLOW_API_KEY` 없음. 가드 경로만 실행됨. `model_format='yolov8'` 과 `Dataset.location` 은 설치본 SDK 소스로만 확인 | 키 있는 환경에서 1회 실행으로만 닫힘 |
| V-4 의 다른 cwd 재현 | 검증자가 노트북을 수정·실행하지 않음. 다만 수정본이 `.resolve()` 를 적용해 **위험 자체가 구조적으로 제거**되어 실익은 사라짐 | 사실상 해소 |

> **강의자료 축을 다른 과목 자료로 대체하지 않았다.** 과목마다 논리 전개와 scope 가 달라, 남의 자료로
> 판정하면 이 과목에서 옳은 것이 틀린 것으로 잡힌다. 이 축은 끝까지 미검증으로 남긴다.
>
> **향후 이 주제의 강의 슬라이드가 제작되면 본 리포트의 축 B(외부 사실)가 축 A′(강의자료)로 승격된다.**
> 그 시점에 이 리포트가 재검증의 기준선이 되며, 특히 IoU/mAP 를 강의가 어느 깊이까지 다루는지에 따라
> Step 4 의 분량과 셀 17 의 수식 범위가 바뀔 수 있다.

##### 라운드별 이력

| 라운드 | 판정 | 내용 |
|---|---|---|
| 1 | FAIL | 치명 3 (V-1 `masks.data`, V-2 mAP 정의, V-9 통제표 B행) + 경미 6 |
| 2 | — | 저자 수정. 셀 23 에 baseline / `last.pt` / `best.pt` 3줄 출력 추가 |
| 3 | FAIL | **수정이 새 결함을 만듦** — V-10: 셀 24 첫 문장이 셀 23 의 `mAP@50-95` 열과 어긋남 |
| **4** | **PASS** | V-10 해소. 첫 문장이 두 지표를 모두 명시 |

##### 최종 확인 (라운드 4 본, 노트북 실물 대조)

| 확인 | 결과 |
|---|---|
| 셀 24 ↔ 셀 23 네 숫자 | `falls below ... mAP@50` (0.63026 < 0.68437) / `rises above ... mAP@50-95` (0.41646 > 0.39986) — **양쪽 다 일치** |
| 셀 24 ↔ 셀 22 로그 | 진행 표시줄 `1/1` × 20 epoch, `AdamW(lr=0.000119, momentum=0.9)` 문자열 그대로 |
| `mAP50 holds two values` | distinct = `{0.688, 0.630}` 정확히 둘. 지표 이름이 명시되어 참 (`mAP50-95` 는 세 값이므로 이름을 빼면 거짓이 됨) |
| 인과 단정·없는 근거 | **소스 기준** `mosaic` 0 / `warmup` 0 / `overfit` 0 / `due to` 0 / `caused by` 0 / `therefore` 0. `because` 1건은 셀 10 의 검증된 인과(sparse `names` → `KeyError`) |
| 분량 | 셀 24 산문 76/80, 문단 문장수 [2,2], 한 줄 두 문장 0건. 노트북 전체 `markdown_budget.py` **0 over cap** |
| 실행 상태 | 35셀, **에러 셀 0 / 미실행 코드셀 0** |
| V-1 해소 증거 | 셀 26 출력이 `masks.data (7, 448, 640)` / `orig_shape (1380, 2070)` — **출력이 본문을 증명한다.** 내가 해석적으로 예측하고 명세자가 재실측한 값과 정확히 일치 |
| V-3 해소 증거 | 셀 18 이 `same_class` 열을 출력하고 `if class_matches and iou > best_iou`, verdict 를 `best_iou >= iou_threshold` 로 계산. 출력에 `broccoli same_class=False IoU=0.423` 이 보여 셀 17 의 "클래스가 맞아야 한다" 가 증명됨 |
| V-9 해소 증거 | 셀 20 통제표에 `checkpoint evaluated \| no selection \| best of 20, chosen on images/val` 행 추가 |

##### 검증 축의 재정의 — 강의자료 축은 성립하지 않는다

| 축 | 원본(왼쪽) | 수행 여부 |
|---|---|---|
| A′. 강의자료 ↔ 노트북 | `lecture_notes/*.pdf`, `md/lectures_and_formulas.md` | **미검증 — 대조할 원본 없음** |
| A. 명세 ↔ 노트북 | `md/p12_workspace/01_spec_Practice12.md` | 수행 |
| B. 외부 사실 ↔ 노트북 | `https://docs.ultralytics.com/` (WebFetch 직접 확인), 설치본 소스 `ultralytics 8.4.120` | 수행 |
| C. 마크다운 주장 ↔ 같은/다음 셀의 실제 출력 | 실행된 노트북의 셀 출력 | 수행 |

**축 A′ 를 수행했다고 보고하지 않는다.** 프로필 `lecture_sources` 가 가리키는 세 위치를 직접 확인한
결과는 명세 §1 의 기록과 일치한다 — `lecture_notes/` 에는 `Ch1-ML 1_Linear Regression.pdf` 하나뿐이고,
`md/` 의 두 문서는 학부 참조본이며 detection/segmentation/YOLO 키워드가 없다. 다른 과목의 강의자료로
대체하지 않았다. 따라서 이 회차의 **notation 권위는 ultralytics API 이름과 COCO 지표 정의**뿐이며,
과목 기호 규약(`rho`, `w`, `E`, `dJ`)이 적용될 자리는 이 노트북에 존재하지 않는다 (정상).

---

#### 대조표 — 축 B (외부 사실 정확성)

리더가 지정한 8개 항목. 전부 `docs.ultralytics.com` 을 이번 라운드에 직접 열어 대조했다. 기억으로 판정한 항목 없음.

| # | 항목 | 원본 (출처) | 노트북 (셀) | 판정 |
|---|------|------------|------------|------|
| B-1 | IoU 정의 | $\mathrm{IoU}=\dfrac{\|A\cap B\|}{\|A\cup B\|}$, "quantifies the overlap between a predicted bounding box and a ground truth bounding box" (`/guides/yolo-performance-metrics/`) | 셀 17 마크다운 수식, 두 번째 등식 $\dfrac{\|A\cap B\|}{\|A\|+\|B\|-\|A\cap B\|}$ | **일치** (수치 확인 완료, 아래 참조) |
| B-2 | IoU 의 `max(0, ...)` | 명세 §2.1 "교집합이 없을 때 음수 폭이 나오므로 `max(0, ...)` 이 정의의 일부다" | 셀 18 `intersection_width = max(0.0, intersection_x2 - intersection_x1)` (+ 같은 줄 주석이 이유를 설명) | **일치** — 셀 18 출력에 실제로 `IoU=0.000` 인 비접촉 박스가 등장해 근거가 눈에 보인다 |
| B-3 | mAP@50 vs mAP@50-95 (표) | "mAP50: Mean average precision calculated at an IoU threshold of 0.50", "mAP50-95: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95" (`/guides/yolo-performance-metrics/`) | 셀 17 표 (`mAP@50` = 0.50 only / `mAP@75` = 0.75 only / `mAP@50-95` = average over 0.50…0.95) | **일치** |
| B-4 | mAP 의 정의 자체 | "mAP **computes the area under the precision-recall curve**" + "extends across **multiple object classes**" (같은 문서) | **셀 32 Summary** "mAP averages precision over a range of IoU thresholds" | **불일치 — 치명 (V-2)** |
| B-5 | `.box.map50` / `.box.map` 대응 | "`metrics.box.map` — map50-95", "`metrics.box.map50` — map50" (`/modes/val/`) | 셀 21·23 `baseline_metrics.box.map50` → `mAP@50`, `.box.map` → `mAP@50-95` | **일치** |
| B-6 | `.box.mp` / `.box.mr` | 문서 페이지에 서술 없음. **셀 21 출력 자체로 교차 확인** — ultralytics 요약행 `all 4 17 0.644 0.567 0.684 0.4` 과 우리 print `precision 0.64370 / recall 0.56667 / mAP@50 0.68437 / mAP@50-95 0.39986` 이 자릿수까지 대응 | 셀 21 | **일치** (문서가 아니라 실행 출력으로 확인) |
| B-7 | TP 판정이 IoU 임계값으로 결정 | "an IoU threshold of 0.50 is used to define true positives" (같은 문서) | 셀 17 "A prediction counts as correct when it names the right class **and** overlaps the ground truth enough." / "That threshold is what separates detection from classification." | 서술은 **일치**. 다만 셀 18 코드가 클래스 조건을 구현하지 않음 → **V-3** |
| B-8 | detection 라벨 포맷 | "`class x_center y_center width height`", "Box coordinates must be in **normalized xywh** format (from 0 to 1)", "divide `x_center` and `width` by image width…", "Class numbers should be zero-indexed" (`/datasets/detect/`) | 셀 8 마크다운, 셀 9 실물 출력, 셀 12·18 복원 코드 | **일치** |
| B-9 | 좌표 복원식 | 명세 §2.4 $x_1=(c_x-w/2)W$ 등 | 셀 12 `center_x = float(tokens[1]) * image_width` … `x1 = center_x - box_width / 2` | **일치** (수치 확인: 최대 절대차 5.7e-14) |
| B-10 | segmentation 라벨 포맷 | "`<class-index> <x1> <y1> … <xn> <yn>`", "normalized … between 0 and 1", "**minimum of 3 (x, y) points**", "The length of each row does not have to be equal" (`/datasets/segment/`) | 셀 8 "a normalized polygon of at least three vertices", 셀 9 토큰 수 출력(25 → 12 vertices) | **일치** — "at least three" 가 문서 문구 그대로다 |
| B-11 | `data.yaml` 키 의미 | `path` = dataset root dir(상대경로면 전역 `datasets_dir` 기준으로 해석), `train`/`val` = "relative to `path`", `names` = index→name (`/datasets/detect/`) | 셀 5 표 | `train`/`val`/`names` **일치**. `path` 서술만 과잉 → **V-6 (경미)** |
| B-12 | `boxes.xyxy` / `xywhn` / `conf` / `cls` | "Return the boxes in xyxy format", "Return the boxes in xywh format **normalized by original image size**" (`/modes/predict/`) | 셀 15 `boxes.xyxy : (7,4) pixel corners` / `boxes.xywhn : (7,4) normalized, same format as the label file` | **일치** |
| B-13 | `masks.xy` 좌표계 | "A list of mask polygons in pixel coordinates"; 설치본 소스 `Masks.xy` = `ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)` → **원본 이미지 픽셀** | 셀 25 표 "polygon already in original image pixels", 셀 27 이 `masks.xy` 로 원본 위에 겹쳐 그림 | **일치** |
| B-14 | `masks.data` 해상도 | 설치본 `SegmentationPredictor.construct_result` — `retina_masks=False`(기본)일 때 `ops.process_mask(..., img.shape[2:], upsample=True)` → **모델 입력(letterbox) 해상도** | 셀 25 표 "binary grid sized by the model input, **not by the image**" | 서술 자체는 **참**. 그러나 셀 26 출력이 이 문장을 반박 → **치명 (V-1)** |
| B-15 | `result.plot()` BGR | "im_bgr = r.plot() # BGR-order numpy array" (`/modes/predict/`) | `plot()` 을 아예 쓰지 않고 `Rectangle`/`Polygon` 으로 직접 그림 | **해당 없음 — 함정을 회피함** |
| B-16 | Roboflow `model_format` | 설치본 `roboflow 1.4.0` `core/version.py` 의 `friendly_formats` 는 `yolov5`/`yolov7` 만 매핑, 나머지는 검증 없이 API 로 전달. `Dataset.__init__` 에 `self.location` 존재 확인 | 셀 31 `roboflow_version.download('yolov8', location=str(DATA / 'roboflow'))`, `roboflow_dataset.location` 사용 | **일치** — `'yolov11'` 없음 |
| B-17 | Roboflow 폴더 배치 서술 | 리더 지시: "포맷은 같고 배치는 단정하지 말 것" | 셀 30 표는 label format / `data.yaml` 역할 / 학습 호출 세 줄만 `same` 으로 두고, "Folder nesting can differ from one export to the next, and `data.yaml` is what absorbs that." 로 마무리. **구체적 폴더 경로를 단정하는 문장 없음.** 셀 31 은 `rglob` 로 실제 생성된 폴더를 출력 | **일치** |

##### 수치 확인 (수행함)

수식이 동치인지 눈으로만 판정하지 않았다. 별도 스크립트로 계산했고 **이 코드는 노트북에 넣지 않았다.**

| 확인 | 결과 |
|---|---|
| 명세 §2.4 $x_1=(c_x-w/2)W$ vs 노트북 `center_x*W` 후 `- box_width/2` | 4좌표 최대 절대차 **5.7e-14** (부동소수 결합순서 차이뿐) → 동치 |
| 셀 18 의 대수적 IoU vs 래스터로 센 집합비 $\|A\cap B\|/\|A\cup B\|$ | 대수 **0.939** vs 래스터 **0.9385** → $\|A\|+\|B\|-\|A\cap B\|$ 형태가 합집합과 같다는 것을 수치로 확인 |
| 셀 18 출력 `intersection=174816.8 union=186164.9 IoU=0.939` | 독립 재계산과 일치 (union 차 2.7 은 내가 출력에 찍힌 소수 1자리 예측좌표를 쓴 탓) |
| 셀 21 print 값 vs 같은 셀 ultralytics 요약행 | `0.64370/0.56667/0.68437/0.39986` ↔ `0.644 0.567 0.684 0.4` 일치 |
| 셀 23 print 값 vs 같은 셀 val 요약행 | `0.68777/0.42254` ↔ `0.688 0.423` 일치 |
| 셀 29 print 값 vs 같은 셀 val 요약행 | box `0.76875/0.48625`, mask `0.73537/0.41137` ↔ `0.769 0.486 … 0.735 0.411` 일치 |

---

#### 대조표 — 축 A (명세 ↔ 노트북)

##### 저자가 스스로 보고한 4건의 판정 (`02_author_report.md` §4)

| # | 저자 주장 | 판정 | 근거 |
|---|---|---|---|
| 4-1 | 명세 §5 에 없는 이름 3종 추가 (`pretrained_model`, `best_iou`/`best_index`, `fill_colors`) | **정당** | 전부 약어 아님, `CLAUDE.md` "Variable Naming" 위반 없음. `pretrained_model` 은 체이닝 금지(`Model Definition Style`)의 결과라 오히려 규약을 지킨 흔적 |
| 4-2 | 셀 9 출력 형식이 명세 예시와 다름 | **정당** | 값 동일함을 출력으로 확인 (`tokens 25` / `vertices 12`) |
| 4-3 | 셀 18 을 `for` 루프로 | **정당, 단 부작용 1건** | 매칭이 순서가 아니라 IoU 로 정해지는 것이 출력에 드러난다(0.068/0.423/0.939/0.409/…). 셀 19 가 루프 잔여값이 아니라 `predicted_box = detected_boxes[best_index]` 로부터 교집합을 **다시 계산**하는 것도 소스로 확인함. 부작용은 **V-3** |
| 4-4 | 셀 25 `masks.data` 서술을 고쳤다 | **불충분 — 고쳐지지 않았다** | 현재 문장에 여전히 "**not by the image**" 가 남아 있고, 셀 26 출력이 그 문장을 정면으로 반박한다. **V-1** |

##### 저자가 보고하지 않은 이탈

| # | 명세 | 노트북 | 판정 |
|---|---|---|---|
| A-1 | §4 셀 26: "**source 는 `data/sample.jpg`** (2070x1380, 리더 판정 §6 #3)" | 셀 26 이 `data/coco8-seg/images/train/000000000009.jpg` (640x480) 사용 | **미보고 이탈 — V-1 의 직접 원인**. 리더 §6 #3 은 sample.jpg 를 "허용" 한 것이므로 규정 위반은 아니나, 이 선택 때문에 `masks.data` 와 `orig_shape` 가 우연히 같아졌다 |
| A-2 | §2.3: $AP=\int_0^1 p(r)dr$, $mAP=\frac{1}{C}\sum_c AP_c$ | 노트북 어디에도 AP 정의도, "m = 클래스 평균" 도 없다 | **미보고 이탈 — V-2 의 근본 원인** |
| A-3 | §5 변수표의 `annotated_image` | 셀 16 이 별도 배열을 만들지 않고 `sample_image` 위에 직접 그림 | **정당한 이탈.** `plot()` 을 쓰지 않기로 한 결정의 결과. 이름을 억지로 만들 필요 없음 |
| A-4 | §4 셀 00: "세 과제의 출력 형태 표(§4.1)" | 셀 0 은 detection/segmentation 두 줄 표, 세 과제 표는 셀 4 에 있음 | **정당한 이탈.** `CLAUDE.md` "같은 말을 두 번 하지 않는다" 를 지킨 결과 |
| A-5 | §4 셀 10: "우리가 쓴 yaml 의 **앞 12줄**" | 셀 10 이 앞 8줄을, 그것도 **Python list repr 로** 출력 | **경미 — V-7** |

##### 명세와 일치 확인된 항목 (요약)

| 항목 | 명세 | 노트북 | 판정 |
|---|---|---|---|
| 셀 수 / 구성 | 35셀 (MD 15 / CODE 20) | 35셀, MD = {0,1,4,5,8,11,13,17,20,24,25,28,30,32,33} = 15개 | 일치 |
| house style | Step 첫 MD 셀은 `---` 후 `## Step N.` | 셀 1·4·5·13·17·20·25·28·30 전부 준수 | 일치 |
| 하이퍼파라미터 (§3.6) | `epochs=20, imgsz=320, batch=4, seed=42, amp=False, workers=0` | 셀 22·29 train, 셀 21·23·29 val 전부 동일 | 일치 |
| 통제표 (§4.2) | 격리 변수는 fine-tuning 여부 하나 | 셀 20 표 8행이 §4.2 와 문자 단위로 일치. 셀 21·22 가 각각 **별개의 `YOLO()` 인스턴스** | 일치 |
| §3.2 함정 회피 | `train()` 은 새 인스턴스로만 | 셀 22 `finetune_model`, 셀 29 `segmentation_finetune_model` 모두 그 자리에서 생성. 지표 0.68777 / 0.76875 로 붕괴(0.00063 / 0.0000) 아님이 **출력으로 증명됨** | 일치 |
| §3.1 경로 봉인 | 가중치 `build/`, `project` 절대경로, `settings.update()` 금지 | 셀 3 `RUNS = (BUILD/'runs').resolve()`, 학습 로그의 `project=D:\...\build\runs`, `save_dir=D:\...\build\runs\detect_finetune`. 루트에 `.pt`·`runs/` 없음(실측) | 일치 |
| §3.4 내장 yaml 출력 금지 | 우리가 쓴 파일만 출력 | 셀 10 이 `data/coco8.yaml` 만 읽어 출력. 이모지·박스문자 없음 | 일치 |
| §3.7 과장 금지 | "성능 향상이 아니라 파이프라인이 도는지" | 셀 24 "An eight-image dataset shows that the pipeline runs, not that fine-tuning helps." + 셀 23 출력이 0.68437→0.68777 로 그 문장을 뒷받침 | 일치 |
| 결과 비교 방식 | `assert`/`allclose` 금지, `print` 로 나란히 | 셀 23 `pretrained` / `fine-tuned` 두 줄 print. 노트북 전체에 `assert`·`allclose` 0건 | 일치 |

---

#### 불일치 상세

##### V-1. 셀 25 의 `masks.data` 서술을 셀 26 의 출력이 반박한다 — **치명**

- 노트북 셀 25 (마크다운):
  > \| `masks.data` \| binary grid sized by the model input, **not by the image** \|
- 노트북 셀 26 (실제 출력):
  ```
  masks.data   : (5, 480, 640)
  orig_shape   : (480, 640)
  ```
- 무엇이 다른가: 문장은 "이미지 크기가 아니다" 라고 단정하는데, **바로 아래 두 줄이 같은 숫자다.**
- 사실관계 (내 판정): **문장은 참이고 예시가 나쁘다.** 설치본 `ultralytics 8.4.120` 의
  `SegmentationPredictor.construct_result` 는 기본값 `retina_masks=False` 에서
  `ops.process_mask(proto, ..., img.shape[2:], upsample=True)` 를 쓰므로 masks 는 **letterbox 된 모델 입력
  해상도**다. 640x480 이미지는 `imgsz=640` 에서 letterbox 결과가 480x640(480 이 32의 배수라 패딩 0)이라
  **우연히 원본과 같아진다.** 즉 이 이미지가 그 문장을 증명할 수 없는 유일한 종류의 이미지다.
- 학생에게 미치는 영향: 규약이 금지한 종류의 결함이다. 학생은 문장과 숫자 중 숫자를 믿고,
  "내가 뭘 잘못 읽었나" 로 멈춘다. 그리고 이 표가 바로 다음 셀에서 `masks.xy` 를 쓰는 이유의 전부다 —
  근거가 무너지면 셀 27 의 선택이 임의로 보인다.
- 수치 확인: **수행함.** `data/sample.jpg` = 2070x1380. `imgsz=640` letterbox → 640/2070=0.3092,
  1380x0.3092=426.7 → stride 32 패딩 → **(N, 448, 640)**. `orig_shape` 는 (1380, 2070) 이므로 **명백히 다르다.**
  이 값은 명세 §3.5 에 기록된 리더 실측 `(7, 448, 640)` vs `(1380, 2070)` 과 정확히 일치한다.
  **`lecture-spec-analyst` 가 내 해석적 예측을 받은 뒤 독립적으로 재실측해 확인했다** (명세 §3.5.1 신설):

  | 이미지 | `orig_shape` | `masks.data` | 같은가 |
  |---|---|---|---|
  | coco8 `000000000009.jpg` | (480, 640) | (5, 480, 640) | **True** |
  | `data/sample.jpg` | (1380, 2070) | (7, 448, 640) | **False** |

  같은 실측에서 coco8-seg 8장의 크기가 `640x480, 640x426, 640x428, 640x425, 481x640, 640x478, 381x500,
  640x488` 로 확인됐다 — **대부분 가로가 640** 이라, 이 데이터셋 안에서 이미지를 바꾸는 것만으로는
  문제가 해소되지 않는다. `data/sample.jpg` 여야 한다.
- 수정 방향 (**둘 중 하나. 첫째를 권한다**):
  1. **셀 26·27 의 `segmentation_image_path` 를 `DATA / 'sample.jpg'` 로 되돌린다.** 그러면
     `masks.data (N, 448, 640)` ≠ `orig_shape (1380, 2070)` 이 되어 **출력이 문장을 증명**한다.
     이것이 명세 §4 셀 26 이 원래 지정한 것이고, 리더 §6 #3 이 승인한 것이며, 2070x1380 이라 마스크
     형태도 더 잘 읽힌다. 파일은 이미 `data/sample.jpg` 에 존재함을 확인했다. 부수 효과: 셀 26 의
     인스턴스 수가 달라지므로 셀 27 `fill_colors` 8색 상한을 함께 볼 것 (V-8).
  2. 이미지를 유지한다면 문장에서 단정을 걷어낸다 —
     "binary grid at the model input size, which may or may not equal the image size" 로 바꾸고
     `masks.xy` 를 쓰는 이유를 "재조정이 필요 없다" 쪽으로 옮긴다.

##### V-2. Summary 의 mAP 정의가 공식 문서와 어긋난다 — **치명**

- 노트북 셀 32 (Summary):
  > IoU decides whether a detection counts, and **mAP averages precision over a range of IoU thresholds.**
- 원본: `https://docs.ultralytics.com/guides/yolo-performance-metrics/`
  > mAP "**computes the area under the precision-recall curve**" … extends across "**multiple object classes**"
  > mAP50-95: "The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95"
- 무엇이 다른가: 세 겹의 평균이 하나로 뭉개졌다.
  1. **AP** = precision-recall 곡선 아래 면적(= recall 에 대한 적분). 노트북 문장은 이것을 "precision 의 평균" 으로 바꿔 놓았다
  2. **m** = **클래스 평균**. 노트북 어디에도 이 말이 없다
  3. IoU 임계값 평균은 **mAP@50-95 에만** 해당한다. 문장은 그것을 mAP 일반의 정의로 승격시켰다.
     이 정의대로면 `mAP@50` 은 "0.50 하나만 쓰는데 평균" 이라는 자기모순이 된다 (셀 17 표가 바로 그렇게 적혀 있다)
- 학생에게 미치는 영향: Summary 는 학생이 마지막으로 읽고 가장 오래 기억하는 셀이다. 이 회차의 주력
  지표 이름을 틀리게 배우면, 이후 논문의 `AP`, `AP50`, `AP_S` 표를 전부 잘못 읽는다.
- 근본 원인: 명세 §2.3 이 $AP=\int_0^1 p(r)dr$, $mAP=\frac{1}{C}\sum_c AP_c$ 를 요구했는데 **노트북이
  이 두 수식을 아예 담지 않았다** (A-2). 본문에 정의가 없으니 요약이 어림짐작으로 채워졌다.
  `CLAUDE.md` "Summary 에 새 내용을 넣지 않는다" 위반이기도 하다 — 본문에 없던 정의가 요약에서 처음 나온다.
- 수치 확인: 해당 없음 (정의 문제).
- 수정 방향:
  1. **셀 17 에 한 줄을 추가한다.** 표 바로 위 또는 아래에
     $$AP = \int_0^1 p(r)\,dr, \qquad mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c$$
     그리고 산문 한 문장: "AP is the area under the precision-recall curve of one class, and the leading m
     averages it over classes." (셀 17 은 현재 산문 4문장이므로, "Raise it and the same prediction stops
     counting." 를 지우고 이 문장을 넣으면 분량 중립이다)
  2. **셀 32 문장을 사실로 고친다.** 예: "IoU decides whether a detection counts; AP is the area under one
     class's precision-recall curve, mAP averages it over classes, and mAP@50-95 averages that again over ten
     IoU thresholds." (1번을 먼저 하면 이 문장은 새 내용이 아니라 요약이 된다)

##### V-3. 셀 17 은 "클래스가 맞아야 한다" 고 하는데 셀 18 의 매칭은 클래스를 보지 않는다 — 경미

- 노트북 셀 17: "A prediction counts as correct when it **names the right class** and overlaps the ground truth enough."
- 노트북 셀 18: 루프가 `for index in range(len(detected_boxes))` 로 **전 검출**을 돌고
  `if iou > best_iou:` 하나로만 최적을 고른 뒤,
  `print(f'best match : IoU={best_iou:.3f}  -> a true positive at threshold 0.50')` 를 **무조건** 출력한다.
- 실제 출력이 이 간극을 눈에 보이게 만든다:
  ```
  bowl         ... IoU=0.068
  broccoli     ... IoU=0.423     <- ground truth 는 bowl 인데 클래스가 다른 예측이 함께 순위에 들어 있다
  bowl         ... IoU=0.939
  ```
- 무엇이 다른가: 서술은 **클래스 ∧ IoU** 두 조건인데, 코드는 **IoU 한 조건**이다.
  게다가 마지막 줄의 "a true positive at threshold 0.50" 은 계산 결과가 아니라 **하드코딩된 문자열**이라,
  `best_iou` 가 0.3 이어도 그대로 출력된다.
- 학생에게 미치는 영향: 이 셀은 "TP 판정이 IoU 임계값으로 정해진다" 를 손으로 보여주기 위해 존재한다
  (명세 §2.2). 조건 하나가 코드에서 빠져 있으면, 남는 교훈은 "가장 겹치는 박스가 정답" 이 된다.
  현재 데이터에서는 우연히 최적이 `bowl` 이라 결론이 참이지만, **참인 이유가 코드에 없다.**
- 수치 확인: 수행함 — 최적 매칭은 `bowl`(IoU 0.939)로 클래스가 실제로 일치한다. Exercises 1·3 의
  경로에서도 최적은 `bowl` 로 유지되므로 현재 노트북이 거짓을 출력하지는 않는다. 그래서 치명이 아니다.
- 수정 방향 (한 줄씩만 추가한다):
  ```python
  class_index = int(detection_result.boxes.cls[index])
  class_matches = class_index == ground_truth_class          # the class condition, not only the overlap
  ...
  if class_matches and iou > best_iou:
  ```
  그리고 마지막 print 를 계산 결과로 바꾼다:
  ```python
  verdict = 'a true positive' if best_iou >= 0.50 else 'not a true positive'
  print(f'best match : IoU={best_iou:.3f}  -> {verdict} at threshold 0.50')
  ```
  `class_matches` 를 각 행 출력에 함께 찍으면 broccoli 행이 IoU 0.423 을 갖고도 후보에서 빠지는 것이
  출력에 드러나 셀 17 의 문장이 그대로 증명된다.

##### V-4. `data=` 가 상대경로라 다른 cwd 에서 조용히 다른 데이터셋을 읽을 수 있다 — 경미

- 노트북 셀 21·22·23·29: `data=str(detection_yaml_path)` → 학습 로그에 `data=data\coco8.yaml` (상대경로).
- 무엇이 다른가: 명세 §3.1 은 `project` 를 절대경로로 못박았지만 `data` 는 그대로 두었다. ultralytics 의
  `check_det_dataset` 은 주어진 이름을 찾지 못하면 **패키지 내부까지 뒤져** 동명의 `coco8.yaml` 을 찾아낸다.
  이 노트북의 파일명이 하필 패키지 내장 파일과 **같은 이름**이라, 학생이 노트북을 다른 cwd 에서 열면
  에러 없이 내장 `coco8.yaml`(전역 datasets 폴더를 가리킴)로 학습이 돌 수 있다.
- 학생에게 미치는 영향: 실패가 조용하다. 이 회차 전체가 "경로를 봉인한다" 를 가르치는데, 마지막 한 곳이
  봉인되지 않았다.
- 수치 확인: 미수행 (다른 cwd 재현은 노트북 재실행이 필요해 검증자 권한 밖).
- 수정 방향: 셀 10 에서 `detection_yaml_path = (DATA / 'coco8.yaml').resolve()` 로 두거나,
  호출부를 `data=str(detection_yaml_path.resolve())` 로 바꾼다. 한 글자 수준이고 `project` 와 형태가 맞는다.

##### V-5. 셀 5 의 `path` 설명이 문서 정의보다 강하다 — 경미

- 노트북 셀 5: \| `path` \| dataset root, **written as an absolute path** \|
- 원본 (`/datasets/detect/`): `path` 는 dataset root directory이며, 상대경로면 전역 datasets 디렉터리 기준으로 해석된다. **절대경로가 요구사항이 아니다.**
- 무엇이 다른가: 우리가 **선택한 것**이 키의 **정의**로 적혀 있다.
- 학생에게 미치는 영향: 남의 `data.yaml`(예: Roboflow export)에서 상대 `path` 를 보면 잘못된 파일이라고
  판단한다. 하필 셀 30 이 "남의 export 를 읽는" 절이다.
- 수정 방향: "dataset root; written here as an absolute path so it does not depend on the global datasets folder"
  — 이렇게 쓰면 명세 §3.1 #4(기본 `datasets_dir` 이 다른 과목 폴더를 가리킴)의 교훈까지 한 줄에 들어간다.

##### V-6. 셀 10 이 yaml 을 Python list repr 로 출력한다 — 경미

- 명세 §4 셀 10 예상 출력: "우리가 쓴 yaml 의 앞 12줄"
- 노트북 셀 10: `print(detection_yaml_path.read_text().strip().split('\n')[:8])` → 실제 출력
  ```
  ['path: D:/Main/.../data/coco8', 'train: images/train', 'val: images/val', 'names:', '  0: person', ...]
  ```
- 무엇이 다른가: 이 셀의 목적은 "우리가 쓴 yaml 이 어떻게 생겼는지 보여주기" 인데, 따옴표와 쉼표가 낀
  한 줄짜리 리스트라 **yaml 로 보이지 않는다.** 들여쓰기 두 칸(`  0: person`)이 `names` 의 하위 항목이라는
  구조가 특히 안 보인다.
- 수정 방향: `print('\n'.join(detection_yaml_path.read_text().strip().split('\n')[:8]))`.
  줄 수는 명세대로 12줄로 늘려도 좋다.

##### V-7. 셀 27 의 `fill_colors` 가 8색 고정이라 인스턴스가 9개면 IndexError — 경미

- 노트북 셀 27: `fill_colors = [... 8개 ...]`, `edgecolor=fill_colors[index]`
- 현재 5 인스턴스라 통과한다. 그러나 Exercises 1 이 `conf` 를 바꾸라고 지시하고 있고, `conf` 를 낮추면
  인스턴스가 늘어난다. **V-1 을 1번 방식(sample.jpg)으로 고치면 인스턴스 수가 바뀌므로 반드시 함께 본다.**
- 수정 방향: `fill_colors[index % len(fill_colors)]`.

##### V-9. 셀 20 통제표의 B 서술과 셀 23 이 실제로 평가하는 가중치가 다르다 — **치명 (추가, 감사자 제보로 재검증)**

- 노트북 셀 20 (통제표): \| **weights before evaluation** \| COCO pretrained, untouched \| **COCO pretrained + 20 epochs on coco8** \|
- 노트북 셀 22 의 실제 출력 (학습 로그 말미):
  ```
  20 epochs completed in 0.004 hours.
  Validating D:\...\build\runs\detect_finetune\weights\best.pt...
  ```
- 셀 22 의 epoch 별 val 행 (내가 로그 전체를 추출해 확인):

  | epoch | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | 1–11 | 0.637 | 0.667 | **0.688** | 0.422 (11 epoch 동안 소수 3자리까지 불변) |
  | 12–20 | 0.535 | 0.583 | **0.630** | 0.416 (9 epoch 동안 불변) |
  | 최종(`best.pt`) | 0.677 | 0.667 | 0.688 | 0.423 |

  segmentation(셀 29)도 같은 모양이다 — box mAP50 이 1–11 epoch `0.769`, 12–20 epoch `0.565`,
  최종 `best.pt` 가 `0.769`.
- 무엇이 다른가: 표의 B 는 "**20 epoch 학습한 가중치**" 라고 적혀 있지만, 셀 23 이 평가하는 것은
  `train()` 이 자동으로 되돌려 놓은 **`best.pt` — 21개 체크포인트 중 val 점수가 가장 높은 것**이다.
  그리고 그 val 은 **비교 결과를 보고하는 바로 그 split** 이다. 즉
  - A(pretrained) = 체크포인트 1개를 val 에서 평가
  - B(fine-tuned) = 체크포인트 21개를 val 에서 평가한 뒤 **최댓값을 고른 것**

  두 팔의 절차가 다르므로 `CLAUDE.md` "비교 실험 통제 변수" 가 요구하는 "격리 변수 하나" 가
  성립하지 않는다. 개선폭 0.68437 → 0.68777 은 이 선택 절차가 만든 값이다.
- 학생에게 미치는 영향: 표를 읽고 "20 epoch 돌린 결과가 이 숫자" 라고 배운다. 실제로 20 epoch 돌린
  가중치의 val mAP50 은 **0.630 으로 사전학습보다 낮다**. 그리고 그 사실이 같은 셀의 출력 안에
  epoch 12–20 행으로 **이미 화면에 찍혀 있다.** V-1 과 정확히 같은 종류의 결함이다 —
  본문이 주장하는 것과 바로 위/아래 출력이 어긋난다.
- 수치 확인: **수행함** (위 표는 셀 22·29 의 stdout 을 전부 추출해 집계한 것). 학습 동역학은
  설치본 소스로 확인했으며, 근거는 셋이다:
  1. **전 학습이 optimizer step 20회다.** train 4장 · `batch=4` → `nb = len(train_loader) = 1`,
     즉 epoch 당 iteration 1회 × 20 epoch. 두 학습 로그 첫머리의
     `AdamW(lr=0.000119, momentum=0.9)` 가 실제 적용된 학습률이다. 가중치가 거의 움직이지 않는다.
  2. **val 4장 · 17 instances 의 mAP 는 성긴 계단함수다.** 지표가 11 epoch 동안 소수 3자리까지
     *완전히* 고정되었다가 한 번에 한 단 떨어지는 모양은 "서서히 나빠졌다" 가 아니라
     "임계를 한 번 넘었다" 를 뜻한다. 값이 연속적으로 흐르지 않는다는 것 자체가 근거다.
  3. ~~**하락 지점이 `close_mosaic=10` 과 맞물린다.**~~ → **철회. 반증됨. 아래 참조.**

- **[철회] 위 3번은 사실이 아니다 — 리더의 통제 실험이 반증했다.**
  나는 소스에서 발화 조건 `if epoch == (self.epochs - self.args.close_mosaic)` = `20-10=10`(0-based)
  → 1-based epoch 11 을 확인하고, 지표가 epoch 12 부터 바뀌는 것과 맞물린다고 적었다.
  **산술은 맞았지만 결론이 틀렸다.** 리더가 `close_mosaic` 하나만 바꾸고 나머지를 노트북과 동일하게 둔
  실행을 돌린 결과:

  | 조건 | 결과 |
  |---|---|
  | `close_mosaic=0` (로그에 `Closing dataloader mosaic` 자체가 없음) | epoch 1–11 mAP50 **0.688**, epoch 12–20 **0.630** |

  **mosaic 종료를 제거해도 같은 지점에서 같은 하락이 난다.** 로그의 두 줄이 가까운 자리에 보인 것은
  상관이었지 인과가 아니었다. 나는 "인과가 아니라 일치로만 적는다" 로 선을 그었지만,
  **일치를 언급하는 것 자체가 학생에게 잘못된 연결을 시사하므로 해설에서 `close_mosaic` 을 빼야 한다.**
  이 확인의 비용은 10초였고, 감사자도 나도 그 10초를 쓰지 않았다.

- **살아남은 관측 1·2 를 노트북 자체 출력으로 재확인했다** (소스가 아니라 학생이 보는 화면에 있는가):

  | 관측 | 노트북 출력의 근거 | 셀 22 | 셀 29 |
  |---|---|---|---|
  | epoch 당 iteration 1회 | 진행 표시줄이 `1/1` (20 epoch 전부) | 확인 | 확인 |
  | 적용 학습률 | `optimizer: AdamW(lr=0.000119, momentum=0.9)` 인쇄됨 | 확인 | 확인 |
  | val 규모 | val 행이 `all 4 17` — 4장 / 17 instances | 확인 | 확인 |
  | 지표가 성긴 계단 | 20 epoch 동안 mAP50 이 **두 값밖에 갖지 않는다** | `0.688`, `0.630` | `0.769`, `0.565` |

  즉 관측 1·2 는 **설치본 소스를 열지 않아도 노트북 출력만으로 학생이 확인할 수 있다.** 해설에 쓰기에
  적합한 근거는 이 넷뿐이다. mAP50-95 는 detection 에서 `0.416/0.422/0.423` 세 값을 가지므로
  ("두 값" 은 mAP50 에 한한 서술) 해설에서 "두 값" 을 쓸 때는 mAP50 을 가리켜야 한다.

- **주의 — 해설에서 빼더라도 `Closing dataloader mosaic` 은 셀 22·29 출력에 인쇄되어 있다** (확인함).
  학생 화면에 그 문장이 하락 지점 근처에 남아 있으므로, 언급하지 않는 것만으로는 학생이 스스로
  인과를 메울 수 있다. 해설의 닫는 문장이 **"이 규모에서는 어느 방향으로도 판단할 수 없다"** 를
  분명히 말해야 그 빈자리가 메워지지 않는다. (해설에 mosaic 을 넣자는 뜻이 **아니다** — 검증되지 않은
  인과를 인쇄하는 것이 더 나쁘다.)
- 확인 과정에서 **내가 틀릴 뻔한 것 하나를 기록해 둔다.** warmup 이 전 구간을 덮어 학습이 성립하지
  않았다는 설명을 쓰려다 소스를 확인했더니, 이 빌드의
  `_get_warmup_iterations` 는 `round(min(warmup_epochs, epochs-1) * nb)` = `round(min(3.0, 19) * 1)` = **3 iteration**
  이다. 하한 100 iteration 을 두는 다른 버전과 다르다. **warmup 은 정상적으로 끝났고 이 현상과 무관하다.**
  기억으로 썼으면 노트북에 거짓 해설이 인쇄될 뻔했다.
- 수정 방향 (둘 다 한다):
  1. 셀 20 표의 B 행을 사실대로: **"COCO pretrained + 20 epochs on coco8, best checkpoint by val mAP"**.
     그리고 표 아래 한 문장으로 "The two arms are not selected the same way" 를 명시하거나,
     `CLAUDE.md` 가 허용하는 정직한 서술 —
     "this comparison picks B's checkpoint on the same split it reports, so B is favoured by construction".
  2. 셀 24 해설에 **epoch 별 지표가 12번째부터 떨어졌다는 사실**을 한 줄 넣는다. 이것이 오히려
     §3.7 이 요구한 "정직한 결과" 를 강화한다 — "the per-epoch log shows val mAP falling after the
     mosaic augmentation closes, and the reported number is the best checkpoint, not the last one."
- 등급 주: 감사자(`convention-auditor`)가 같은 출력을 보고 BLOCK 으로 올렸다. 감사자는 이것을
  "학습이 모델을 열화시켰는데 셀 24 가 반대로 서술" 로 읽었고, 나는 **"셀 24 의 문장은 거짓이 아니지만
  셀 20 의 표가 거짓"** 으로 읽는다. `best.pt` 를 배포하는 것은 표준 관행이고 셀 24 의
  "fine-tuning helps 라고 말할 수 없다" 는 결론은 오히려 이 데이터가 뒷받침한다. 어긋난 것은
  **표의 B 서술과 두 팔의 선택 절차**다. 저자에게는 이 프레이밍으로 단일 지시가 나가야 한다.

##### V-8. 저자 리포트 §6 의 "일치" 표기가 과하다 — 경미 (노트북 아닌 리포트 문제)

- 저자 리포트: detection fine-tune mAP@50-95 `0.42254`(노트북) vs `0.41836`(리더 프로브) → "일치"
- 두 값은 0.004 차이로 다르다. 원인은 프로브와 노트북의 실행 조건 차이(예: 이전 셀들이 소비한 RNG 상태)일
  가능성이 높고 결론에 영향은 없으나, **"일치" 로 적으면 다음 회차가 재현성 기준선을 잘못 잡는다.**
- 수정 방향: 리포트 문구를 "동일 자릿수, 0.004 차 — fuse 붕괴(0.00063)와는 명확히 구분됨" 으로.
  노트북 수정은 불필요.

---

#### 라운드 3 재검증 — 셀 24 (변경된 유일한 셀)

##### 판정: **FAIL** — 5개 확인 항목 중 4개 통과, 1개 불일치 (치명)

셀 24 는 **셀 22 의 학습 로그와 셀 23 의 출력 표 두 곳을 동시에 주장한다.** 양쪽을 다 열어 대조했다.

| # | 셀 24 의 주장 | 대조 대상 | 실제 | 판정 |
|---|---|---|---|---|
| 1 | `one iteration per epoch` | 셀 22 진행 표시줄 | 20 epoch 전부 `1/1` | **통과** |
| 2 | `` `lr=0.000119` `` | 셀 22 optimizer 행 | `AdamW(lr=0.000119, momentum=0.9)` — 문자열 그대로 | **통과** (반올림·표기차 없음) |
| 3 | `` `mAP50` holds two values `` | 셀 22 의 epoch 별 val 행 | distinct mAP50 = `{0.688, 0.630}` — 정확히 둘 | **통과** (주 아래) |
| 4 | `scores below the pretrained baseline` | **셀 23 출력 표** | **아래 참조** | **불일치 — 치명** |
| 5 | 인과 표현·없는 근거 | 셀 24 전문 | `because` / `caused` / `due to` / `mosaic` / `warmup` / `overfit` / `therefore` **전부 0건** | **통과** |

- 3번 주 — **참이지만 좁다.** mAP50-95 는 `{0.416, 0.422, 0.423}` 로 **세 값**이다. 문장이 `mAP50` 을
  명시했기 때문에만 참이므로, 지표 이름을 빼거나 일반화하면 즉시 거짓이 된다.
- 5번 주 — `so the weights barely moved` 의 `so` 는 남아 있으나, 이는 (1 iteration × 작은 lr) → (가중치가
  거의 안 움직임) 이라는 **자명하게 건전한 추론**이며, 반증된 `close_mosaic` 류의 인과 주장이 아니다. 허용.

##### 불일치 상세 — V-10. 셀 24 첫 문장을 셀 23 의 두 번째 열이 반박한다 — **치명**

- 노트북 셀 24 첫 문장:
  > The model that ran all 20 epochs **scores below the pretrained baseline.**
- 노트북 셀 23 의 실제 출력 표:
  ```
                          mAP@50    mAP@50-95
  A pretrained           0.68437    0.39986
  B after 20 epochs      0.63026    0.41646
  B best checkpoint      0.68777    0.42254
  ```
- 무엇이 다른가: `mAP@50` 열에서는 `0.63026 < 0.68437` 로 문장이 맞다. 그러나 **`mAP@50-95` 열에서는
  `0.41646 > 0.39986` 으로 방향이 반대**다. 문장은 지표를 한정하지 않았으므로, **같은 표의 두 번째 열이
  문장을 반박한다.**
- 학생에게 미치는 영향: **이 라운드 내내 우리가 잡아 온 결함(V-1·V-2·V-9)과 정확히 같은 종류다.**
  게다가 그 결함들을 고치는 수정이 새 결함을 만들었다. 학생은 세 줄짜리 표를 보라고 안내받은 직후
  표를 읽고, 두 번째 열에서 문장과 어긋나는 것을 본다.
- 수치 확인: **수행함** (위 표는 셀 23 의 실제 stdout).
- 참고로 **2문단의 주장은 두 열 모두에서 성립한다** — "The higher number belongs to the checkpoint
  chosen automatically" 는 `mAP@50`(0.68777 최대), `mAP@50-95`(0.42254 최대) 양쪽에서 참이다. 확인함.
  결함은 첫 문장 하나뿐이다.

##### 수정안 — 분량 문제 없음 (오히려 줄어든다)

리더가 "산문 여유 5단어(75/80)" 를 걱정했으나, **권장안은 현재보다 1단어 짧다.**
과목 스크립트 `.claude/scripts/markdown_budget.py` 의 실제 카운터를 import 해 측정했다 (눈대중 아님):

| 안 | 산문 | 문단별 문장 수 | 한 줄 두 문장 | 비고 |
|---|---|---|---|---|
| 현재 | 75/80 | [2, 3] | 0 | 첫 문장이 반박됨 |
| **권장 (B)** | **74/80** | [2, 3] | 0 | **지울 문장 없음** |
| 대안 (C) | 79/80 | [2, 3] | 0 | "two values" 관측 유지, 여유 1단어 |
| 최소 (A) | 77/80 | [2, 3] | 0 | 첫 문장에 `` on `mAP@50` `` 만 덧붙임 |

**권장안 (B) — 첫 두 줄만 교체, 나머지 3줄은 그대로:**
> After all 20 epochs the model scores below the pretrained baseline on `mAP@50` and above it on `mAP@50-95`.
> The log shows one iteration per epoch at `lr=0.000119`, so the weights barely moved and neither column settles anything.

- **두 지표가 어긋난다는 사실 자체가 결론의 근거가 된다.** 숨겨야 할 구멍이 아니라 "이 규모에서는
  어느 방향으로도 판단할 수 없다" 의 가장 강한 증거이며, 마지막 문장("the pipeline runs, not that
  fine-tuning helps")과 곧바로 이어진다.
- 부수 이득: 3번의 좁은 주장(`mAP50` 만 두 값)이 빠져 **다음 회차에 지표 이름이 바뀌면 거짓이 되는
  취약점이 사라진다.**
- 최소 수정만 원하면 (A) — 첫 문장 끝에 `` on `mAP@50` `` 두 단어를 붙인다(77/80). 다만 두 번째 열이
  왜 반대 방향인지는 설명되지 않은 채 남는다.

---

#### 미검증

| 항목 | 사유 |
|------|------|
| **강의자료 ↔ 노트북 전 항목** | **대조할 원본 없음.** `lecture_notes/` 에 이 주제의 슬라이드가 존재하지 않고, `md/` 두 문서는 학부 참조본이며 detection/segmentation/YOLO 키워드가 0건이다. 다른 과목 자료로 대체하지 않았다 |
| Roboflow `.download()` 의 실제 반환과 폴더 배치 | `ROBOFLOW_API_KEY` 없음. 셀 31 의 가드 경로만 실행됐다. `model_format='yolov8'` 과 `Dataset.location` 은 설치본 SDK 소스로 확인했으나, **API 왕복은 확인하지 못했다.** 명세 §6 이 이미 "저작·검증 단계에서 해소되지 않는 잔여 위험" 으로 기록한 것과 같은 항목 |
| V-4 (상대 `data=` 경로)의 실제 오작동 재현 | 다른 cwd 에서 노트북을 재실행해야 하고, 검증자는 파일을 수정·실행하지 않는다. 정성 판정에 그쳤다 |

---

#### 상위로 올릴 항목

##### `spec` 에게 — 명세 정정 요청 (노트북만 고치면 다음 라운드에 재생산된다)

1. **§2.3 의 $AP$ / $mAP$ 수식이 §4 셀 명세로 내려가지 않았다.** §4 셀 17 의 지시는
   "IoU 수식(§2.1) + precision/recall(§2.2) + **mAP 표**(§2.3)" 이라, 저자가 표만 옮기고 정의 수식을
   빠뜨려도 명세를 지킨 것이 된다. 그 빈자리를 Summary 가 어림짐작으로 채운 것이 **V-2(치명)** 이다.
   → §4 셀 17 행을 "**§2.3 의 AP·mAP 정의 수식을 그대로 넣고** + mAP 표" 로 고칠 것.
2. **§4 셀 26 의 이미지 지정이 근거와 함께 오지 않았다.** "`data/sample.jpg` (2070x1380, 리더 판정)" 이라고만
   적혀 있어, 저자가 "로컬 coco8 이 더 낫다" 는 일반 원칙으로 대체할 여지가 있었다. 실제로 그렇게 됐고
   **V-1(치명)** 이 나왔다. → "**`masks.data` 와 `orig_shape` 가 달라야 §3.5 의 서술이 출력으로 증명되므로
   Step 6 은 반드시 대형 이미지를 쓴다**" 는 *이유*를 명세에 붙일 것. 이유가 없는 지정은 재량으로 뒤집힌다.

##### 리더에게

- **판정 FAIL.** 치명 2건(V-1, V-2)은 학생이 첫 읽기에 부딪히는 자리다. 나머지 7건은 경미.
- **강의자료 축은 미검증이다.** 이 리포트는 "노트북이 강의와 맞는가" 를 판정하지 않았다 — 판정할 원본이 없다.
  명세 §6 #1 대로 이 명세와 노트북이 **향후 슬라이드 제작 시의 대조 기준**이 된다면, 그 순간 이 리포트의
  축 B(외부 사실)가 축 A′(강의자료)로 승격된다. 그때 재검증이 필요하다.
- **강의자료 자체의 오류로 올릴 항목은 없다** (대조할 강의자료가 없으므로).

---

#### 하네스 갱신 제출

**[검증자 트리거 — 규약에 없어 판단이 갈릴 수 있는 항목]**
"마크다운의 단정문은 **같은/다음 셀의 출력이 그것을 증명하거나 최소한 반박하지 않아야 한다**" 를
`CLAUDE.md` 의 규약으로 승격 / 위치: `CLAUDE.md` "Notebook Comments" 또는 "설명 분량" 다음에 새 절 /
근거: 이번 라운드 V-1. 저자는 이 문제를 **스스로 인지하고 문장을 한 번 고쳤는데도 고쳐지지 않았다**
(리포트 §4-4). 문장을 다듬는 것으로는 해결되지 않고 **예시(입력 데이터)를 바꿔야 하는 종류의 결함**이라,
"문장을 완화하라" 가 아니라 "출력이 문장을 증명하게 만들라" 로 규약이 서 있어야 다음 회차가 같은 자리에서
같은 실패를 반복하지 않는다. 정규식 감사로는 잡히지 않으므로 사람(검증자) 체크리스트 항목이어야 한다.

**[검증자 트리거 — 명세가 강의/원본을 옮기는 패턴의 결함]**
명세 §2(수식)의 항목이 §4(셀 명세)로 내려갈 때 **"이 수식을 어느 셀에 문자 그대로 넣는가" 가 셀 행에
명시되지 않으면 누락된다** / 위치: `.claude/skills/lecture-spec-extraction/` 의 §4 작성 지침 /
근거: 이번 라운드 V-2·A-2. §2.3 에 $AP=\int_0^1 p(r)dr$ 가 있었으나 §4 셀 17 행이 "mAP 표" 라고만 적어
수식이 통째로 사라졌고, 저자·감사자 어느 쪽도 "표는 있으니 명세 준수" 로 통과시킬 수 있었다.
**§2 의 각 수식에 "→ 셀 NN" 역참조를 붙이는 것**을 명세 스킬에 넣으면 검증자가 기계적으로 대조할 수 있다.

**[검증자 트리거 — 지정에 이유가 없으면 재량으로 뒤집힌다]**
명세가 특정 데이터·이미지·하이퍼파라미터를 지정할 때 **"왜 그것이어야 하는가" 를 한 줄로 함께 적는다** /
위치: `.claude/skills/lecture-spec-extraction/` /
근거: 이번 라운드 A-1. 리더가 Step 6 에 `data/sample.jpg` 를 지정한 진짜 이유는 "마스크가 잘 보인다" 만이
아니라 "`masks.data ≠ orig_shape` 이 되어 본문 서술이 출력으로 증명된다" 였는데, 후자가 명세에 적히지
않아 저자가 다른 원칙으로 갈아탔고 치명 결함이 나왔다.

**[검증자 트리거 — 로그의 인접성은 근거가 아니다. 라운드 3 에서 내가 직접 틀렸다]**
학습 로그의 지표 변화를 결함 근거로 쓰기 전에 (a) epoch 당 iteration 수 (b) 실제 적용 lr
(c) 스케줄 변화 지점을 확인하는 것으로는 **부족하다.** 두 사건이 로그의 가까운 자리에 보이는 것은
상관이며, **인과를 주장하려면 그 변수만 바꾼 실행이 필요하다** / 위치: 이 스킬의 방법론 문서 /
근거: 나는 소스에서 `close_mosaic` 발화 조건(`20-10=10` 0-based → 1-based 11)까지 확인하고
"epoch 12 하락과 일치한다" 를 노트북 해설 초안에 넣자고 제안했다. 리더가 `close_mosaic=0` 으로
그 변수만 제거해 돌리자 **하락이 똑같이 epoch 12 에서 났다.** 확인 비용은 10초였다.
**"소스로 확인했다" 는 감각이 검증을 멈추게 한다** — 기전을 확인하는 것과 그 기전이 이 현상의
원인인지 확인하는 것은 다른 일이다. 감사자가 같은 로그로 먼저 틀렸고, 나는 소스를 열었다는 이유로
한 겹 더 확신했을 뿐 더 옳지 않았다.

**[검증자 트리거 — 넘겨받는 쪽도 기억에는 면역이 없다]**
"동역학 해석은 감사자가 등급을 올리지 말고 검증자에게 넘긴다" 를 기본값으로 하되,
**넘겨받는 쪽에는 기억이 아니라 설치본 소스로 확인할 의무가 따라붙는다** 는 단서를 함께 둘 것 /
근거: 나는 "warmup 이 20 iteration 전 구간을 덮었다" 를 쓰려다 소스를 열어 막았다. 이 빌드의
`_get_warmup_iterations` 는 `round(min(3.0, 19) * 1)` = **3 iteration** 으로, 하한 100 을 두는
다른 버전과 다르다. 기억으로 썼으면 거짓 해설이 학생 노트북에 인쇄됐다.

**[규약 모순 — `CLAUDE.md` 내부에서 generator 규칙이 두 번 다르게 적혀 있다]**
"디렉터리 레이아웃" 표는 `gen/` 에 `p{NN}.py` 를 두라고 하고 프로필 `generator_glob` 도 `gen/*.py` 인데,
문서 말미 "Notebook Generator Scripts" 절은 **`_gen_p{NN}.py`** 형태를 쓰고 **"검증이 끝나면
generator 를 삭제하라"** 고 한다. 실제 저장소는 `gen/p01.py`~`p12.py` 로 전자를 따르며 삭제하지도 않는다 /
위치: 과목 `CLAUDE.md` — 말미 절을 레이아웃 표에 맞춰 정정 (사용자 승인 대상) /
근거: 이번 회차에서 저자·검증자 모두 레이아웃 표를 따랐으나, 말미 절을 먼저 읽은 다음 회차 저자는
generator 를 삭제해 **수정 이력과 재생성 능력을 잃는다.** 규약 문서 자체의 모순이므로 사람이 정한다.

**[프로필 항목 — `lecture_sources.authority` 는 있으나 적용 대상이 없다]**
`notebook-profile.json` 의 `authority: "slides"` 는 이 회차에서 **가리키는 대상이 존재하지 않는다.**
프로필에 "이 과목의 `lecture_notes/` 는 현재 Ch1 하나뿐이며, 대응 슬라이드가 없는 회차는 정합성 검증의
축 A′ 를 미검증으로 낸다" 는 문장을 `authority_note` 에 추가할 것을 제안한다 / 근거: 이번 라운드 전체.
사용자 확인 필요 — 슬라이드가 추가될 예정인지에 따라 문구가 달라진다.
