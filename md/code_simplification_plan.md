# 코드 단순화 작업 계획 — 다른 세션에서 이어서 실행할 것

> **이 문서 하나로 작업이 재개되도록 썼다.** 앞선 세션의 진단·실측을 옮겨 담았으므로 다시 조사하지 말 것.
>
> **기준(사용자 지시).** "최대한 직관적이고 간단하게 효과적인 실습자료".
> "구문이 깔끔한 것보다 직관적인 것이 좋다 — **기계공학도는 파이썬을 잘 하지 못한다**."
> 즉 줄 수를 줄이는 것이 목표가 아니다. **파이썬 숙련도를 요구하는 구문을 없애는 것**이 목표이며,
> 그 대가로 코드가 길어지고 반복되는 것은 허용된다 (`CLAUDE.md` "코드 복잡도", "가독성 > 간결성").

## 0. 작업 규칙 (모든 항목 공통)

1. **`.ipynb` 를 직접 편집하지 않는다.** `gen/p{NN}.py` 를 고치고 과목 루트에서 재생성한다.
   ```
   python gen/p06.py
   ```
2. 전 셀 실행 검증 (과목 루트에서):
   ```
   KMP_DUPLICATE_LIB_OK=TRUE jupyter nbconvert --to notebook --execute --inplace "Practice06_ML_General_Tips.ipynb"
   ```
3. 검사 2종:
   ```
   python .claude/scripts/markdown_budget.py "노트북.ipynb"                      # exit 1 이면 초과
   python ../.claude/skills/notebook-convention-audit/scripts/audit_notebook.py "노트북.ipynb"
   ```
4. 실행 후 **과목 루트 오염 확인**: `ls *.pt runs` 가 비어 있어야 한다 (ultralytics 회차 한정).
5. 수정으로 **숫자가 바뀌면 그 셀 뒤의 마크다운 해설도 함께 고친다.** 해설이 출력과 어긋나는 것이
   이 저장소에서 반복된 사고다.
6. 한 항목을 고칠 때마다 **그 노트북만** 재생성·재실행한다. 한꺼번에 몰아 고치지 말 것.

---

## 1. 묶음 A — 재실행이 가볍다 (권장 시작 지점)

### A-1. `Practice06_ML_General_Tips.ipynb` 셀 18 — 과목 전체에서 가장 과한 셀  **[완료]**

> 재실행 11초. 셀 41개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음, 루트 오염 0.
> 표 숫자는 전부 그대로다 (계산이 같다). 히스토그램 2패널 셀 + ROC 1패널 셀로 갈랐고,
> `auc_64` / `auc_2` 를 위 셀에서 한 번만 계산한다. Youden 점 표시와 `annotate` 는 버렸다.
> ROC 셀 앞에 캡션 마크다운 한 개(2문장)를 새로 넣었다.

`gen/p06.py` 264행 부근. 현재 한 셀 30줄에 다음이 겹쳐 있다.

- `panels = [(...4-튜플...), (...)]` 리스트를 만들고 `zip(axes, panels)` 로 언패킹
- 그 안에서 다시 `for name, values, colour in [(...), (...)]` 두 번째 튜플 리스트 순회
- `colour + '-'`, `colour + 'o'` 로 선 스타일 문자열 조립
- `annotate(..., textcoords='offset points', xytext=(12, -12))`
- 100자 초과 줄 5개

**바꿀 형태 — 두 셀로 쪼개고 루프를 없앤다.**

```python
# cell A: 점수 분포 두 장 (패널마다 다섯 줄, 그대로 두 번)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0, 1, 31)

axes[0].hist(score[y_test == 0], bins=bins, alpha=0.6, color='steelblue', density=True, label='true 0')
axes[0].hist(score[y_test == 1], bins=bins, alpha=0.6, color='crimson', density=True, label='true 1')
axes[0].axvline(0.5, color='k', linestyle='--', linewidth=1.5, label='default 0.5')
axes[0].axvline(youden, color='green', linewidth=1.5, label=f'Youden {youden:.2f}')
axes[0].set_title(f'64 pixels (AUC {auc_64:.3f})'); axes[0].set_xlabel('predicted score'); axes[0].legend(fontsize=8)

axes[1].hist(score_two_pixels[y_test == 0], ...)   # 같은 다섯 줄, 변수만 two_pixels 판
...

# cell B: ROC 한 장
false_positive_rate_64, true_positive_rate_64, thresholds_64 = roc_curve(y_test, score)
false_positive_rate_2, true_positive_rate_2, thresholds_2 = roc_curve(y_test, score_two_pixels)

plt.figure(figsize=(5.5, 4.5))
plt.plot(false_positive_rate_64, true_positive_rate_64, 'b-', linewidth=2, label=f'64 pixels (AUC {auc_64:.3f})')
plt.plot(false_positive_rate_2, true_positive_rate_2, 'r--', linewidth=2, label=f'2 pixels (AUC {auc_2:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random (AUC 0.5)')
...
```

- `auc_64`, `auc_2` 는 위 셀에서 `roc_auc_score` 로 **한 번만** 계산해 변수에 담는다
  (현재는 f-string 안에서 반복 호출한다).
- Youden 점 표시(`markerfacecolor='none'` + `annotate`)는 **버린다.** 곡선과 임계값 표만으로 충분하다.

### A-2. `Practice06` 셀 17 — 금지된 helper function  **[완료]**

> A-1 과 같은 재생성으로 처리. 함수를 없애고 모델마다 한 셀씩(현재 셀 17·18) 같은 8줄을 반복한다.
> 출력 임계값(0.053 / 0.809 / 0.177 / 0.191) 은 이전과 동일하다.

`gen/p06.py` 245행. `def best_thresholds(y_true, values)` 가 값 두 개를 튜플로 반환하고, 내부에
임계값 전체를 도는 comprehension 이 있다. `CLAUDE.md` "Model Definition Style" 은 `train`/`evaluate`
외의 helper 를 금지한다.

**바꿀 형태 — 함수를 없애고 두 번 그대로 쓴다.**

```python
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, score)

# Youden's J is defined as argmax(TPR - FPR), so the formula is the code
youden = thresholds[np.argmax(true_positive_rate - false_positive_rate)]

# the max-F1 threshold needs one F1 score per candidate threshold
f1_values = []
for threshold in thresholds:
    f1_values.append(f1_score(y_test, (score >= threshold).astype(int), zero_division=0))
best_f1 = thresholds[np.argmax(f1_values)]
```

같은 8줄을 `score_two_pixels` 로 한 번 더 쓴다 (`youden_two_pixels`, `best_f1_two_pixels`).

### A-3. `Practice06` 셀 19 — 이중 루프로 표 만들기  **[완료]**

> A-1 과 같은 재생성으로 처리 (현재 셀 22). 표 6행의 숫자는 이전과 동일하다.

현재 `for name, values, youden_at, f1_at in panels:` 안에서 다시
`for label, threshold in [...]` 를 돌며 dict 를 쌓는다.

**바꿀 형태 — 중첩을 없애고 6행을 평평한 리스트로 적은 뒤 한 겹 루프.**

```python
threshold_rules = [
    ('64 pixels', score, 'default 0.5', 0.5),
    ('64 pixels', score, 'Youden', youden),
    ('64 pixels', score, 'max F1', best_f1),
    ('2 pixels', score_two_pixels, 'default 0.5', 0.5),
    ('2 pixels', score_two_pixels, 'Youden', youden_two_pixels),
    ('2 pixels', score_two_pixels, 'max F1', best_f1_two_pixels),
]

rows = []
for name, values, rule, threshold in threshold_rules:
    predicted = (values >= threshold).astype(int)
    rows.append({'model': name, 'rule': rule, 'threshold': threshold,
                 'accuracy': accuracy_score(y_test, predicted),
                 'precision': precision_score(y_test, predicted, zero_division=0),
                 'recall': recall_score(y_test, predicted),
                 'f1': f1_score(y_test, predicted, zero_division=0)})
display(pd.DataFrame(rows).round(3))
```

### A-4. `Practice05_Unsupervised_Learning.ipynb` 셀 22 — 튜플 리스트 순회 + ARI 3회 재계산  **[완료]**

> 재실행 8초. 셀 32개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음, 루트 오염 0.
> ARI 는 `ari_kmeans` / `ari_dbscan` 로 한 번만 계산한다. 출력 숫자(0.2542 / 1.0000) 는 그대로.

`gen/p05.py` 362~372행. 제목 f-string 안에서 `adjusted_rand_score` 를 **세 번** 다시 부른다.

```python
ari_kmeans = adjusted_rand_score(y_moons, kmeans_moons)
ari_dbscan = adjusted_rand_score(y_moons, dbscan_moons)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', s=25, alpha=0.85)
axes[0].set_title('True structure')
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons, cmap='viridis', s=25, alpha=0.85)
axes[1].set_title(f'K-means (ARI {ari_kmeans:.2f})')
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_moons, cmap='viridis', s=25, alpha=0.85)
axes[2].set_title(f'DBSCAN (ARI {ari_dbscan:.2f})')
for panel_axis in axes:
    panel_axis.set_xlabel('Feature 1'); panel_axis.set_ylabel('Feature 2'); panel_axis.grid(alpha=0.3)
```

위쪽 `print` 두 줄도 새 변수(`ari_kmeans`, `ari_dbscan`)를 쓰도록 함께 고친다.

### A-5. `Practice05` 셀 16 — `- (1 if -1 in labels else 0)` 관용구  **[완료]**

> A-4 와 같은 재생성으로 처리. 세 자리(셀 16 루프, 셀 16 요약 print, 셀 22 print) 전부
> `cluster_ids = set(...)` + `discard(-1)` 로 바꿨다.
> **출력 한 줄이 바뀐다** — `clusters : [-1, 0, 1]   (-1 means noise)` →
> `clusters : 2   (label -1 is noise, not a cluster)`.
> 레이블 목록을 그대로 찍으면 numpy 2.x 에서 `[np.int64(-1), ...]` 로 나오고, 그것을 막으려고
> 넣었던 `sorted(int(v) for v in ...)` 가 바로 이 항목이 지적한 comprehension 이었다.
> 클러스터 개수는 바로 위 sweep 과 아래 `cluster sizes` 가 이미 보여주므로 정보 손실은 없다.
> 이 줄을 참조하는 마크다운은 없다 (셀 17 은 "two clusters" 라고만 쓴다).

두 군데(루프 안, 셀 22 print)에 같은 표현이 있다.

```python
cluster_ids = set(labels_at_eps)
cluster_ids.discard(-1)          # -1 marks noise, not a cluster
n_clusters = len(cluster_ids)
```

f-string 안의 `sorted(int(v) for v in set(dbscan_labels))` 도 위 `cluster_ids` 를 만들어 놓고
`sorted(cluster_ids)` 로 바꾼다.

### A-6. `Practice04_ML_Models.ipynb` 셀 22 — 금지된 `sort_values`  **[완료]**

> 재실행 7초. 셀 34개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음, 루트 오염 0.
> `pd.Series` 를 거치지 않고 `feature_importances_` 배열과 `np.argsort` 만 쓴다.
> 출력은 네 줄의 값(0.437 / 0.431 / 0.116 / 0.015)이 그대로이고 `dtype: float64` 줄만 사라졌다.

`gen/p04.py` 330·340행. `CLAUDE.md` "코드 복잡도" 가 `sort_values` 를 명시적으로 금지하고,
`np.argsort` 는 "선을 순서대로 그리기 위한" 용도로 예외 허용한다.

```python
importances = model_random_forest.feature_importances_
feature_names = np.array(iris.feature_names)
order = np.argsort(importances)          # ascending, so the longest bar ends up on top

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(feature_names[order], importances[order], color='steelblue')
...

# 아래 print 는 내림차순으로 다시 정렬하지 말고 order 를 뒤집어 쓴다
for index in order[::-1]:
    print(f'{feature_names[index]:<20s} {importances[index]:.3f}')
```

### A-7. `Practice12_Object_Detection_and_Segmentation.ipynb` 셀 25·31 — `np.ma.masked_where`  **[완료]**

> 재실행 9초(추론만, GPU). 셀 35개, 에러 0, `markdown_budget` 0 over cap,
> `audit_notebook` 후보 없음, 루트 오염 0 (`*.pt`·`runs` 없음).
>
> - 셀 25 는 **네 패널** — `입력 | 손으로 그린 박스 | 마스크 한 장(gray) | result.plot()`.
>   계획서의 세 패널에 마스크 한 장 패널을 더한 형태다.
> - 마스크 패널은 `masks.data[0]`(첫 chair) 가 아니라 `masks.data[2]`(potted plant) 를 쓴다.
>   index 0 은 화면 아래에 잘린 의자라 흰 덩어리로만 보여 "0/1 격자" 가 눈에 들어오지 않았다.
>   그림을 뽑아 확인하고 바꿨다. 제목은 `retina_result.names[...]` 로 클래스명을 넣어
>   검출 순서가 바뀌어도 패널이 자기를 설명하게 했다.
> - 셀 31 은 `second_segmentation.plot()[:, :, ::-1]` 한 줄. `second_instance_map` 블록 삭제.
> - 손으로 `Rectangle` 을 그리는 코드는 셀 25·31 양쪽 다 남겼다.
> - 해설 마크다운 두 개를 그림에 맞게 고쳤다 — Step 6 도입부("Painting each object with its own
>   number ..." → 세 번째 패널 설명), Step 6 뒤 해설("The middle panel ... Overlapping chairs keep
>   their own colours" → 박스와 마스크의 차이 + `result.plot()` 언급).

`gen/p12.py` 358행(Step 6 세 패널), 432행(Step 8 두 번째 이미지).
`instance_map` 을 0으로 채워 인스턴스 번호를 덧칠한 뒤 masked array 로 겹쳐 그리고 있다.
**Practice13 에서는 이미 걷어낸 패턴이며, 여기만 남았다.**

- Step 6 (셀 25): 세 패널을 `[원본 | 손으로 그린 박스 | result.plot()]` 로 바꾼다.
  **마스크 한 장을 그대로 보여주는 패널을 하나 넣는 편이 이 회차의 주제에 맞다** —
  `axes[?].imshow(retina_result.masks.data[0].cpu().numpy(), cmap='gray')` 로 "마스크는 0/1 격자" 를
  눈으로 보이고, 전체 겹치기는 `result.plot()` 에 맡긴다.
- Step 8 (셀 31): `second_instance_map` 블록을 지우고 `second_segmentation.plot()[:, :, ::-1]` 한 줄.
- **손으로 `Rectangle` 을 그리는 코드는 남긴다.** 이 회차의 주제가 "결과 구조를 열어 덧그리기" 다.
- 그림이 바뀌므로 셀 25·31 뒤의 해설 마크다운("Overlapping chairs keep their own colours ...")을
  실제 그림에 맞게 고칠 것.

### A-8. `Practice04` 셀 4 — 금지된 `value_counts`  **[완료]**

> 계획서에 없던 항목. A-6 작업 중 발견해 사용자 승인을 받고 처리했다. 재실행 6초.
> 셀 34개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음.
>
> `print(iris_frame['label'].value_counts().sort_index())` (앞의 빈 `print()` 와 헤더 줄까지 3줄)
> → `print(f'samples per class: {np.bincount(iris.target)}')` 한 줄.
> 출력이 pandas Series 6줄(`label` / `0  50` / ... / `Name: count, dtype: int64`) 에서
> `samples per class: [50 50 50]` 한 줄로 바뀐다. 참조하는 마크다운은 없다.
>
> 이로써 `gen/p04.py` 에 `groupby`·`sort_values`·`value_counts`·`apply(lambda ...)` 가 하나도 없다.

**묶음 A 재실행 비용:** Practice04·05·06 은 sklearn 계산뿐이라 각 1~2분, Practice12 는 추론만이라 1분.

**실측 (묶음 A 전 항목 완료).** 예상보다 훨씬 가벼웠다.

| 노트북 | 항목 | 재실행 | 결과 |
|---|---|---|---|
| Practice06 | A-1·A-2·A-3 | 11초 | 셀 41, 에러 0, 0 over cap, 후보 없음 |
| Practice05 | A-4·A-5 | 8초 | 셀 32, 에러 0, 0 over cap, 후보 없음 |
| Practice04 | A-6 | 7초 | 셀 34, 에러 0, 0 over cap, 후보 없음 |
| Practice12 | A-7 | 9초 | 셀 35, 에러 0, 0 over cap, 후보 없음, 루트 오염 0 |

---

## 2. 묶음 B — 재실행이 무겁다 (학습이 들어간다)

먼저 **한 번 재실행해 소요 시간을 재고**, 그 값을 이 문서에 적은 뒤 진행할 것.

**실측 (수정 전 baseline, GPU).** 셋 다 예상보다 가볍다. Practice10 만 3분대다.

| 노트북 | baseline 재실행 | 수정 후 재실행 |
|---|---|---|
| Practice08 | 17초 | 23초 |
| Practice09 | 11초 | 11초 |
| Practice10 | 204초 | 207초 |

(Practice08 의 23초는 Practice10 재실행과 겹쳐 돌린 값이라 부풀려져 있다.)

### B-1. `Practice10_CNN_Pipeline.ipynb` 셀 13 — `getattr` 로 블록 순회  **[완료]**

> 재실행 207초. 셀 32개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음.
> 블록마다 두 줄(호출 + print)씩 다섯 번 펼쳐 적었다. shape 출력 6줄은 전과 동일하다.

`gen/p10.py` 351행.

```python
for name in ['block1', 'block2', 'block3', 'block4', 'block5']:
    x = getattr(model_vgg, name)(x)
```

→ 다섯 번 그대로 적는다. `CLAUDE.md` "Model Definition Style" 도 "for 문으로 옵션을 순회하지 않고
코드 반복" 을 정하고 있다.

```python
x = torch.randn(1, 3, 96, 96).to(device)
print(f'input        : {tuple(x.shape)}')
x = model_vgg.block1(x)
print(f'after block1 : {tuple(x.shape)}')
x = model_vgg.block2(x)
print(f'after block2 : {tuple(x.shape)}')
...
```

### B-2. `Practice08` 셀 13 / `Practice10` 셀 7 — 한 셀에 함수 두 개, 57줄  **[완료]**

> Practice08 재실행 23초(셀 36개), Practice10 은 B-1 과 같은 재생성.
> 양쪽 다 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음.
> 함수 본문은 한 글자도 바꾸지 않았다. `evaluate` 셀 → `train` 셀 순서.
> 각 셀 앞 마크다운 한 줄은 **원래 코드 첫 줄에 있던 `#` 주석을 옮긴 것**이다
> (마크다운과 주석이 같은 말을 두 번 하지 않게).

`gen/p08.py` 228행, `gen/p10.py` 174행. 함수 **내용은 규약이 정한 형태이므로 바꾸지 않는다**
(`.claude/references/code-patterns.md` §7). 셀만 둘로 나눈다.

- 먼저 `evaluate` 셀, 다음 `train` 셀 (읽는 순서가 호출 순서와 맞는다).
- `code(r"""...""")` 하나를 둘로 쪼개면 되고, 마크다운도 각 셀 앞에 한 줄씩 붙인다.

### B-3. `Practice09_Dataloader_for_Image.ipynb` 셀 6 — 38줄에 다섯 가지  **[완료]**

> 재실행 11초. 셀 33개, 에러 0, `markdown_budget` 0 over cap, `audit_notebook` 후보 없음.
> 출력 숫자는 전부 그대로 (train_mean/std, batch shape, 4×4 픽셀 블록).
> 계획대로 세 셀 — ① float32 + 통계 + 표준화 ② 텐서/Dataset/DataLoader + batch shape
> ③ 4×4 픽셀 미리보기. Step 2 의 "Read the 4×4 block against the one printed in Step 1" 이
> 이 세 번째 셀을 가리키므로 Step 1 안에 남겼다.
> `unnormalize` 는 처음 쓰이는 자리(Step 2 의 이미지 그리드) 바로 앞 셀로 내렸고,
> 그것을 소개하던 문장("Standardized pixels are negative as often as positive ...")도
> Step 1 도입부에서 그 자리로 함께 옮겼다.

`gen/p09.py` 119행. 스케일링 → 통계 → 표준화 → `unnormalize` 정의 → 텐서 변환 → Dataset/DataLoader
→ 배치 미리보기가 한 셀에 있다. `CLAUDE.md` "한 셀에 한 가지" 위반.

세 셀로 나눈다.

1. `float32` 변환 + `train_mean`/`train_std` + 표준화 (+ print)
2. `(N,H,W,C) → (N,C,H,W)` 변환 + `TensorDataset` + `DataLoader` + 배치 shape print
3. `unnormalize` 정의 + 픽셀 미리보기 (그림에서 처음 필요한 자리 바로 앞)

`unnormalize` 는 `CLAUDE.md` 가 금지하는 helper factory 가 아니라 표시 변환이므로 **유지한다.**

---

## 3. 손대지 말 것 (정규식에 걸리지만 정당한 것)

다음 세션이 다시 조사하지 않도록 판정 결과를 남긴다.

| 걸린 것 | 판정 |
|---|---|
| `Practice01` 의 `lambda`, `*args/**kwargs`, comprehension, `value_counts` | **정당.** 그 회차가 파이썬 문법 자체를 가르친다 |
| `device = 'cuda' if torch.cuda.is_available() else 'cpu'` (6개 노트북) | **정당.** 관용구 |
| `(output.argmax(1) == Y_batch).sum().item()` | **정당.** torch 표준 표현 |
| `Practice10` 의 `VGG11` 클래스 63줄 | **정당.** 블록을 펼쳐 쓴 것이 목적 |
| `Practice07` 의 optuna `objective` if/elif | **정당.** optuna API 가 요구하는 구조 |
| `Practice07` 셀 50 `pd.DataFrame([{...},{...}])` | **정당.** 규약이 권하는 형태. 단 100자 초과 줄만 정리하면 좋다 |
| `model.eval()` 4건이 `getattr` 로 잡힌 것 | **오탐** (정규식이 `eval(` 을 잡았다) |
| `Practice10` 셀 17 `[p for p in model.parameters() if p.requires_grad]` | 경미. 고칠 거면 얼린 층을 뺀 `model_resnet.fc.parameters()` 로 |
| `Practice03` 셀 18 `np.c_[...]`, 체이닝 | 경미. 여유가 있으면 `np.column_stack` + 두 줄로 분리 |

## 4. 진행 순서와 완료 조건

1. A-1 ~ A-3 (`Practice06`) → 재생성 → 실행 → 검사 2종 → 해설 문구 대조
2. A-4 ~ A-5 (`Practice05`), A-6 (`Practice04`), A-7 (`Practice12`) — 각각 같은 절차
3. 시간을 재고 B-1 ~ B-3 진행
4. 끝나면 이 문서의 각 항목에 **[완료]** 와 실측 재실행 시간을 적는다

완료 조건: 해당 노트북이 **에러 0으로 전 셀 실행**, `markdown_budget.py` **0 over cap**,
`audit_notebook.py` **후보 없음**, 루트 오염 0.

### 진행 결과 — A-1 ~ B-3 전부 [완료] (+ 계획서에 없던 A-8)

일곱 개 노트북 모두 위 완료 조건을 만족한다. 과목 전체 `markdown_budget.py` 도 **0 over cap** 이다.

| 노트북 | 항목 | 셀 | 재실행 |
|---|---|---|---|
| Practice04 | A-6·A-8 | 34 | 6초 |
| Practice05 | A-4·A-5 | 32 | 8초 |
| Practice06 | A-1·A-2·A-3 | 41 | 11초 |
| Practice08 | B-2 | 36 | 23초 |
| Practice09 | B-3 | 33 | 11초 |
| Practice10 | B-1·B-2 | 32 | 207초 |
| Practice12 | A-7 | 35 | 9초 |

**출력이 바뀐 곳은 세 군데뿐이고 모두 의도한 것이다.**

| 노트북 | 바뀐 것 | 해설 마크다운 |
|---|---|---|
| Practice04 | `dtype: float64` 줄이 사라짐 (값 4개는 동일), 클래스 개수 표가 `[50 50 50]` 한 줄로 | 손댈 것 없음 |
| Practice05 | `clusters : [-1, 0, 1]` → `clusters : 2` | 참조하는 마크다운 없음 |
| Practice12 | Step 6·8 그림 구성 | 두 셀 고침 (A-7 참조) |

**Practice10 의 VGG 숫자는 재실행마다 흔들린다. 코드와 무관하다 — 세 번 돌려 확인했다.**

| 실행 | VGG val_acc | VGG test_acc | ResNet val/test |
|---|---|---|---|
| 수정 전 | 0.7190 | 0.7140 | 0.8480 / 0.8350 |
| 수정 후 | 0.6915 | 0.7035 | 0.8480 / 0.8350 |
| 수정 후, 코드 그대로 한 번 더 | 0.7180 | 0.7230 | 0.8480 / 0.8350 |

세 번째 행이 판정의 근거다. **코드를 한 글자도 바꾸지 않고 다시 돌렸는데 VGG 값만 또 달라졌다.**
GPU 학습의 비결정성이며 코드 변경 탓이 아니다 —
`getattr` 루프와 펼쳐 쓴 형태가 RNG 상태를 똑같이 남긴다는 것도 따로 확인했다
(블록에 Dropout 을 넣고 두 형태를 돌려 이후 `torch.randn` 값이 일치함을 봤다).
세 실행 모두에서 ResNet 쪽 숫자가 소수 넷째 자리까지 같은 이유는
모델 생성 직전에 seed 를 다시 잡기 때문이다 (`CLAUDE.md` "Reproducibility").
**Practice10 의 마크다운은 숫자를 인용하지 않으므로 고칠 것이 없다** —
"VGG 곡선이 아직 오르는 중", "같은 15 epoch 안에서 따라잡지 못했다" 는 서술은 두 값 모두에서 참이다.
숫자를 인용하고 싶어지면 그때 VGG 값이 흔들린다는 것을 기억할 것.

## 5. 이 작업과 별개로 남아 있는 것

- **`build/pcb_defect_640.zip` (50.6MB) 을 공개 URL 에 올려야 한다.** `HW/HW_PCB_Defect_Detection.ipynb`
  가 `https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/main/data/pcb_defect_640.zip`
  를 받도록 생성돼 있고, 업로드 전까지 그 한 줄만 미검증이다. 자세한 내용은
  `md/p12_workspace/06_restructure_record.md` §4.
- CPU 런타임 학습 시간 미측정 (GPU 기준만 노트북에 적혀 있다).
- `data/coco8`, `data/coco8-seg`, `data/coco8*.yaml` 은 어느 노트북도 쓰지 않는다. 삭제 가능.

## 6. 배경 문서

| 문서 | 내용 |
|---|---|
| `md/p12_workspace/05_revision_plan.md` | Practice12 개정 진단 (왜 fine-tuning 회차를 갈랐는지) |
| `md/p12_workspace/06_restructure_record.md` | Practice12/13/HW 재구성 결과와 실측값. **세 노트북의 현재 기준** |
| `md/p12_workspace/00_env_probe.md` | ultralytics 경로 오염 6종과 해법 (가중치·`project` 절대경로·`amp=False`) |
| `.claude/references/code-patterns.md` | 반복 코드 형태. §7 이 `train`/`evaluate` 시그니처를 고정한다 |
| `CLAUDE.md` | 규약 본문. 충돌 시 이 문서가 우선 |
