# Project Guidelines — 01_AI-ME_Graduate

Advanced Artificial Intelligence for Mechanical Engineering (아주대학교 기계공학과 대학원, 2026 Fall) 의
강의 노트북 저장소. 이 문서가 이 과목의 **규약과 그 이유**를 담는다.

기계 설정(감사 스크립트가 읽는 것)은 `.claude/notebook-profile.json`, 반복되는 코드 형태는
`.claude/references/code-patterns.md` 에 있다. 셋이 어긋나면 **이 문서가 우선**이다.

## 디렉터리 레이아웃

과목 폴더는 **독립적으로 완결된다.** 다른 과목을 참조하지 않고 이 폴더만으로 노트북을 만들고
검증할 수 있어야 한다. 데이터셋도 과목마다 각자 갖는다.

대신 **이름과 자리는 모든 과목이 같게** 둔다. 같은 도구가 어느 과목에서든 같은 명령으로 돌고,
과목 간 비교가 같은 경로에서 성립하며, 새 과목은 이 골격을 복사해 시작하기 위해서다.

| 자리 | 내용 |
|------|------|
| 루트 | `Practice{NN}_{Name}.ipynb` 와 `CLAUDE.md`. **그 외 파일을 두지 않는다** |
| `gen/` | 노트북 생성기 `p{NN}.py`, `hw{NN}.py` |
| `tools/` | 데이터 준비 등 보조 스크립트 |
| `.claude/` | `notebook-profile.json`, `references/`, `scripts/`, `fixtures/` |
| `HW/`, `HW/Answer/` | 과제 배포본과 정답본 |
| `data/` | 이 과목이 쓰는 데이터셋 |
| `lecture_notes/` | 강의 슬라이드 PDF |
| `md/` | 작업 문서 |
| `build/` | 로그·임시 플롯 등 **지워도 되는 것.** 커밋하지 않는다 |

**generator 는 과목 루트에서 실행한다** — `python gen/p08.py`. 출력 경로가 루트 기준이라
`gen/` 안에서 실행하면 노트북이 엉뚱한 자리에 생긴다. 검사 스크립트도 마찬가지다.

- **파일명과 디렉터리명에 공백을 쓰지 않는다.** 명령줄과 스크립트에서 매번 따옴표가 필요해지고,
  따옴표를 빠뜨린 자리가 조용히 다른 경로를 가리킨다
- 다른 과목의 회차를 옮겨 왔다면 **generator 첫 줄 주석에 출처를 남긴다.** 개정하면서 회차 대응이
  바뀌므로 그 기록이 없으면 추적이 끊긴다. 노트북 본문에는 쓰지 않는다 (No Cross-references)

## 강의자료

- `lecture_notes/*.pdf` — **원본이자 최종 권위.** 표기·수식·하이퍼파라미터가 갈리면 PDF 를 따른다
- `md/lectures_and_formulas.md`, `md/practice_outline_ref.md` — **참고용 요약본.**
  학부 과목(`02_ML_Undergraduate`)에서 가져온 문서라 회차 번호(`Practice NN`, `Lec NN`)와
  일부 컨벤션이 이 과목과 다르다. 수식·notation·변수명 컨벤션만 참고하고,
  PDF 와 충돌하면 임의로 고르지 말고 **병기해 사람에게 올린다**

## Target Audience

- **기계공학 대학원생** — 전공 수식(선형대수·미적분)에 익숙하고 연구 목적으로 코드를 읽는다
- Python 문법 자체는 설명하지 않는다. 대신 **수식의 각 항이 코드의 어느 줄인지**를 드러낸다
- 코드는 여전히 **가독성 > 간결성**. 한 줄로 압축하기보다 수식 순서대로 풀어 쓴다
- 비교 실험은 **2개 항목(A vs B)** 을 기본으로 하되, 같은 문제를 여러 도구로 푸는 비교
  (수식 유도 / autograd / optimizer / sklearn) 는 예외적으로 3~4개까지 허용한다.
  이 경우 모든 방법이 **같은 데이터·같은 초기값·같은 학습률·같은 반복 수**를 써야 한다

## Language — English only

**이 과목은 영어 강의다.** 학생에게 배포되는 노트북의 모든 텍스트는 영어로 쓴다.

- **마크다운 셀: 영어**
- **코드 주석(`#`): 영어**
- **`print()` 로 출력하는 문자열: 영어**
- **차트 텍스트(title, label, legend, annotation): 영어.**
  차트는 특히 예외가 없다 — matplotlib 기본 폰트에서 한글은 두부(tofu) 문자로 렌더링된다
- 변수명·파일명도 영어

**노트북 안에 한글을 한 글자도 남기지 않는다.** 감사 체크포인트 `hangul` 이 코드·마크다운
모든 셀을 검사한다.

적용 범위는 **학생 배포물(`.ipynb`)** 이다. 이 문서, `notebook-profile.json`,
`.claude/references/` 같은 작업 문서와 generator script 의 주석은 한국어로 둔다.
LaTeX 수식 기호는 언어와 무관하므로 그대로 쓴다.

## Notation — 이 과목의 수식 컨벤션

강의 PDF (`lecture_notes/`) 의 표기를 따른다. 학부 과목과 갈리는 지점이 있으므로 주의한다.

| 기호 | 의미 | 코드 |
|------|------|------|
| $\mathbf{X}$ | augmented 설계행렬 $(N, d+1)$, 0번 열 = 1 | `X` |
| $\mathbf{w}$ | 가중치 열벡터 $(d+1, 1)$ | `w` (`theta` 금지) |
| $\hat{\mathbf{y}}$ | 예측 $(N, 1)$ | `y_hat` |
| $\mathbf{E}$ | 잔차 $\mathbf{y} - \hat{\mathbf{y}}$ | `E` |
| $\partial J/\partial \mathbf{w}$ | 그래디언트 ($\nabla J$ 는 쓰지 않음) | `dJ` |
| $\rho$ | 학습률 | `rho` (`alpha`, `eta`, `step_size` 금지) |
| $N$, $d$ | 샘플 수, 입력 특징 수 | `N`, `d` |

### 손실함수에 $1/2$ 를 붙인다 — 이 과목의 결정

$$J(\mathbf{w}) = \sum_{n=1}^{N}(\hat{y}_n - y_n)^2 / 2 = \tfrac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$$

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{X}^T\mathbf{y}
= \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y}) = -\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}}) = -\mathbf{X}^T\mathbf{E}$$

**이유.** $1/2$ 없이 $J = \|\mathbf{E}\|_2^2$ 로 두면 엄밀한 미분값은 $-2\mathbf{X}^T\mathbf{E}$ 이고,
상수 2 를 학습률에 흡수시켜야 $-\mathbf{X}^T\mathbf{E}$ 가 된다. 그런데 이 과목의 노트북은
**손으로 유도한 기울기를 `loss.backward()` 결과와 나란히 놓고 같은 값인지 눈으로 확인**한다.
autograd 는 코드에 적힌 $J$ 를 그대로 미분하므로, 2 를 흡수시킨 상태에서는 두 값이 2배 차이 난다.
$J$ 정의에 $1/2$ 를 넣으면 흡수할 상수가 애초에 없고, 수식·NumPy·autograd 세 값이 정확히 일치한다.

- 코드의 손실도 같은 정의를 쓴다: `loss = ((y_hat - y) ** 2).sum() / 2`
- `nn.MSELoss(reduction='sum')` 을 쓰면 $\|\mathbf{E}\|_2^2$ 이므로 **`/ 2` 를 곱해 맞춘다**
- 학부 요약본(`md/lectures_and_formulas.md`) 은 "상수 2 를 $\rho$ 에 흡수" 라고 쓰여 있다.
  **이 과목은 그 규칙을 따르지 않는다** — 위 이유로 PDF 쪽이 맞다

### 잔차의 부호

$$\mathbf{E} = \mathbf{y} - \hat{\mathbf{y}}, \qquad \frac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T\mathbf{E},
\qquad \mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}}$$

강의 PDF 의 일부 슬라이드는 같은 것을 $\mathbf{E} = \hat{\mathbf{y}} - \mathbf{y}$,
$\partial J/\partial \mathbf{w} = \mathbf{X}^T\mathbf{E}$ 로 적는다. 값은 동일하다.
**노트북은 PDF 의 코드 블록과 같은 형태(`E = y - y_hat`, `dJ = -X.T @ E`)로 통일**하고,
두 표기가 같은 식이라는 점만 마크다운 한 줄로 남긴다.

### 행렬 연산

- 행렬곱은 **항상 `@`**. `np.matrix` / `np.asmatrix` / `*` 를 이용한 행렬곱은 쓰지 않는다
  (`*` 가 원소곱인지 행렬곱인지 학생이 매번 확인해야 한다). 역행렬은 `np.linalg.inv`
- augmented 행렬: `X = np.hstack([x**0, x])` (강의 PDF 표기) 또는 `np.hstack([np.ones((N, 1)), x])`
- 벡터는 **2D 열배열** `(N, 1)` / `(d+1, 1)` 기준. 1D `(N,)` 와 값은 같지만 shape 출력이 달라지므로
  한 노트북 안에서 섞지 않는다

## 설명 분량 — 마크다운 셀

측정해서 정한 기준이다. 이 과목 노트북의 마크다운 셀 103개를 재면 중앙값 50단어, p70 79, p90 126,
최대 237이었다. **평균은 문제가 아니고 꼬리가 문제다.** 그리고 길어지는 자리는 언제나 같다 —
Summary, 섹션 도입부, 그림 뒤 해설.

| 대상 | 상한 |
|------|------|
| 노트북 첫 셀 (제목 + 개요) | 150 단어 |
| Summary | 100 단어 |
| 그 외 마크다운 셀 | **120 단어** (하드 상한) |
| 〃 | 80 단어를 넘으면 표로 바꾸거나 쪼갤 것 (소프트) |

### 줄글은 따로 센다

전체 단어 수만 재면 표를 늘려 상한을 지키면서 줄글은 그대로 둘 수 있다. 실제로 그 일이 일어나서
**표·헤더·수식을 뺀 산문 단어 수**를 따로 잰다. 이 과목 노트북 171개 셀 기준 중앙값 45단어,
p70 62, p90 82.

| 대상 | 상한 |
|------|------|
| 셀당 산문 (표·헤더·수식 제외) | **80 단어** (하드) |
| 〃 | 60 단어를 넘으면 불릿·표로 바꾸거나 버릴 것 (소프트) |
| 한 문단 | **3 문장** (하드) |
| 한 줄 | **1 문장** (하드) |

**문장마다 줄을 바꾼다.** 마크다운에서 줄바꿈 하나는 렌더링을 바꾸지 않으므로 학생 화면은 그대로고,
바뀌는 것은 셋이다 — 쓰는 사람이 문장 수를 눈으로 세게 되고, diff 가 문장 단위로 떨어지고,
문단이 길어지는 순간 스크립트가 잡는다. **읽는 화면에서 줄을 나누고 싶으면 빈 줄로 문단을 쪼개거나
불릿으로 만든다.** 줄바꿈 하나로는 그렇게 되지 않는다.

검사:

```
python .claude/scripts/markdown_budget.py "노트북.ipynb"     # 초과가 있으면 exit 1
python .claude/scripts/markdown_budget.py                    # 과목 전체
python .claude/scripts/markdown_budget.py --stats            # 기준을 다시 잴 때
```

정규식 감사(`audit_notebook.py`)로는 셀 단위 문장을 셀 수 없어 별도 스크립트로 둔다.
노트북을 만들거나 고친 뒤 이 스크립트를 함께 돌린다.

### 길어지지 않게 하는 규칙

- **한 셀에 한 가지.** 두 가지를 말하고 있으면 쪼개거나 하나를 버린다
- **산문보다 표.** 항목이 셋 이상 나열되면 표가 거의 항상 더 짧고 잘 읽힌다
- **문단은 두 문장까지.** 세 문장째가 붙으면 대개 앞 문장을 다시 말하고 있다
- **코드 뒤 해설은 "이 출력이 무엇을 말하는가" 만.** 개념을 다시 설명하지 않는다. 3줄 이내
- **같은 말을 두 번 하지 않는다.** 도입부에서 한 말을 마무리에서 되풀이하지 않고,
  표와 산문이 같은 내용이면 표만 남긴다
- **불릿은 3개까지.** 넘으면 표로 바꾸거나 덜 중요한 것을 버린다
- **Summary 에 새 내용을 넣지 않는다.** 본문에 없던 설명이 요약에서 처음 나오면 자리가 잘못된 것이다

### 왜 숫자로 박아 두는가

설명은 고칠 때마다 늘어난다. 지적을 반영할 때 문장을 **덧붙이지 지우지는 않기** 때문이다.
상한이 숫자로 없으면 개정마다 조금씩 부풀고, 실제로 이 과목에서 그 일이 반복됐다.
"짧게 쓰라" 는 지시는 다음 개정에서 잊히지만, exit 1 로 떨어지는 스크립트는 잊히지 않는다.

## 코드 복잡도 — ML 이 아닌 구문은 배제

노트북의 코드는 ML/DL 을 보여주는 데 쓴다.
데이터를 다듬는 pandas 관용구는 그 자체가 학습 목표가 아니면 쓰지 않는다.

| 금지 | 대신 |
|------|------|
| `groupby`, `sort_values`, `value_counts`, `rename`, `agg`, `pivot` | 라이브러리가 이미 준 값 |
| `apply(lambda ...)`, 중첩 comprehension | 평범한 `for` 문 |
| `cv_results_` 를 다시 집계 | `best_params_`, `best_score_`, `best_estimator_` |

- **라이브러리가 답을 이미 주면 그걸 쓴다.** `best_params_`, `best_loss_per_estimator` 처럼
  이름이 붙어 나오는 값을 두고 같은 숫자를 다시 계산하지 않는다
- 항목별 최고값이 필요하면 대개 **항목마다 한 번씩 도는 `for` 문**이 더 짧고 읽기 쉽다
- 표로 보여줄 것이 있으면 `pd.DataFrame([{...}, {...}])` 처럼 **행을 그대로 적는다**

### 예외

플롯 축 범위(`.min()`, `.max()`), 선을 x 순서로 그리기 위한 `np.argsort`,
그리고 **지표의 정의 자체인 경우** — 예를 들어 Youden 임계값은 정의가 $\arg\max(TPR-FPR)$ 이므로
`np.argmax(true_positive_rate - false_positive_rate)` 가 곧 수식이다.

### 왜

학생이 코드를 읽다 멈추는 지점이 `groupby(...).max().sort_values(...)` 라면,
그 셀은 ML 을 가르치는 데 실패하고 pandas 를 가르친 것이다.
대상 독자는 Python 문법이 아니라 방법을 배우러 왔다.

## No Emojis / Icons

- 노트북에 이모지·아이콘 사용 금지 (`✏️ 📌 ⚠️ ✅ 🚀` 등 일체)
- 이유: AI 생성물 티가 나고 강의 배포물의 톤과 맞지 않는다
- 강조는 `**bold**`, blockquote (`>`), 마크다운 헤더로만

## No Cross-references — 노트북 간 / 강의 간 참조 금지

- 다른 회차 번호(`Ch 2`, `Chapter 3`) 언급 금지 — 노트북이 독립적으로 읽혀야 한다
- 학부 과목의 회차 번호(`Practice 02`, `P07`, `Lec 03`) 언급 금지.
  **강의 PDF 에 "Practice 02", "Practice 07-08" 같은 포인터가 있어도 노트북에 옮기지 않는다.**
  그 번호는 다른 과목의 파일명이며 이 저장소에는 존재하지 않는다
- "다음 노트북", "이전 회차", "후속 챕터" 포인터 금지
- 노트북 자신의 제목(`# Ch 1. Machine Learning (1) — Linear Regression`) 은 유지 — 자기 식별자다

## Notebook Comments — 내부 지침 / 메타 코멘트 금지

- 코드 주석과 마크다운은 **"이 줄/셀이 무엇을 하는가"만** 설명한다.
  설계 결정, 다른 방식을 안 쓴 이유, 참조 출처, diff 이력은 전부 금지
- 금지 예: `# (not using asmatrix here)`, `# per the lecture slides`, `# <- added`,
  `# newly added`, `# as reviewer suggested`, `# same pattern as Section 3`
- 허용 예: `# residual E = y - y_hat` (구체적 동작),
  `# mean and std are computed on the training set only` (도메인 지식)
- 비교·대안·근거는 학생 친화적 문장으로 **마크다운 셀**에 쓰거나 `md/` 에 보관한다

## Notebook 편집 — 전체 맥락 우선

- 새 셀을 추가하거나 기존 셀을 고치기 전에 **노트북 전체를 먼저 읽고** 이미 정의된 변수·전제를 파악한다.
  일부만 보고 작업하면 위에서 만든 데이터를 아래에서 또 만들거나, 이미 덮어쓴 변수를 원본인 줄 알고 참조한다
- in-place 덮어쓰기 주의: `X = (X - train_mean) / train_std` 이후의 `X` 는 더 이상 raw 가 아니다.
  raw 가 뒤에서도 필요하면 처음부터 `X_raw` / `X` 로 분리한다
- 한 번 정의한 데이터는 노트북 전체에서 같은 이름으로 재사용한다

## Variable Naming

- **암호 같은 약어 금지**: `tl`, `ta`, `vl`, `va`, `tr`, `te`, `mu`, `sigma`, `opt`, `imp`, `base`,
  `r_a` 전부 금지. 풀어 쓴다 — `optimizer`, `improved`, `baseline`, `train_mean`, `train_std`
- **메서드 shadowing 회피**: `mean = X.mean(...)`, `std = X.std(...)` 금지 → `train_mean`, `train_std`
- **같은 것을 여러 방법으로 계산할 때는 방법명을 풀어서 suffix 로 붙인다**:
  `w_least_squares`, `w_numpy`, `w_autograd`, `w_optimizer`, `w_sklearn`.
  `w_ls2`, `w_t`, `w2` 같은 형태는 쓰지 않는다
- 마크다운 수식에는 $\mu, \sigma, \rho$ 를 쓰고, 코드 변수는 의미를 풀어쓴 영어로 둔다
  (학습률만 예외 — 수식 $\rho$ ↔ 코드 `rho` 로 1:1 대응시킨다)
- PyTorch optimizer 를 쓸 때 인자명은 API 그대로 `lr` 이며, **같은 학습률임을 코드로 드러내기 위해
  `lr=rho` 로 넘긴다**

## 데이터 분할과 스케일링 — 모든 노트북 공통

분할이 먼저다. 스케일러의 통계량을 전체 데이터에서 구하면 test 정보가 학습으로 새고, 그 뒤의 모든
점수가 낙관적으로 부풀려진다.

표준화는 **라이브러리 호출 대신 아래 형태로 직접 쓴다.** 강의 수식
$x_i^{\text{new}} = (x_i - \mu_i)/\sigma_i$ 가 코드 줄과 1:1로 대응하고, numpy 노트북과 torch
노트북에서 같은 코드가 그대로 쓰이기 때문이다.

```python
train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
train_std[train_std == 0] = 1.0    # train 에서 상수인 특징은 스케일하지 않는다

X_train = (X_train - train_mean) / train_std
X_val   = (X_val   - train_mean) / train_std   # 반드시 train 의 통계치 사용
X_test  = (X_test  - train_mean) / train_std
```

- `axis=0` 을 명시한다 (`.mean(0)` 보다 읽기 쉽다)
- 통계량 이름은 `train_mean`, `train_std`. `mean`/`std` 는 메서드를 가리고 `mu`/`sigma` 는 약어다
- 결과는 덮어쓴다. raw 를 뒤에서 다시 쓰는 노트북에서만 `X_train_raw` 로 분리한다
- 데이터셋이 둘 이상이면 접두사를 맞춘다: `X_train_iris`, `iris_train_mean`

### `+ 1e-8` 을 쓰지 않는 이유

0으로 나누는 것을 막는다는 목적은 같지만 실패 경로가 다르다. train 에서 상수였던 특징이 test 에서
움직이면, 그 편차가 $10^{-8}$ 로 나뉘어 $10^{8}$ 규모의 입력이 된다. **에러는 나지 않고 예측만
조용히 망가진다.** `train_std[train_std == 0] = 1.0` 은 그 특징을 "평균과의 차이" 로 남겨 값이
유계로 유지되며, sklearn `StandardScaler` 도 같은 방식이다.

(digits 데이터에서 실측: train 상수 픽셀 3개. 그 픽셀 값이 test 에서 0.0 → 1.0 으로 하나만 달라져도
`+1e-8` 은 `1.0e8`, 위 가드는 `1.0` 을 준다.)

### sklearn 스케일러를 쓸 때

`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler` 를 소개하는 회차에서는 라이브러리를
써도 된다. 단 `fit_transform` 은 train 에만, test 에는 `transform` 만 호출한다.
`fit_transform` 을 두 번 부르는 것이 이 규약에서 가장 흔한 위반이다.

## 비교 실험 통제 변수

A vs B 비교(또는 같은 문제를 여러 도구로 푸는 비교)에서는 **격리하려는 변수 하나만** 다르고
나머지는 반드시 동일해야 한다. 이 원칙이 무너지면 학생에게 전달되는 결론이
"X 가 더 좋다" 가 아니라 "여러 조건이 섞여 비교 자체가 무의미" 가 진실이 된다.

**통제 항목:** 데이터셋과 분할, 전처리·정규화 통계, 초기 가중치, 학습률, 반복 횟수(epoch/iteration),
batch size, optimizer 설정, random seed.

**같은 문제를 여러 방법으로 푸는 셀(수식 / autograd / optimizer / sklearn)에서는
`w_init`, `rho`, `n_iter` 를 위에서 한 번만 정의하고 모든 방법이 그것을 재사용한다.**
방법마다 따로 초기화하면 "세 방법이 같은 답에 도달한다" 는 결론 자체가 성립하지 않는다.

변경된 항목은 마크다운 표(`| 항목 | A | B |`) 나 한 문장으로 명시한다. 의도된 다중 차이라면
"이 비교는 X·Y 가 모두 다르므로 어떤 요인이 주범인지 단정할 수 없다" 고 솔직히 적는다.

## Result Comparison

- 두 결과(수식 vs autograd, NumPy vs torch, 섹션 간 결과)를 비교할 때
  `np.allclose` / `torch.allclose` / `assert` / `np.array_equal` 같은 **검증 코드 금지**
- `print()` 로 나란히 출력하거나 같은 그래프에 겹쳐 그려 **눈으로** 같음을 확인하게 한다
- 이유: 검증 코드는 본질과 무관한 노이즈이며, 값을 직접 보고 일치를 인지하는 편이 학습 흐름에 맞다

## Reproducibility

- `np.random.seed(42)` + `torch.manual_seed(42)` 를 노트북 상단에서 설정한다
- 비교 대상 모델을 생성하기 직전마다 seed 를 재설정한다.
  그러지 않으면 격리하려던 변수 외에 초기값 차이가 섞인다

## Visualization

- 차트 텍스트는 전부 영어
- 회귀: 왼쪽 scatter + 회귀선, 오른쪽 loss 곡선. 분류: 왼쪽 Loss, 오른쪽 Accuracy
- 여러 방법의 loss 곡선을 겹쳐 그릴 때는 선 스타일을 달리해(`'-'`, `'--'`, `':'`) 겹친 곡선이
  보이게 한다. 완전히 포개지는 것이 곧 "같은 결과" 라는 증거다
- `gridspec` / `fig.add_gridspec` 금지 — 독립 셀 + `plt.subplots` 로 분리한다
- 데이터 로드 후 `pd.DataFrame(...).head(10)` 으로 테이블을 미리 본다

## Model Definition Style

- `nn.Sequential()` 또는 `nn.Module` 상속 클래스로 **명시적으로** 정의한다
- helper function / model factory (`make_model`, `build_net`) 금지
- 비교 실험에서 각 모델을 **코드 반복**으로 정의한다. for 문으로 옵션을 순회하지 않는다
  (예외: Grid Search 처럼 본질적으로 반복이 필요한 경우)
- 코드 셀에 docstring(`"""`) 대신 `#` 주석을 쓴다 — generator script 의 삼중 따옴표와 충돌한다
- **모델 생성과 `fit` 을 한 줄에 붙이지 않는다.** 객체를 먼저 이름에 담고 다음 줄에서 학습시킨다.
  `fit_predict` / `fit_transform` 도 같다.

  ```python
  grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1)   # 무엇을 탐색하는가
  grid_search.fit(X_train, y_train)                                    # 언제 도는가
  ```

  체이닝하면 한 줄이 "무엇을 만드는가" 와 "언제 학습하는가" 를 동시에 하고, 그 줄이 길어져
  하이퍼파라미터가 눈에 들어오지 않는다. 시간을 재는 셀에서는 측정 구간까지 흐려진다.

## Notebook 명명과 구성

**강의자료와 실습 노트북은 번호 체계가 다르다.**

| 대상 | 명명 | 예 |
|------|------|-----|
| 강의 슬라이드 | `lecture_notes/Ch{N}-{Area} {k}_{Name}.pdf` | `Ch1-ML 1_Linear Regression.pdf` |
| 실습 노트북 | `Practice{NN}_{Name}.ipynb` | `Practice01_Linear_Regression.ipynb` |
| 과제 | `HW/HW{NN}_{Name}.ipynb` | |
| 과제 정답 | `HW/Answer/HW{NN}_{Name}_answer.ipynb` | |

강의 슬라이드가 본문에서 `Practice 02`, `Practice 07-08` 처럼 실습 번호를 직접 가리킨다.
노트북 번호는 그 포인터를 따른다. **한 회차 슬라이드가 여러 실습으로 갈라지므로 Ch 번호와
Practice 번호는 1:1 이 아니다** — 파일명에서 대응 관계를 추측하지 말고 슬라이드 본문을 확인한다.

- 노트북 제목은 자기 파일명과 일치시킨다: `# Practice 01 — Linear Regression`
- 노트북 안에서는 **다른 Practice 번호도, Ch 번호도 언급하지 않는다** (No Cross-references)
- 노트북은 슬라이드의 섹션 순서를 따르되, 슬라이드에 없는 내용을 임의로 추가하지 않는다

## HW 규약

- `HW/HW{NN}_{Name}.ipynb` 는 **학생 배포본**, `HW/Answer/HW{NN}_{Name}_answer.ipynb` 는 정답 완성본.
  두 파일을 **하나의 generator script 가 `answer=True/False` 플래그로 동시에 생성**한다.
  따로 관리하면 문제 문구와 정답이 어긋난다
- 문제마다 직전 마크다운 셀에 **답을 담을 변수명을 명시**한다 (이름 · 타입 · shape).
  뒤따르는 시각화 셀이 그 이름을 그대로 참조하므로, 명시하지 않으면 학생 코드와 연결되지 않는다
- **시각화 셀은 배포본에서도 완성 상태로 둔다.** 원본 서식처럼 `# ax.scatter(...)` 로 주석 처리해
  두지 않는다. 학생이 위 문제를 풀어 지정된 변수를 만들면 그대로 실행되어야 한다
- 하이퍼파라미터(초기값 · 학습률 · 반복 횟수 · 손실 정의)는 문제 마크다운에 **표로 준다.**
  학생이 임의로 고르면 정답본과 결과가 달라져 채점 기준이 사라진다
- 배포본의 TODO 주석은 **힌트만** 준다. 변수명과 골격은 보여주되 우변은 비운다
- **문제 수와 구성은 원본 서식을 그대로 따른다.** 설명이나 서술형 문항을 임의로 덧붙이지 않는다
- 여러 문제가 같은 학습 설정을 공유하면 `w_init`, `rho`, `n_iter` 를 **주어진 셀에서 한 번만** 정의하고
  모든 문제가 재사용한다. 문제마다 따로 초기화하면 "두 방법의 결과가 같다" 는 확인 자체가 성립하지 않는다
- Practice 노트북의 수식·변수명 컨벤션을 그대로 상속한다. HW 에서 기호를 임의로 바꾸지 않는다
- **실행 검증은 정답본으로 한다.** 배포본은 학생 작성 셀이 비어 있어 중간에서 멈추는 것이 정상이다

## 실행 환경

Anaconda base 환경에서 torch 와 MKL 이 각각 `libiomp5md.dll` 을 올려 **OMP Error #15 로 커널이 죽는다**
(3D 플롯이 있는 노트북에서 실제로 발생). 실행 검증 시 `KMP_DUPLICATE_LIB_OK=TRUE` 로 회피할 수 있으나
이는 임시방편이며, 그렇게 얻은 결과는 torch 를 쓰지 않는 계산과 교차 확인해야 한다.

`01_AI-ME_Graduate/.venv` (uv) 에서는 이 변수 없이도 정상 동작한다. **이 과목에도 uv 전용 환경을
만드는 것이 근본 해결**이며, 학생에게도 같은 환경을 안내하는 편이 낫다.

## Notebook Generator Scripts

- `_gen_p{NN}.py` (실습) / `_gen_hw{NN}.py` (과제) 형태의 generator script 로 노트북을 생성한다.
  `.ipynb` 를 직접 편집하지 않는다
- 생성 후 `jupyter nbconvert --to notebook --execute --inplace` 로 전 셀 실행 검증한다
- 검증이 끝나고 추가 수정이 없다고 확인되면 `_gen_*.py` 를 삭제한다
