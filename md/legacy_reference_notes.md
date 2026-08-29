# Legacy Reference Notes (01_AI-ME_Graduate 구버전 → 현재 Practice 시리즈)

구버전 백업 위치: `_archive/01_AI-ME_Graduate_backup/` (읽기 전용)
비교 대상: 현재 `Practice01`~`Practice12` 와 `CLAUDE.md`

- 셀 번호는 백업 노트북의 **0-based 인덱스**다 (빈 셀 포함).
- `Chapter3_Deep Learning_3_RNN.ipynb` 는 별도 작업으로 진행 중이라 이 문서에서 제외했다.
- `Ch3_DL_2_VGG.ipynb` 와 `Chapter3_Deep Learning_2_1_VGG.ipynb` 는 셀 구성이 거의 동일하다
  (후자는 제목만 "ResNet" 이고 내용은 VGG). 아래에서는 `Ch3_DL_2_VGG.ipynb` 쪽을 인용한다.
- 이 문서는 **정리·요약**이다. 무엇을 하라는 지시가 아니라, 구버전에 무엇이 있었고
  현재 어떤 상태인지를 적는다. 판단과 실행은 사람이 한다.

---

## 누락된 내용 후보

### 1. L1 손실 / 이상치에 강건한 회귀

`Ch2_ML_3_General_Tips_1.ipynb` cell [4]~[11].

정상 데이터 100개에 x 는 평균 3, y 는 평균 -0.5 인 이상치 5개를 `np.vstack` 으로 붙인 뒤,

- cell [6]~[8]: L2 손실 `((y - y_hat)**2).sum()` 으로 적합
- cell [10]~[11]: 같은 클래스를 상속해 `loss()` 만 `torch.abs(...).sum()` 으로 바꿔 L1 으로 적합

두 경우 모두 **"이상치 포함 데이터 / 적합된 직선 / 원래 ground-truth 직선"** 세 가지를 한 그림에
겹쳐 그려, L2 선이 이상치 쪽으로 끌려가는 정도와 L1 선이 버티는 정도를 눈으로 비교한다.

현재 상태: 새 시리즈 전체에 `L1Loss` / `SmoothL1Loss` / `HuberLoss` / `torch.abs` 가 한 번도
나오지 않는다. `Practice01` 은 MSE 계열만, `Practice05` 는 분류 지표와 R2, bias-variance 를 다룬다.
손실함수의 선택이 결과를 바꾼다는 관점 자체가 새 시리즈에 없다.

### 2. Perceptron

`Ch2_ML_1_Basics.ipynb` cell [33]~[35] (NumPy 클래스 구현), cell [47]~[48]
(`sklearn.linear_model.Perceptron` 과 `LogisticRegression` 을 같은 데이터에 나란히 적용).

구버전 구현의 특징:

- 샘플 하나씩 도는 온라인 갱신 (`for xi, target in zip(X, y)`)
- **epoch 마다 오분류 개수를 `errors_` 에 기록** — logistic regression 이 손실을 기록하는 것과 대비되는
  "퍼셉트론에는 매끄러운 손실이 없다" 는 점이 코드로 드러난다
- `predict` 가 `np.where(net_input >= 0.0, 1, 0)` 인 계단 함수

현재 상태: 새 시리즈에 "perceptron" 문자열이 없다. `Practice02` 는 곧바로 logistic regression 으로 간다.

### 3. 다중 클래스 ROC (One-vs-Rest)

`Ch2_ML_4_Unsupervised_Learning.ipynb` cell [60]~[64].

`predict_proba` 다음 `label_binarize(y_test, classes=[0,1,2])`, 클래스마다 `roc_curve` 와
`roc_auc_score` 를 구해 3개 subplot 에 클래스별 ROC 를 각각 그린다.
"다중 클래스에서 ROC 는 클래스마다 하나씩" 이라는 점이 그림 구조로 전달된다.

현재 상태: `Practice05` 의 ROC 는 digits 를 "이 이미지가 3인가" 로 만든 **이진 문제 하나**다
(`y_binary = (digits.target == 3).astype(int)`). 다중 클래스 확장은 없다.

### 4. 상관분석 (Pearson / Spearman / Kendall, 상관 히트맵)

두 군데에 있었다.

- `Ch1_Basics_Pandas.ipynb` cell [49]~[52]: `df.corr()` 기본(pearson)과 `method="spearman"`,
  그리고 `scipy.stats` 의 `pearsonr`, `spearmanr`, `kendalltau` 를 같은 두 변수에 대해 나란히 출력.
  세 계수가 무엇이 다른가를 값으로 비교한다.
- `HW/HW3_answer.ipynb` cell [13]~[14]: 상관행렬 히트맵을 그리고, 타깃(`status`) 과의 상관을
  **절대값 기준 내림차순으로 정렬**해 중요 변수를 고르는 흐름.
  이 결과를 뒤의 `feature_importances_` 상위 3개와 대조하는 구성이었다.

현재 상태: 새 시리즈에 상관계수와 상관 히트맵이 없다 (`Practice06` 의 heatmap 은 grid search 결과
표시용). 특징 선택의 근거로 상관을 보던 흐름이 통째로 빠져 있다.

### 5. 결측치가 있는 데이터 다루기

`Ch1_Basics_Pandas.ipynb` cell [41]~[48]. `iris_missing.csv` 를 읽어
`notna()` 다음 `all(axis=1)` 로 결측 없는 행만 남기고, 남은 행 수를 확인한 뒤 `describe()` 로
통계를 다시 낸다.

현재 상태: 새 시리즈의 데이터는 sklearn 내장 데이터셋과 이미지뿐이라 결측이 존재하지 않는다.
단, 이 항목은 `CLAUDE.md` 의 "ML 이 아닌 pandas 관용구 배제" 와 정면으로 겹치는 영역이므로
누락인지 의도적 제외인지 판단이 필요한 자리다.

### 6. CNN 필터 시각화 / activation maximization

`Chapter3_Deep Learning_2_2_CNN_visualization.ipynb` cell [7]~[10].

특정 conv 레이어(`conv2d_1`) 의 필터 번호 `[0, 2, 4]` 를 고르고, `tf.random.uniform` 으로 만든
random seed input 에서 시작해 **그 필터의 활성이 최대가 되도록 입력 이미지 자체를 최적화**한 뒤
결과를 3칸으로 나란히 보여준다 (`ActivationMaximization`).

feature map 과 질문이 다르다:

| 방식 | 답하는 질문 |
|------|-------------|
| feature map | 이 **입력**에 대해 이 레이어의 무엇이 켜졌는가 |
| activation maximization | 이 **필터**가 가장 좋아하는 입력은 무엇인가 |

현재 상태: `Practice10` 은 Grad-CAM (Step 3) 과 feature map (Step 4) 두 가지뿐이고,
노트북 첫 셀의 표에도 두 항목만 적혀 있다. 필터 자체를 보는 관점은 없다.

### 7. Saliency map

`Ch3_DL_3_CNN_visualization.ipynb` cell [0] 의 개요에 "Grad-CAM, Saliency, Feature-map visualization"
이라고 적혀 있으나, **실제 셀에는 Grad-CAM(cell [10]) 과 feature map(cell [12]) 만 있고 saliency 구현은
없다.** 구버전에서도 미구현 상태였다는 기록으로만 남긴다.

### 8. DL 튜닝 항목 체크리스트 (선택지 목록)

`Ch3_DL_1_Basics.ipynb` cell [11] (= `Chapter3_Deep Learning_1.ipynb` cell [13]).

한 마크다운 셀에 여섯 항목을 번호 매겨 정리하고 각각 한두 줄 코드 예를 붙였다:
activation layer 종류 / optimizer 종류 / mini-batch (DataLoader batch_size) / weight initialization
(`nn.init.xavier_uniform_` 과 `model.apply(init_weights)`) / batch normalization / regularization
(`weight_decay` 로 L2, `nn.Dropout`). 마지막에 이것들을 전부 적용한 모델 코드 예시가 붙는다.

현재 상태: `Practice07` 은 이 중 BatchNorm, Dropout, He 초기화, weight_decay 를 improved 모델에서
**실제로 적용**한다 (Step 8, `nn.init.kaiming_normal_` 을 레이어 순회로 적용).
개별 기법은 커버되지만, **선택지를 한눈에 훑는 목록형 정리**와 optimizer/activation 대안 나열은 없다.
`model.apply(init_weights)` 패턴도 새 시리즈에는 없다 (명시적 for 문으로 대체됨).

### 9. 모델 구조와 파라미터 수 요약 출력

`Ch3_DL_1_Basics.ipynb` cell [6], [15]:
`from torchsummary import summary` 다음 `summary(model, (1, 28 * 28), device=str(device))`.
레이어별 **출력 shape 와 파라미터 수**가 표로 나온다.

현재 상태: `Practice07`, `Practice09`, `Practice10` 모두 `print(model)` 만 한다.
레이어 목록은 보이지만 각 단계의 shape 와 파라미터 수는 나오지 않는다.
`Practice09` 는 대신 블록별 shape 를 코드 주석으로 적어 첫 linear 층의 `512 * 3 * 3` 이 어디서
왔는지 설명하는 방식을 쓴다.

### 10. 군집에 대한 AutoML / 자동 평가 플롯

`Ch2_ML_4_Unsupervised_Learning.ipynb` cell [65]~[76] (PyCaret clustering).
`setup(..., ignore_features=['Name'])` 다음 `create_model('kmeans', num_clusters = 5)`,
`assign_model` 로 각 샘플의 소속 군집을 원본 표에 붙여 보고, `plot_model(kmeans, plot='elbow')`,
`plot_model(kmeans, plot='silhouette')` 로 elbow 와 실루엣 그림을 한 줄로 얻는다.
cell [74] 주석에 실루엣 그림 읽는 법(스코어가 높을수록, 군집별 평균의 편차가 작을수록)이 적혀 있다.

현재 상태: `Practice04` 는 elbow 와 실루엣 스코어를 직접 계산해 그린다 (Step 3 Choosing k).
그림 자체는 대체되었으나, **군집 결과를 원본 표에 다시 붙여 보는 단계**(`assign_model` 에 해당)와
PyCaret 계열 도구는 없다. `Practice06` 의 AutoML 은 지도학습(분류) 전용이다.

---

## 참고할 만한 표현/코드 방식

### A. 세 가지 방식을 한 화면에 놓고 값을 비교하는 셀

`Ch2_ML_1_Basics.ipynb` cell [18]. optimizer 로 한 스텝 갱신하면서, 같은 iteration 안에서
(1) torch 가 준 `.grad` 로 손으로 계산한 다음 값, (2) 수식으로 계산한 다음 값,
(3) `optimizer.step()` 이 실제로 만든 값 — 셋을 한 줄 `print` 에 나란히 찍는다.

관점: 새 `Practice01` 은 **끝난 뒤의 최종 결과 7개**와 loss 곡선 겹치기로 일치를 확인한다
(Step 11). 구버전은 **매 iteration 의 중간값**을 나란히 놓았다. 확인하는 지점이 다르다.
구버전 셀에는 "이 블록은 조사용이고 optimizer 동작에 필요 없다" 는 구분 주석이 코드 안에
들어 있는데, 이 형태는 현재 규약(내부 메타 코멘트 금지)과 충돌한다.

### B. 상속으로 "한 군데만 바꾸기" 를 드러내는 비교

`Ch2_ML_3_General_Tips_1.ipynb` cell [10] 은 `find_theta` 를 상속해 `loss()` 한 메서드만
L1 으로 바꾼다. `Ch2_ML_1_Basics.ipynb` cell [40]~[42] 도 같은 방식이다 —
`LogisticRegressionGD_Pytorch` 를 상속해 `update_params` 만 `torch.optim.SGD` 로 바꾸고(cell [41]),
다시 상속해 `loss_function` 만 `BCEWithLogitsLoss` 로 바꾼다(cell [42]).
**"직전 것과 딱 한 메서드가 다르다"** 가 diff 없이 코드 형태로 보인다.

관점: 같은 교육 목표를 새 시리즈는 다른 수단으로 달성한다 — `Practice01` 은 Step 3 부터 Step 9 까지를
독립 셀로 **반복해서 쓰고** 마지막에 표로 "각 단계에서 무엇을 torch 에 넘겼는가" 를 정리한다.
`CLAUDE.md` 의 "비교 실험에서 각 모델을 코드 반복으로 정의한다", "helper / model factory 금지" 와
상속 방식은 방향이 다르다. 다만 "무엇 하나만 바뀌었는지" 를 표로 명시하는 아이디어는
`Practice01` Summary 표에 이미 반영되어 있다.

### C. 4차원 데이터의 군집 결과를 특징쌍 3개 패널로 보기

`Ch2_ML_4_Unsupervised_Learning.ipynb` cell [9], [19]. iris 4개 특징 중 (0,1), (1,2), (0,3) 쌍을
각각 산점도로 그려 세 패널을 나란히 놓는다. cell [19] 는 여기에 군집 중심(`cluster_centers_`)까지
같은 좌표계로 찍는다.

관점: 새 `Practice04` 는 petal length/width 한 쌍(`X[:, 2]`, `X[:, 3]`)만 반복해서 쓰고,
차원 축소는 Step 7 PCA 에서 따로 다룬다. 4차원을 어디까지 보여줄지에 대한 선택이 다르다.

### D. 이상치 실험을 "정상 데이터 + 붙인 이상치" 로 구성하는 방식

`Ch2_ML_3_General_Tips_1.ipynb` cell [3]~[4]. 먼저 `Ch2_ML_1_Basics.ipynb` cell [3] 과 **완전히
같은 코드**로 깨끗한 데이터를 만들고, 그 다음 셀에서 이상치만 덧붙인다. 그림에서도 정상점은
빨강, 이상치는 파랑으로 색을 나눈다. 통제된 비교 조건이 데이터 생성 단계에서부터 코드로 보장된다.

### E. predict_proba 임계값을 바꿔 F1 을 비교하는 문제 구성

`HW/HW3.ipynb` 와 `HW3_answer.ipynb` cell [32]~[33]. threshold 0.5 와 0.8 에서 F1 을 각각 구해
비교하고 해석하게 하는 문항이다. 임계값이 지표를 바꾼다는 점을 과제로 만든 형태.

관점: `Practice05` 는 같은 개념을 **곡선 위 한 점을 찍는** 방식으로 다룬다
(Youden J 는 TPR - FPR 의 argmax, 그리고 max-F1 임계값). 개념은 이미 커버되어 있고,
구버전 쪽은 과제 문항 형태라는 점만 다르다.

### F. DL 실험 조건을 인자 객체로 묶고 결과를 파일로 남기는 패턴

`Ch3_DL_2_VGG.ipynb`:

- cell [3]: `args` 하나에 exp_name / act / l2 / optim / lr / epoch / batch_size 를 모아 둠
- cell [16]: `experiment()` 가 `vars(args)` 와 result dict 를 함께 반환
- cell [18]: `save_exp_result` 와 `load_exp_result` 로 `./results/{exp_name}.csv` 왕복
- cell [23]: 두 변수(l2, optim) 이중 for 문으로 4조합 실행
- cell [20]: `plot_acc` 이 2x3 막대 그리드로 train/val/test 의 acc 와 loss 6개를 한 화면에

`experiment/a_setup.ipynb`, `b_run.ipynb`, `c_setup_run_simple.ipynb` 에 이 패턴만 떼어낸
최소 예제가 따로 있다 — `argparse` 로 기본값을 정의하고(`a_setup`), `sys.argv` 를 바꿔가며
`%run` 으로 반복 실행하며(`b_run`), 마지막에 argparse 없이 `args` 객체와 함수 호출로만
같은 일을 하는 단순화 버전(`c_setup_run_simple`)까지 3단계로 보여준다.

관점: 새 시리즈에서 대응하는 자리는 `Practice06` 이지만 그쪽은 sklearn 전용(GridSearchCV,
Optuna, FLAML)이고, DL 쪽(`Practice07`, `Practice09`)에는 실험 조건 관리와 결과 저장 패턴이 없다.
`Practice09` 는 "VGG from scratch vs pretrained ResNet18" A/B 비교 한 쌍으로 끝난다.
단, cell [23] 의 이중 for 문 sweep 은 `CLAUDE.md` 의 "for 문으로 옵션을 순회하지 않는다" 와
충돌하고, cell [20] 의 `groupby` 는 "pandas 관용구 배제" 와 충돌한다.
`c_setup_run_simple.ipynb` 쪽이 규약과의 거리가 가장 가깝다.

### G. 임의의 sklearn 분류기에 대해 결정 영역을 그리는 유틸

`Ch2_ML_1_Basics.ipynb` cell [26] (`draw_scatter`), cell [28] (`plot_decision_regions`).
`meshgrid` 다음 `classifier.predict(...)`, 그리고 `contourf` 구조이고,
`test_idx` 를 주면 test 샘플만 테두리 있는 빈 원으로 덧그려 train 과 test 를 한 그림에서 구분한다.
Perceptron, LogisticRegressionGD, 각종 torch 래퍼, sklearn 모델 어디에나 `predict` 만 있으면 붙는다.

관점: `Practice02` 는 같은 아이디어를 인라인으로 쓰지만(Step 7, torch 모델을 grid 에 통과시켜
`contourf`) 재사용 형태는 아니고, `Practice03` 에는 결정 영역 그림이 없다.
**test 샘플을 그림 위에 표시해 구분하는 장치**는 새 시리즈 어디에도 없다.

### H. 자동 미분을 손계산 가능한 예제로 단계별로 올리는 구성

`Ch2_ML_1_Basics_Pytorch.ipynb` cell [7]~[18]. 마크다운에 수식과 도함수를 먼저 쓰고
바로 아래 셀에서 `backward()` 로 확인하는 쌍을 **여섯 번** 반복한다:
2차식 → 선형 → 계수가 붙은 2차식 → 합성함수 → 정답이 붙은 제곱오차 → 파라미터 2개.

관점: 새 `Practice01` Step 4 가 같은 구성을 **세 예제**로 압축했다 (2차식, 선형, 합성).
구버전에는 있고 새 쪽에 없는 마지막 두 단계는 "정답 y 가 손실 안에 들어오는 형태" 와
"파라미터가 둘일 때 `.grad` 가 각각 나온다" 이다. 후자는 `Practice01` Step 5 에서 벡터 w 로
한 번에 처리된다.
cell [3]~[5] 의 `detach()`, `.numpy()`, `.item()` 세 메서드가 각각 무엇을 반환하는지
`print(type(...))` 까지 찍어 구분하는 셀도 새 시리즈에는 대응이 없다.

---

## 이미 의도적으로 다르게 감 (되돌리지 말 것)

`CLAUDE.md` 에 근거가 명시된 항목들이다. 구버전 코드를 참고하다 무심코 되돌리기 쉬운 자리를 적어 둔다.

| 항목 | 구버전 | 현재 | 근거 |
|------|--------|------|------|
| 손실함수 상수 | 손실에 1/2 없음, 기울기는 2 를 곱한 형태 (`Ch2_ML_1_Basics` cell [8], [14]) | J 정의에 1/2 를 넣어 `((y_hat - y) ** 2).sum() / 2` | `CLAUDE.md` "손실함수에 1/2 를 붙인다". 수식, NumPy, autograd 세 값이 정확히 일치해야 눈으로 대조할 수 있다 |
| 파라미터 기호 | `theta`, `theta0`, `theta1`, `t0`, `t1` | `w`, `w0`, `w1` (열벡터) | `CLAUDE.md` Notation 표. `theta` 명시적 금지 |
| 학습률 이름 | `alpha` (`Ch2_ML_1_Basics` cell [8]), `step_size` (cell [15]), `eta` (Perceptron 클래스) | `rho`, torch 에는 `lr=rho` | `CLAUDE.md` "alpha, eta, step_size 금지" |
| 기울기 변수 | `df` | `dJ` | `CLAUDE.md` Notation 표 |
| 행렬 연산 | `np.asmatrix(A)` 후 `A.T * A * theta`, 역행렬은 `.I` (`Ch2_ML_1_Basics` cell [5], [8]) | `@` 와 `np.linalg.inv` | `CLAUDE.md` 행렬 연산. `*` 가 원소곱인지 행렬곱인지 학생이 매번 확인해야 한다 |
| 잔차 부호 | 클래스마다 제각각 | `E = y - y_hat`, `dJ = -X.T @ E` 로 통일 | `CLAUDE.md` "잔차의 부호" |
| 비교 실험 초기값 | 방법마다 따로 랜덤 초기화 (`Ch2_ML_1_Basics` cell [15], [16], [18]) | `w_init`, `rho`, `n_iter` 를 위에서 한 번 정의하고 전부 재사용 (`Practice01` Step 3) | `CLAUDE.md` "비교 실험 통제 변수". 따로 초기화하면 "여러 방법이 같은 답에 도달" 이라는 결론 자체가 성립하지 않는다 |
| 모델 정의 | `find_theta` 클래스가 데이터, 파라미터, optimizer, 학습 루프를 전부 안고 있고 비교는 상속으로 (`Ch2_ML_3_General_Tips_1` cell [6], [10]) | `nn.Module` 또는 `nn.Sequential` 로 명시 정의, 비교는 코드 반복 | `CLAUDE.md` Model Definition Style, helper / model factory 금지 |
| 언어 | 마크다운, 주석, print 가 한국어 (`Ch2_ML_2_ML_Models`, `Ch2_ML_4`, `Ch3_DL_1` 전반) | 전부 영어 | `CLAUDE.md` Language English only. 노트북에 한글을 한 글자도 남기지 않는다 |
| 노트북 간 참조 | `HW/HW2.ipynb` cell [2] 이 "as shown in the Practice Code (Ch2_ML_1_Basics)" 로 다른 노트북을 지목 | 다른 회차 번호와 파일명 언급 금지 | `CLAUDE.md` No Cross-references |
| 스케일링 | `Ch2_ML_2_ML_Models` cell [8] 은 raw 를 덮어쓰는 형태, `Ch2_ML_4` cell [40] 은 **분할 전 전체 데이터**에 `MinMaxScaler().fit_transform(df)` | 분할이 먼저, 통계는 train 에서만, `train_std[train_std == 0] = 1.0` 가드 | `CLAUDE.md` "데이터 분할과 스케일링". `Ch2_ML_4` cell [40] 은 명백한 leakage 형태다 |
| pandas 집계 | `apply(lambda ...)` (`Ch2_ML_2_ML_Models` cell [23], [43]), `groupby` (`Ch3_DL_2_VGG` cell [20]), `cv_results_` 재집계 (`Ch2_ML_2_ML_Models` cell [31]) | 평범한 for 문, 행을 그대로 적는 DataFrame, `best_params_` / `best_score_` / `best_estimator_` | `CLAUDE.md` "코드 복잡도 — ML 이 아닌 구문은 배제" |
| 시각화 라이브러리 | seaborn (`Ch2_ML_2` cell [24], `Ch2_ML_4` cell [9], [19], [57]) | matplotlib 만 — 새 시리즈에 seaborn import 가 한 번도 없다 | `CLAUDE.md` Visualization 절이 matplotlib 형태만 규정 |
| 디바이스 | `Ch3_DL_2_VGG` cell [12]~[14], [16] 이 `.cuda()` 하드코딩 (CPU 에서 실행 불가) | `device` 를 잡고 `.to(device)`, CPU 에서도 돌아감 | 전 셀 실행 검증 요구 (`CLAUDE.md` Notebook Generator Scripts) |
| 이미지 정규화 통계 | `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` (`Ch3_DL_2_VGG` cell [5]) | `Practice08` 은 학습 데이터에서 채널별 통계를 직접 계산해 보이고, `Practice09` 와 `Practice10` 은 사전학습 가중치가 쓴 ImageNet 통계를 명시 | 통계의 출처를 코드로 드러내는 방향 |
| AutoML 도구 | PyCaret (`Ch2_ML_3_General_Tips_2_automl` cell [4]~[17], `Ch2_ML_4` cell [65]~[76], `HW3_answer` cell [36]~[42]) | GridSearchCV + Optuna + FLAML (`Practice06`) | 구버전 cell [3] 자체가 "Conda 사용시 환경 충돌 발생" 을 경고하며 별도 venv / uv 구축을 요구했다. 교체 이유가 구버전 노트북 안에 이미 적혀 있다 |
| 노트북 파일명 | 공백 포함 (`Chapter3_Deep Learning_1.ipynb`) | `Practice{NN}_{Name}.ipynb`, 공백 없음 | `CLAUDE.md` "파일명과 디렉터리명에 공백을 쓰지 않는다" |
| 결과 비교 | `Ch2_ML_1_Basics` cell [44] 이 배열 비교로 정확도를 검산 | print 로 나란히 출력하거나 곡선을 겹쳐 그려 눈으로 확인, allclose 와 assert 금지 | `CLAUDE.md` Result Comparison |
| 코드 셀 안 설명 | 삼중따옴표 블록으로 설명 (`Ch2_ML_2_ML_Models` cell [7], `Ch3_DL_1_Basics` cell [3]) | `#` 주석만 — generator script 의 삼중따옴표와 충돌 | `CLAUDE.md` Model Definition Style |
| 노트북 안 설치 명령 | 셸 매직으로 pip 설치와 mkdir (`Ch2_ML_2` cell [16], `Ch3_DL_2_VGG` cell [1], `Ch3_DL_3` cell [9]) | 새 시리즈에는 셸 매직이 없다 | 실행 검증 가능한 노트북 요구 |

추가로, 되돌릴 대상이 아니라 **기록으로만 남길 것**:

- `Chapter3_Deep Learning_2_1_VGG.ipynb` 는 cell [0] 제목이 "CIFAR-10 Classification with ResNet"
  이지만 내용은 VGG 다 (`Ch3_DL_2_VGG.ipynb` 와 셀 구성이 거의 동일). 구버전 자체의 오류다.
- `Ch1_ML_1_Basics.ipynb` 는 **셀이 0개인 빈 노트북**이다 (255 bytes).
- `Ch2_ML_1_Basics.ipynb` cell [12] 이 "Go to Pytorch Basics and do the practice!" 로
  다른 노트북 이동을 지시한다. 노트북이 독립적으로 읽히지 않던 구조의 흔적.
- `Ch1_Basics_Pandas.ipynb` cell [1]~[5] 는 Python 클래스 상속 문법(`super().__init__()`,
  `*args` 와 `**kwargs`) 자체를 가르친다. 현재 대상 독자(대학원생, Python 문법은 설명하지 않음)와
  전제가 다르다.

---

## 재사용 가능한 자산

### 데이터셋 (현재 `data/` 에 없음)

`_archive/01_AI-ME_Graduate_backup/data/` 와
`_archive/01_AI-ME_Graduate_backup/AI-ME-Practice-1/data/` 양쪽에 같은 CSV 세트가 있다.
현재 `01_AI-ME_Graduate/data/` 는 이미지, MNIST, CIFAR, COCO, crack 계열뿐이라
**표 형식 CSV 가 하나도 없다.**

| 파일 | 구버전에서의 쓰임 / 성격 |
|------|--------------------------|
| `parkinsons.csv` | `HW/HW3.ipynb` 의 과제 데이터. 음성 특징으로 파킨슨병 `status` 예측. 이진 분류 + 특징 중요도 + 임계값별 F1 을 한 문제에 담기에 적당한 크기 |
| `iris_missing.csv` | `Ch1_Basics_Pandas.ipynb` cell [41], [50]. **결측이 들어간 iris** — 결측 처리와 상관분석 예제용 |
| `production_monthly_1990.csv` | `Ch1_Basics_Pandas.ipynb` cell [9]. year / month / volumn 컬럼의 생산량 시계열. 제조 도메인 데이터 |
| `kospi.csv` | 구버전 노트북 본문에서는 참조되지 않음. 시계열 |
| `Mall_Customers.csv` | 본문 미참조. 군집 예제에 흔히 쓰이는 형태 |
| `Bank_Personal_Loan_Modelling.csv` | 본문 미참조. 불균형 이진 분류 |
| `CVD_cleaned.csv` | 본문 미참조. 대형 이진 분류 |
| `E-commerce-Shipping-Data.csv` | 본문 미참조. 이진 분류 |
| `TV_new.csv` | 본문 미참조. 단순 회귀용으로 보임 |

이 중 실제로 구버전 노트북이나 과제가 사용한 것은 `parkinsons.csv`, `iris_missing.csv`,
`production_monthly_1990.csv` 세 개뿐이다. 나머지는 `data/` 에만 있고 인용처가 없다.

### 데이터 로딩 방식

`Ch1_Basics_Pandas.ipynb` cell [9], [41] 과 `HW/HW3.ipynb` cell [2] 가
`pd.read_csv` 에 GitHub raw URL(`jongmoonha/AI-ME-Practice` 저장소의 `data/` 경로)을 그대로 넣어
**원격에서 직접 읽는다.** 학생이 파일을 따로 받지 않아도 되는 대신 네트워크와 외부 저장소에
의존한다. 현재 저장소는 `data/` 로컬 파일 방식이다.

### 유틸리티 코드 (현재 `tools/` 에 없음)

현재 `tools/` 에는 `prepare_pcb_dataset.py` 하나뿐이다.

| 유틸 | 출처 | 내용 |
|------|------|------|
| `plot_decision_regions` / `draw_scatter` | `Ch2_ML_1_Basics.ipynb` cell [26], [28] | `predict` 를 가진 어떤 분류기에도 붙는 결정 영역 그림. `test_idx` 로 test 샘플을 테두리 원으로 표시 |
| `bias_variance_decomp` | `Ch2_ML_3_General_Tips_1.ipynb` cell [13] | 부트스트랩 `n_rounds` 회 재학습으로 bias 제곱 / variance / MSE 를 분해. sklearn estimator 면 무엇이든 받음. `Practice05` Step 3 에 같은 개념이 이미 반영되어 있어 중복 확인 필요 |
| `experiment` / `save_exp_result` / `load_exp_result` / `plot_acc` | `Ch3_DL_2_VGG.ipynb` cell [16], [18], [20] | DL 실험 1회 실행, 결과 dict 반환, csv 저장과 로드, 2변수 막대 그리드. 단 `.cuda()` 하드코딩과 `groupby` 사용은 현재 규약과 충돌 |
| argparse 실험 설정 3단계 예제 | `experiment/a_setup.ipynb`, `b_run.ipynb`, `c_setup_run_simple.ipynb` | 설정 정의 / `sys.argv` 교체 후 `%run` 반복 / argparse 없는 단순 버전. 셋 중 `c_setup_run_simple.ipynb` 가 규약과 가장 가깝다 |
| `ensure(pkg)` 의존성 설치 헬퍼 | `HW/HW4.ipynb` cell [0], `HW4_answer.ipynb` cell [0] | import 시도 후 실패하면 subprocess 로 pip 설치. 현재는 `.venv` 로 해결하는 문제 |
| `dimesion_check(vgg_name)` | `Ch3_DL_2_VGG.ipynb` cell [10]~[11] | 랜덤 텐서 `(2, 3, 32, 32)` 를 통과시켜 출력 shape 만 찍는 셀. 모델 정의 직후 shape 를 확인하는 습관. `Practice09` 는 같은 목적을 주석으로 처리 (오타 포함 이름은 그대로 옮길 것이 아님) |

### 과제 문항 (현재 `HW/` 에 대응이 없는 것)

현재 `HW/` 에는 `HW01_Linear Regression.ipynb` 와 `HW_PCB_Defect_Detection.ipynb` 두 개뿐이다.
구버전 `HW/` 에는 다섯 세트가 있었다.

| 과제 | 내용 | 현재 대응 |
|------|------|-----------|
| `HW1_updated.ipynb` | 입력 두 개의 제곱항 회귀를 최소제곱 / 수식 기울기 GD / autograd GD / `optimizer.step()` 네 가지로 푸는 4문항. cell [3] 에 `plot_surface` 3D 곡면 시각화 | `HW01_Linear Regression.ipynb` 가 대응. 다만 구버전 쪽은 **입력이 2개이고 항이 제곱** 이라 augmented 행렬이 3열이 되는 확장 예제다 |
| `HW2.ipynb` | iris 3클래스 multi-class logistic regression 을 (1) **손실함수를 직접 정의**해서, (2) torch 제공 손실로 두 번 풀게 함 | 대응 없음. `Practice02` 가 같은 주제를 다루지만 과제는 없다 |
| `HW3.ipynb` / `HW3_answer.ipynb` | parkinsons 데이터. 컬럼 제거, EDA(describe, 클래스 분포, 상관 히트맵), 9:1 분할, MinMax, GridSearchCV(하이퍼파라미터 4종), 특징 중요도 상위 3개, 임계값 0.5 와 0.8 의 F1 비교 | 대응 없음. `HW_PCB_Defect_Detection` 은 도메인이 다르다 |
| `HW4.ipynb` / `HW4_answer.ipynb` | YOLO11 로 두 이미지에 detection 과 instance segmentation 을 적용하고 결과를 그림. 이미지 1은 원격 URL, 이미지 2는 `HW4_bus.jpg` | `Practice11` 이 같은 내용을 실습으로 다룬다. 과제 대응은 없다 |
| `HW5.pdf` / `HW5.docx` | 노트북이 아닌 문서 형식 과제. 내용 미확인 | — |

`HW3.ipynb` 의 배포본 형식이 현재 `CLAUDE.md` HW 규약(배포본은 힌트만, 시각화 셀은 완성 상태)과
가까운 형태다 — 문제마다 답을 담을 변수명을 `X = `, `y = ` 처럼 좌변만 남겨 두었고,
"Show the followings for every experiment" 처럼 출력 형식을 마크다운에 목록으로 지정한다.
