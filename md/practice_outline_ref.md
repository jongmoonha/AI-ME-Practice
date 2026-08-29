# Practice Outline

각 Practice 노트북의 구성, 사용 데이터셋, 라이브러리 정리.

---

## Practice 01 — Python Basics (Lec 02)

**파일:** `Practice01_Python_Basics.ipynb`

| 섹션 | 내용 |
|------|------|
| 0. 용어 정리 | 라이브러리, 함수, 메서드, 속성, print/f-string |
| 1. 변수와 자료형 | int, float, str, bool, 산술 연산, 형변환 |
| 2. 자료구조 | list, tuple, dict (하이퍼파라미터 관리) |
| 3. 제어문 | if/elif/else, for (range/enumerate/zip), while, break/continue |
| 4. 함수 | 정의, `*args`, `**kwargs` |
| 5. 리스트 컴프리헨션 & 람다 | 조건부 컴프리헨션, lambda |
| 6. 문자열 포매팅 | f-string, `.format()`, join/split |
| 7. 클래스 & OOP | `__init__`, fit/predict 패턴, 상속, `super()` |
| 8. NumPy 기초 | 배열 생성, dtype, reshape, 인덱싱, 통계, argmax, vstack/hstack, `@` |
| 9. Matplotlib 기초 | line, subplot, scatter, histogram, bar, imshow |
| 10. 데이터 불러오기 | sklearn (Iris, Digits), UCI (Wine), CSV (Penguins) |
| 11. 종합 정리 | import 패턴, 종합 연습 (뉴런, 경사하강법) |

- **데이터셋:** Iris, Digits, Wine Quality, Palmer Penguins
- **라이브러리:** numpy, matplotlib, pandas, sklearn, ucimlrepo

---

## Practice 02 — Linear Regression (Lec 03)

**파일:** `Practice02_Linear_Regression.ipynb`

| 섹션 | 내용 |
|------|------|
| 수식 정리 | LS vs GD 비교표 |
| Step 1 | 합성 데이터 생성 ($y = 0.1 + 0.3x + \mathcal{N}(0, 0.1)$, N=100) |
| Step 2 | 정규방정식 (Least Squares): $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| Step 3 | 경사하강법: $\alpha=0.00001$, 3000 iterations |
| 시각화 | scatter + 회귀선, $w_0$/$w_1$ 수렴 그래프 |

- **데이터셋:** 합성 1D 선형 데이터 (N=100)
- **라이브러리:** numpy, matplotlib

---

## Practice 03 — Perceptron (Lec 03)

**파일:** `Practice03_Perceptron.ipynb`

| 섹션 | 내용 |
|------|------|
| 수식 정리 | unit step, 그래디언트, 갱신 규칙 |
| Step 1 | Toy 데이터 (5샘플, 2D, 이진) + augmented matrix |
| Step 2 | 초기 가중치 $\mathbf{w}=(-25, 10, 10)$, $\rho=1.0$ |
| Step 3 | 활성함수 정의 (unit step) |
| Step 4–5 | 순전파 $\mathbf{z}=\mathbf{Xw}$ → $\sigma(\mathbf{z})$ → 오분류 확인 |
| Step 6–7 | 그래디언트 계산 → 가중치 1회 갱신 |
| Step 8 | 갱신 후 재분류 검증 |
| Step 9 | 결정경계 시각화 (Before vs After) |

- **데이터셋:** 합성 2D 이진 분류 (N=5, 선형 분리 가능)
- **라이브러리:** numpy, matplotlib

---

## Practice 04 — Logistic Regression, Binary (Lec 04)

**파일:** `Practice04_Logistic_regression.ipynb`

| 섹션 | 내용 |
|------|------|
| 수식 정리 | sigmoid, BCE, 그래디언트, 갱신 규칙 |
| Step 1 | Toy 데이터 (5샘플, 2D, 이진) + augmented matrix |
| Step 2 | 초기 가중치 $\mathbf{w}=(-25, 10, 10)$ |
| Step 3 | sigmoid 활성함수 정의 |
| Step 4–5 | 순전파 → 확률 출력 → 오분류 확인 |
| Step 6–7 | 그래디언트 $\mathbf{X}^T(\sigma-\mathbf{y})$ → 가중치 갱신 ($\rho=1.0$) |
| Step 8 | 갱신 후 재분류 검증 |
| Step 9 | 결정경계 시각화 |
| Step 10 | 확률 등고선 (contour) — 로지스틱 회귀의 진짜 출력 |

- **데이터셋:** 합성 2D 이진 분류 (N=5)
- **라이브러리:** numpy, matplotlib

---

## Practice 05 — Logistic Regression, Multi-class (Lec 04)

**파일:** `Practice05_Logistic_regression_multiclass.ipynb`

| 섹션 | 내용 |
|------|------|
| 수식 정리 | softmax, CE, 그래디언트, notation ($i$=입력, $k$=클래스, $n$=샘플) |
| Step 1 | Toy 데이터 (6샘플, 3클래스, 2D) |
| Step 2 | 데이터 시각화 |
| Step 3 | 순전파: pre-activation → softmax → $\hat{y}=\arg\max$ |
| Step 4 | CE 손실: $\ell=-\log o_{y^{(n)}}$ |
| Step 5 | 오차 $o_k - y_k$ (CE+Softmax 핵심 결과) |
| Step 6 | 배치 그래디언트: $(\mathbf{O}-\mathbf{Y})^T\mathbf{X}$ |
| Step 7 | 가중치 갱신 |
| Step 8 | 갱신 후 재분류 검증 |
| Step 9 | 결정경계 시각화 (Before vs After, 쌍별 경계) |
| Step 10 | 다중 Epoch 학습 & 경계 변화 |
| 핵심 정리 | Binary vs Multi-class 비교표 |

- **데이터셋:** 합성 2D 3-class 분류 (N=6)
- **라이브러리:** numpy, matplotlib

---

## Practice 06 — Linear Model Summary (Lec 03-04)

**파일:** `Practice06_Linear_Model_Summary.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. Linear Regression | Diabetes 데이터 (BMI 1개 특징), MSE, R² |
| 2. Perceptron | Iris 2-class (sepal 2개 특징), step |
| 3. Logistic Binary | 동일 Iris 2-class, sigmoid, BCE |
| 4. Logistic Multi | Iris 3-class (4개 특징), softmax, CE |
| 5. 비교 요약 | 4개 모델 활성함수/손실/출력 비교표 |

각 섹션: 데이터 로드 → 학습 파라미터 → 학습 → 평가 + subplot(1,2) 시각화

- **데이터셋:** Diabetes (sklearn), Iris (sklearn)
- **라이브러리:** numpy, pandas, matplotlib, sklearn

---

## Practice 06 Extra — ME Applications (HW)

**파일:** `HW_answer/Practice06_extra_ME_Applications.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. Linear Regression | Auto MPG (UCI #9): 차량 무게 → 연비 예측, 표준화 |
| 2. Perceptron | Steel Plates (UCI #198): 2-class 2-feature 결함 분류 |
| 3. Logistic Binary | 동일 Steel Plates, 확률 출력 |
| 4. Logistic Multi | Steel Plates 3-class (K_Scratch, Bumps, Z_Scratch), 4-feature |
| 5. 비교 요약 | 4개 모델 비교표 |

- **데이터셋:** Auto MPG (UCI #9, NaN 제거), Steel Plates Faults (UCI #198, one-hot→argmax)
- **라이브러리:** numpy, pandas, matplotlib, ucimlrepo

---

## Practice 07 — PyTorch Basics (Lec 05)

**파일:** `Practice07_Pytorch_Basics.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. Tensor 기초 | `requires_grad=True`, 계산 그래프 |
| 2. Autograd | $y=\theta^2+2\theta+1$ 미분, chain rule |
| 3. Autograd로 GD | 수동 가중치 업데이트 linear regression |
| 4. torch.optim | optimizer 활용 linear regression |
| 비교 | NumPy 수동 vs PyTorch 수동 vs torch.optim |

- **데이터셋:** 합성 1D 선형 ($y=0.3x+0.1+\text{noise}$, N=50)
- **라이브러리:** numpy, torch, matplotlib

---

## Practice 08 — Tensor Manipulation (Lec 05)

**파일:** `Practice08_Tensor_Manipulation.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. NumPy → PyTorch | 변환, 동일 API 비교 |
| 2. 텐서 연산 | 원소별 곱 vs 행렬 곱, broadcasting |
| 3. 축별 집계 | mean, sum, max, argmax (dim 지정) |
| 4. Shape 변환 | view, reshape, squeeze, unsqueeze |
| 5. 기타 연산 | type casting, cat, stack, scatter_ (one-hot) |

- **데이터셋:** 예제 텐서만 사용
- **라이브러리:** numpy, torch

---

## Practice 09 — PyTorch Linear Models (Lec 05)

**파일:** `Practice09_Pytorch_Linear_Models.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. Linear Regression | `nn.MSELoss()`, SGD optimizer |
| 2. Binary Classification | `nn.BCEWithLogitsLoss()` (sigmoid 내장) |
| 3. Multi-class Classification | `nn.CrossEntropyLoss()` (softmax+NLL 내장) |
| 비교 | logit → 활성함수 → loss 관계 정리 |

Lec 03-04의 NumPy 수동 구현을 `torch.nn`으로 재구현.

- **데이터셋:** 합성 데이터
- **라이브러리:** torch, torch.nn, torch.optim, matplotlib

---

## Practice 10 — Multilayer Perceptron (Lec 06)

**파일:** `Practice10_Mulilayer_Perceptron.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. XOR 문제 | 단층으로 풀 수 없는 이유 시연 |
| 2. PyTorch MLP로 XOR 해결 | `nn.Module` class 로 2→4→1 MLP, BCEWithLogitsLoss |
| 3. Criterion 변화 | Binary vs Multi-class (`nn.Module` class 통일) — XOR / Iris |
| 4. `nn.Sequential` 소개 | 같은 모델을 두 방식으로 정의, 같은 seed → 동일 결과 확인 |

**아키텍처 예시:**
```
Input(d) → Linear(p) → Sigmoid/ReLU → Linear(c)
```

- **데이터셋:** XOR (합성, N=4), Iris (sklearn)
- **라이브러리:** torch, torch.nn, torch.optim, numpy, matplotlib, sklearn
- **수동 backpropagation 수식·NumPy 구현은 제거** (autograd 사용)

---

## Practice 11 — Dataloader (Lec 07)

**파일:** `Practice11_Dataloader.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. 데이터 준비와 분할 | train(60%) / val(20%) / test(20%), `stratify=y` |
| 1.1 데이터 정규화 | 표준화 ($\mu$, $\sigma$ train에서만 계산) |
| 2. TensorDataset & DataLoader | NumPy → Tensor → Dataset → DataLoader |
| 2.1 TensorDataset | FloatTensor(X), LongTensor(y) 묶기 |
| 2.2 DataLoader | batch_size, shuffle 설정 |
| 2.3 Batch/Iteration/Epoch | 용어 정의와 계산 예시 |

학습은 다음 노트북 P12에서 진행. 이 노트북은 **데이터 준비 파이프라인만** 다룹니다.

- **데이터셋:** Digits (sklearn, 1797샘플, 64특징, 10클래스)
- **라이브러리:** torch, torch.utils.data, sklearn, matplotlib

---

## Practice 12 — Training Pipeline (Lec 07)

**파일:** `Practice12_Training_Pipeline.ipynb`

P11 의 DataLoader 를 사용해 가장 간단한 학습 파이프라인을 구축. 후속 노트북 (P13~P16) 의 표준 패턴.

| 섹션 | 내용 |
|------|------|
| (데이터) | P11 패턴을 한 셀에 압축 — Digits + split + 표준화 + DataLoader |
| 1. 모델 정의 | `nn.Module` 상속 클래스 + `.to(device)` |
| 2. Loss + Optimizer | `CrossEntropyLoss` + Adam |
| 3. `train` + `evaluate` 함수 | **이 노트북에서 처음 정의, P13~P16 에서 그대로 재사용** |
| 4. 학습 | `train(model, train_loader, test_loader, optimizer, epochs=50, device=device)` 한 줄 |
| 5. 학습 곡선 시각화 | subplot(1,2): Train/Test Loss, Test Accuracy |

**과적합 데모 / Confusion Matrix / Misclassified 시각화는 제거** (각각 P13b, 별도 진단 영역으로 분리).

**train/evaluate 시그니처 (P13~P16 통일):**
```python
def train(model, train_loader, test_loader, optimizer, epochs, device):
    # epoch 루프 + 매 epoch evaluate 호출 → (train_losses, test_losses, test_accs)

def evaluate(model, loader, device):
    # model.eval() + torch.no_grad() → (loss, accuracy)
```

- **데이터셋:** Digits (sklearn)
- **라이브러리:** torch, torch.nn, torch.optim, sklearn, matplotlib

---

## Practice 13 — Training Example (Lec 07)

**파일:** `Practice13_Training_Example.ipynb`

P11+P12 의 기본 파이프라인에 **일반적으로 도입되는 4가지 기법** (He 초기화 + BatchNorm + Dropout + L2) 을 한꺼번에 적용해서, Baseline vs Improved 학습 곡선을 한 셀에서 비교. 각 기법의 세부는 P13a (Optimization) / P13b (Regularization) 에서.

| 모델 | 적용 기법 |
|------|----------|
| **Baseline** | 기본 MLP, 추가 기법 없음 |
| **Improved** | He 초기화 + BatchNorm1d + Dropout(0.3) + L2 (weight_decay=1e-3) |

같은 데이터·같은 옵티마이저·같은 epoch 에서 Test Accuracy 가 어떻게 달라지는지 한눈에 보여주는 **compact 예시**.

- **데이터셋:** Digits (sklearn)
- **라이브러리:** torch, torch.nn, torch.optim, sklearn, matplotlib

---

## Practice 13a — Optimization (선택 심화)

**파일:** `Practice13a_optional_Optimization.ipynb`

P13 에서 본 표준 기법들 중 **최적화 관련** 부분을 깊게 다룹니다. 각 요소를 한 번에 한 가지씩 격리해서 A vs B 비교.

| 섹션 | A vs B |
|------|--------|
| 1. 데이터 전처리 | skewed vs standardized |
| 2. 가중치 초기화 | random vs He (Kaiming) |
| 3. 옵티마이저 비교 | SGD vs Adam |
| 4. 활성 함수 | Sigmoid vs ReLU (gradient vanishing) |
| 5. 배치 정규화 | BatchNorm1d 적용 전후 |

**아키텍처 (8-layer deep):** `Input(64) → [Linear(128) → ReLU] ×7 → Linear(10)`

- **데이터셋:** Digits (sklearn)
- **라이브러리:** torch, torch.nn, torch.optim, torch.nn.init, sklearn, matplotlib

---

## Practice 13b — Regularization (선택 심화)

**파일:** `Practice13b_optional_Regularization.ipynb`

P13 에서 본 표준 기법들 중 **규제(Regularization) 관련** 부분을 깊게 다룹니다. 과적합을 명확히 시연한 뒤 각 기법으로 어떻게 완화하는지 비교.

| 섹션 | 내용 |
|------|------|
| **0. 과적합 시연** | 큰 모델 (563K params) + 70 train samples → train 100% / val ~80% (학습 곡선 시각화) |
| 1. L2 규제 | `weight_decay` 파라미터 |
| 2. L1 규제 | 가중치 분포 비교 (histogram) |
| 3. 드롭아웃 | `nn.Dropout(p=0.3)`, train/eval 차이 시연 |
| 4. 하이퍼파라미터 탐색 | LR×WD Grid Search (heatmap) |

의도적 과적합: 본문 학습은 train 10%만 사용 (179샘플), test 1618샘플.

- **데이터셋:** Digits (sklearn)
- **라이브러리:** torch, torch.nn, torch.optim, torch.nn.init, numpy, sklearn, matplotlib

---

## Practice 14 — Dataloader for Image (Lec 08~09)

**파일:** `Practice14_Dataloader_for_Image.ipynb`

P11 의 표(table) 데이터 파이프라인을 **이미지 데이터** 로 확장. 학습된 모델은 다루지 않고 **데이터 준비 패턴만** 다룹니다.

| 섹션 | 내용 |
|------|------|
| 1. torchvision 내장 데이터셋 | `datasets.CIFAR10` + 기본 transform (ToTensor, Normalize) |
| 2. Custom Dataset 클래스 | 9줄 boilerplate template — `__init__`/`__len__`/`__getitem__` |
| 3. ImageFolder | 폴더 구조에서 자동 로드 (자기 사진도 그대로 활용) |
| 4. Augmentation transform | `RandomCrop(32, pad=4)`, `RandomHorizontalFlip` — 같은 이미지 4번 증강 시각화 |

**ImageFolder 디렉터리 구조 (Section 3):**
```
./data/cifar5_imagefolder/
├── train/  plane/ car/ bird/ cat/ deer/   (100장씩)
└── test/   plane/ car/ bird/ cat/ deer/   (50장씩)
```
→ GitHub Release 에서 자동 다운로드 (~2MB). 학생이 자기 사진을 같은 구조로 두면 그대로 동작.

- **데이터셋:** CIFAR-10 (torchvision 내장) + CIFAR5 ImageFolder (5클래스 부분집합)
- **라이브러리:** torch, torchvision (`datasets.CIFAR10`, `datasets.ImageFolder`, `transforms`), PIL, matplotlib

---

## Practice 15 — CNN Basics (Lec 08)

**파일:** `Practice15_CNN_Basics.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. MLP의 한계 | 28x28 → 784 평탄화 시 공간정보 소실, 1픽셀 시프트 효과 |
| 2. Convolution 직관 | 수동 3x3 합성곱 (NumPy 슬라이딩) → `nn.Conv2d` 검증, shape 공식 |
| 3. Pooling | `MaxPool2d(2)` 입출력 시각화 |
| 4. CNN 모델 | `class CNN(nn.Module)` + `self.layer1`, `self.layer2` (lab-11-1 스타일), 단계별 shape 출력 |
| 5. MNIST 파이프라인 | `torchvision.datasets.MNIST` 부분집합 (10000/2000), 샘플 10장 시각화 |
| 6. 학습 | P12 의 `train` + `evaluate` 함수 그대로, Adam, 3 epoch |
| 7. MLP vs CNN 비교 | 같은 데이터에 두 모델 학습 — CNN 94.9% vs MLP 89.2% |
| 8. (보너스) | 같은 CNN을 `nn.Sequential` 한 줄로 작성 — 두 스타일 비교 |
| 9. (도전) | 전체 MNIST 60k + 8 epoch → **~99% 도달** |

**아키텍처 (CNN):**
```
Input (N, 1, 28, 28)
  layer1: Conv2d(1->32, k=3, p=1) -> ReLU -> MaxPool(2)   -> (N, 32, 14, 14)
  layer2: Conv2d(32->64, k=3, p=1) -> ReLU -> MaxPool(2)  -> (N, 64,  7,  7)
  flatten + Linear(64*7*7, 10)
```

- **데이터셋:** MNIST (60k train + 10k test, 1×28×28, 10 클래스, 부분집합 사용)
- **라이브러리:** torch, torch.nn, torchvision, matplotlib
- **GPU 자동 활용:** `.to(device)` 패턴 (이 노트북부터 본격 시작)

---

## Practice 16 — Advanced CNN (Lec 09)

**파일:** `Practice16_Advanced_CNN.ipynb`

데이터셋: **CIFAR-10 (32x32 컬러 3채널, 10클래스)**. **두 모델만** compact 하게 비교 — 모던 CNN 기법을 모두 합친 Custom Final 과 ImageNet 사전학습 ResNet18.

**중요:** 두 모델은 **정확히 같은 `train_loader`, `val_loader`** 위에서 학습. 격리되는 차이는 모델 구조 + (의도된) optimizer · epoch 수 뿐.

| 항목 | 값 (두 모델 공통) |
|------|------------------|
| 데이터 | CIFAR-10 10000장(train) / 2000장(val) subset (속도 절충) |
| Input | Resize 96 + ImageNet mean/std 정규화 |
| Augmentation (train) | RandomCrop(96, padding=8) + RandomHorizontalFlip(p=0.5) |
| Batch size | 64 |

| 모델 | 구성 | Optimizer | Epoch |
|------|------|-----------|-------|
| **Custom Final CNN** | Conv 4-layer + Kaiming + BN + Dropout(0.5), from scratch | SGD(lr=0.01, momentum=0.9) | 15 |
| **Pretrained ResNet18 (partial freeze)** | conv1+layer1+layer2 동결, layer3+layer4+새 FC fine-tuning | SGD(lr=0.001, momentum=0.9), trainable 만 | 5 |

**Layer freezing 패턴 (Section 4):** `for p in resnet.parameters(): p.requires_grad = False` → `for p in resnet.layer3.parameters(): p.requires_grad = True` 같이 단계적으로 동결/해제. 앞쪽 layer 는 일반 시각 특징 (엣지·코너·텍스처) 으로 이미 충분 → 재학습 불필요.

**섹션 구성:**
1. CIFAR-10 데이터 파이프라인 (공용 train_loader / val_loader)
2. `train` / `evaluate` 함수
3. Custom Final CNN — 정의 + 학습
4. Pretrained ResNet18 — 정의 + fine-tuning
5. 비교 (val loss / acc 곡선)
6. 훈련 결과 시각화 (틀린 예측 4개 + Confusion Matrix, ResNet18 기준)
7. 정리

**핵심 메시지:** 데이터·전처리·aug 가 정확히 동일한 조건에서, **ImageNet 사전학습 backbone 의 fine-tuning** 이 from-scratch 4-layer CNN 보다 더 짧은 학습으로 더 높은 정확도 달성.

각 기법(깊이·BN·Dropout·Augmentation) 의 격리 비교 실험과 학습된 필터/feature map 시각화는 **Practice 16a** 에서 별도로 다룸.

- **데이터셋:** CIFAR-10 (10000장 train + 2000장 val subset)
- **라이브러리:** torch, torch.nn, torchvision (`datasets.CIFAR10`, `transforms`, `models.resnet18`), PIL, matplotlib

---

## Practice 16a — Advanced CNN (선택 심화)

**파일:** `Practice16a_optional_Advanced_CNN.ipynb`

P16 에서 한꺼번에 적용한 기법들을 **한 번에 한 가지씩 격리**해 효과를 비교. 학습된 필터·feature map 시각화 포함.

**옵티마이저:** SGD(lr=0.01, momentum=0.9). 모델 간 차이가 명확히 드러나도록 동일 설정.

| 섹션 | A | B | 결과 (peak) |
|------|---|---|------------|
| 3-1. 깊이 효과 | 얕은 (2-layer) | 깊은 (4-layer, Kaiming init) | Deep ≈ Shallow (과적합 동반) |
| 3-2. BatchNorm | 깊은 | 깊은 + BN | **BN +몇 %p** |
| 3-3. Dropout | 깊은+BN | 깊은 + BN + Dropout(0.5) | **Drop 향상** |
| 4. 데이터 증강 | 깊은+BN+Drop | 같은 모델 + RandomCrop + HFlip | **Aug 향상** |
| 5. Custom CNN 시각화 | — | 첫 conv 3x3 필터 (8개) + feature map | random-looking 필터 |
| **6. ResNet18 fine-tune** | 학습 전 baseline (random FC, ~10%) | Pretrained ResNet18, Resize(96), 5 epoch | **~10% → ~89%** (CUDA ~30초) |
| 7. 훈련 결과 시각화 | — | 틀린 예측 + Confusion Matrix | — |
| 8. ResNet18 시각화 | — | 첫 conv 7x7 필터 (8개) + feature map vs Custom CNN | **edge/color detector** (Gabor 모양) |

**핵심 패턴 (Conv-BN-ReLU 블록):**
```
Conv2d -> BatchNorm2d -> ReLU
```

**Custom CNN 아키텍처 (CNN_Deep_BN_Drop, 입력 3채널 32x32, Kaiming init):**
```
[Conv-BN-ReLU-Pool] x3 -> [Conv-BN-ReLU] -> Flatten -> Dropout(0.5) -> Linear(128*4*4, 10)
```

- **데이터셋:** CIFAR-10 (10000장 일관)
- **라이브러리:** torch, torch.nn, torchvision, PIL, matplotlib

---

## Appendix A — PyTorch Lightning (부록)

**파일:** `Practice_Appendix_A_PyTorch_Lightning.ipynb`

P12 의 수동 훈련 루프를 **PyTorch Lightning** 으로 감싸 같은 학습을 재현. 같은 데이터·같은 모델·같은 결과지만 코드 구조가 어떻게 갈라지는지 보여주는 부록.

| 섹션 | 내용 |
|------|------|
| 1. 수동 → Lightning 매핑 | 수동 루프의 각 부분이 `LightningModule`의 어디로 옮겨가는지 표 |
| 2. 데이터 준비 | P11~P12 와 동일 (Digits, 60/20/20, 표준화, DataLoader) |
| 3. `LightningModule` 정의 | `training_step` / `validation_step` / `configure_optimizers` |
| 4. `Trainer` 로 학습 | `Trainer(max_epochs=50).fit(model, ...)` 한 줄, CSVLogger 자동 |
| 5. 학습 곡선 시각화 | `metrics.csv` → P12 와 같은 subplot(1,2) |
| 6. 테스트 + Confusion Matrix | `trainer.test(...)` + inference |
| 7. 책임 분리 비교표 | 사용자 정의 vs Lightning 자동 |

- **데이터셋:** Digits (sklearn)
- **라이브러리:** torch, torch.nn, lightning.pytorch, pandas, sklearn, matplotlib

---

## Appendix B — TensorFlow / Keras (부록)

**파일:** `Practice_Appendix_B_TensorFlow_Keras.ipynb`

P12 의 PyTorch 학습 파이프라인을 같은 데이터·같은 모델로 **TensorFlow / Keras** 에서 재현. 두 가지 추상화 (`tf.GradientTape` 수동 vs `model.fit()` 고수준) 를 병렬로 제시.

| 섹션 | 내용 |
|------|------|
| 1. 데이터 준비 | P11 패턴을 `tf.data.Dataset` 으로 |
| 2. 방법 A — `tf.GradientTape` | 수동 훈련 루프 (PyTorch 루프와 1:1 대응) |
| 3. 방법 B — `model.compile` + `model.fit` | 같은 학습을 두 줄로 |
| 4. PyTorch ↔ TF 매핑표 | 모델·데이터·loss·optimizer·훈련 루프 핵심 대조 |
| 5. 테스트 세트 평가 | `model.evaluate` + Confusion Matrix |
| 6. P13a/P13b 개념의 TF 대응 | 초기화·BatchNorm·L2/L1·Dropout 매핑표 |

- **데이터셋:** Digits (sklearn)
- **라이브러리:** tensorflow, keras, numpy, sklearn, matplotlib
- **실행 환경:** Colab 권장 (TF 사전 설치, GPU 무료). 로컬은 CPU TF 로 충분.

---

## Practice 99 — MLP Predictive Maintenance (응용)

**파일:** `Practice99_MLP_Predictive_Maintenance.ipynb`

| 섹션 | 내용 |
|------|------|
| 1. 데이터 로드 | AI4I 2020 (UCI #601), 클래스 불균형 (3.4% failure) |
| 2. Logistic 2-feature | RPM, Torque만 사용, NumPy 수동 구현 → ~75% |
| 3. Logistic 5-feature | 전체 5개 특징 → ~83% |
| 4. PyTorch MLP | 5→64→32→1 (ReLU), BCEWithLogitsLoss, Adam → ~96% |
| 5. 요약 | 선형 vs 비선형, 특징 수에 따른 성능 비교 |

클래스 균형: undersampling (362 normal + 362 failure = 724)
표준화: train set 기준 $\mu$, $\sigma$

- **데이터셋:** AI4I 2020 Predictive Maintenance (UCI #601, 5특징, 이진)
- **라이브러리:** torch, torch.nn, torch.optim, numpy, matplotlib, ucimlrepo
