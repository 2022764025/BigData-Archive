# BigData-Archive

## ML Case Study: Titanic Survival Prediction

> **Data-Driven Approach to Predictive Modeling & Feature Engineering**

본 프로젝트는 타이타닉 침몰 사고 데이터를 활용하여 생존 여부를 분류하는 머신러닝 파이프라인을 구축한 사례이다. 전처리 전략의 수립부터 모델 최적화, 그리고 XAI를 통한 결과 해석까지의 엔드 투 엔드(End-to-End) 워크플로우를 포함하고 있다.

---

## 1. Project Objectives

- **Pipeline 기반 자동화**: 전처리 및 학습 과정을 `Pipeline` 객체로 통합하여 모델의 안정성 확보

- **도메인 기반 특성 공학**: 승객의 호칭(Title) 및 가족 관계(FamilySize)를 분석하여 유의미한 파생 변수 생성

- **성능 지표 다각화**: 단순 정확도를 넘어 F1-score, ROC-AUC 등 다중 지표를 통한 모델 검증

## 2. Experimental Design (Preprocessing Strategy)

| 실험 ID | 결측치 처리 | 인코딩 전략 | 스케일링 | 변수 선택 | Accuracy (CV) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Base** | N/A | N/A | N/A | N/A | - |
| **Exp-1** | Mean | One-Hot | Standard | X | 0.8036 |
| **Exp-2 (Final)** | **Median** | **Ordinal** | **MinMax** | **O** | **0.8092** |
| **Exp-3** | Most_Freq | One-Hot | Robust | O | 0.8081 |

## 3. Final Model Performance

실험 결과 가장 우수한 성능을 보인 **XGBoost(Exp-2)** 모델의 최종 평가 지표이다.

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Accuracy** | **0.7207** | 모델의 전체 정답률 |
| **F1-score** | **0.6094** | 정밀도와 재현율의 균형 지표 |
| **ROC-AUC** | **0.7806** | 모델의 이진 분류 변별력 |

## 4. Model Interpretation (SHAP)

**SHAP(SHapley Additive exPlanations)** 분석을 통해 모델이 생존 확률을 판단한 핵심 근거를 시각화하였다.

- **주요 변수**: `Embarked`, `FamilySize`, `Pclass` 등이 예측에 결정적인 기여를 함.

- **인사이트**: 1등석 승객(`Pclass`) 및 높은 요금(`Fare`) 결제 승객의 생존 기여도가 높음을 통계적으로 확인.

## 5. Key Learning & Conclusion

1. **중앙값(Median)의 강건함**: 비대칭 분포 데이터(Age) 처리에 있어 평균보다 중앙값 대체가 일반화 성능 향상에 기여함.

2. **인코딩의 선택**: 트리 기반 모델에서는 One-Hot 인코딩보다 특성 공간을 압축적으로 사용하는 **Ordinal 인코딩**이 더 효율적임을 입증함.

3. **특성 공학의 위력**: 파생 변수(`Title`, `FamilySize`)가 모델 중요도 상위에 랭크되며, 성능 개선의 가장 핵심적인 동력으로 작용함.

4. **변수 선택의 트레이드오프**: Feature Selection을 통해 모델의 복잡도를 낮추어 과적합 위험을 감소시킴.

## 6. Technology Stack

- **Language**: Python 3.10+

- **Framework**: Scikit-learn, XGBoost, SHAP

- **Environment**: VS Code, Jupyter Notebook, GitHub

---