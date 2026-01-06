 # 🏛️ Project Master Blueprint (v2.0)

## 0. Git Branch Policy
- **Target**: `https://github.com/graviton94/claim-analysis-engine/tree/main`
- **Rule**: 모든 코드 변경사항은 `main` 브랜치에 직접 커밋하거나, `main`으로 향하는 PR이어야 한다.

## 1. Data Strategy: Series Data Mart
- **분절화(Segmentation)**: 전체 데이터를 `[플랜트 | 제품범주2 | 중분류]` 단위로 쪼개어 `data/series/`에 개별 파일로 저장.
- **효율성**: 전체 대용량 파일을 읽지 않고, 업데이트가 필요한 시리즈 파일만 접근하여 속도 최적화.

## 2. Detection Engine (Hybrid)
### 2.1 Rule-based (P6 & P2)
- 사용자가 설정한 조건(예: 특정 제품 건수 > N건)을 실시간 쿼리하여 감지.
### 2.2 ML-based (P4)
- **Top-down**: `플랜트 | 대분류` 단위로 챔피언 모델 학습.
- **Seasonal Allocation**: 과거 동월 비중을 활용하여 하위 피벗 행에 예측값 배분.
- **Risk Scoring**: 예측치의 기울기, 과거 Max 대비 비율을 분석하여 Warning Level 부여.

## 3. Analysis Intelligence (P3)
- **Outlier Detection**: IQR(Interquartile Range) 또는 Z-Score를 활용하여 시계열 중 튀는 값 강조.
- **Lag Analysis**: `제조일자`와 `접수일자` 사이의 차이를 계산하여 시차 분포 시각화.

## 4. Integration Logic
- P1에서 데이터 업로드 시 -> 영향받는 `data/series/` 파일만 갱신 -> P4 엔진이 해당 시리즈만 재학습 -> P2 리스트 자동 업데이트.
