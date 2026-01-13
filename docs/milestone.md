# 📅 Development Milestone (v3.0)

## Phase 1 & 2: Foundation & Intelligence (✅ 완료)
- [x] 대용량 데이터 파티셔닝 저장 (`core/storage`).
- [x] **Risk Engine**: Nelson Rules 및 통계적 이상 탐지 구현 (`core/analytics`).
- [x] **Fast Forecast**: 대시보드용 앙상블 예측 로직 구현 (`forecasting.py`).

## Phase 3: Forecast Stabilization (🚧 진행 중 - Priority High)
- [ ] **Data Guard**: `forecasting.py` 학습 시 '진행 중인 당월 데이터' 자동 제외 로직 적용.
- [ ] **Weight Logic**: 월초/월말 가중치 동적 변화 로직 튜닝 및 검증.
- [ ] **UI Connection**: `app.py` 메인 카드에 `forecasting.py` 결과 연동 (에러 핸들링 포함).

## Phase 4: Simulation Lab Rebuilding (예정 - Next Sprint)
- [ ] **Legacy Cleanup**: 작동하지 않는 `4_예측_시뮬레이션.py`의 구형 코드를 `core/engine` 구조에 맞춰 리팩토링.
- [ ] **Lab UI**: 사용자가 기간/모델을 선택하는 제어 패널(Control Panel) 구축.
- [ ] **Visualizer**: 과거 실제값 vs 시뮬레이션 예측값을 겹쳐보는 **Backtesting 차트** 구현.

## Phase 5: Automation (Future)
- [ ] **Auto-Reporting**: 매월 1일, 고위험군(Red Grade) 자동 리포트 생성.
- [ ] **Feedback Loop**: 사용자가 예측값을 수정하면 이를 보정 계수로 저장.