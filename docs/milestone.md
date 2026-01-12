# 📅 Development Milestone (v3.0)

## Phase 1: Foundation (✅ 완료)
- [x] 대용량 데이터 파티셔닝 저장 구조 설계 (`core/storage`).
- [x] 기본 대시보드 및 업로드 파이프라인 구축.

## Phase 2: Intelligence Engine (✅ 완료)
- [x] **Risk Engine**: Nelson Rules 및 통계적 이상 탐지 구현 (`core/analytics`).
- [x] **Forecast Engine**: 앙상블 예측 및 영업일 보정 로직 구현 (`core/forecasting`).
- [x] **Detail Analysis**: 동적 피벗 및 Lag 분석 UI (`3_플랜트_분석`).

## Phase 3: Optimization & Stability (🚧 진행 중)
- [ ] **Hyperparameter Tuning**: Optuna를 활용한 모델 파라미터 최적화 자동화.
- [ ] **Performance**: 대용량 Series 연산 속도 개선 (Vectorization).
- [ ] **UX Polish**: 예측 신뢰구간 시각화 및 리스크 진단 텍스트 가독성 개선.

## Phase 4: Automation & Expansion (Next Step)
- [ ] **Auto-Reporting**: 매월 초 주요 이슈를 요약하여 이메일/슬랙 자동 발송.
- [ ] **Feedback Loop**: 사용자가 예측값이나 리스크 등급을 수정하면 이를 학습에 반영하는 피드백 파이프라인.
- [ ] **LLM Integration**: "지난달 A공장 이슈가 뭐였어?"와 같은 질문에 답하는 챗봇 인터페이스 (Gemini API 연동).