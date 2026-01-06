# 📅 Development Milestone

## Phase 1: Data Foundation & Sales Integration (D+1~2)
- [ ] `core/storage.py`: `save_partitioned(df, ['접수년', '접수월'])` 함수 구현.
- [ ] `core/etl.py`: 54개 필드 강제 추출 및 인코딩(`utf-8-sig`) 처리.
- [ ] `pages/1_데이터_업로드.py`: 대용량 파일 청크 처리 및 파티션 저장.
- [ ] `pages/2_매출수량_관리.py`: **[신규]** 플랜트/년/월별 매출수량 CRUD(Create, Read, Update, Delete) UI 구현.

## Phase 2: Pivot Dashboard Implementation (D+3~4)
- [ ] `pages/3_플랜트_분석.py`: 
  - [ ] **플랜트 필터** 최상단 배치.
  - [ ] **Dynamic Pivot**: `groupby` 대상을 사용자가 선택(`st.multiselect`)하는 로직 구현.
  - [ ] 매출 데이터 연동: `클레임건수 / 매출수량` 자동 계산 로직 추가.

## Phase 3: ML/DL Engine & Optuna (D+5~6)
- [ ] `core/engine/models.py`: CatBoost, LSTM, SARIMAX 모델링.
- [ ] `core/engine/trainer.py`: Optuna 하이퍼파라미터 튜닝 (매출수량 피처 포함).
- [ ] `pages/4_예측_시뮬레이션.py`: 챔피언 모델 선정 결과 및 향후 6개월 예측 시각화.

## Phase 4: Integration (D+7)
- [ ] 전체 데이터 파이프라인(업로드 → 매출입력 → 피벗분석 → 예측) 통합 테스트.
