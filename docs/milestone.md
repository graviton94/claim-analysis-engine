# 📅 Development Milestone (v2.0)

## Phase 1: Foundation (✅ 완료)
- [x] 54개 필드 표준화 및 연/월 파티셔닝 저장.
- [x] 매출 수량 관리 UI 및 데이터 연동.

## Phase 2: Deep Dive Analysis (D+1~2)
- [ ] **P3 플랜트 분석** 고도화: 동적 피벗 UI 재구축.
- [ ] **이상치 감지** 로직: 테이블 내 튀는 값 하이라이트.
- [ ] **Lag 분석**: 제조~접수 시차 분포 차트 구현.

## Phase 3: Rule & Summary (D+3~4)
- [ ] **P6 감지 대상 관리**: 사용자 정의 규칙 설정 및 저장 로직.
- [ ] **P2 통합 요약**: 최신 현황 요약 및 P6 규칙 기반 고위험 리스트 출력.
- [ ] **요약 레포트**: 리스트 클릭 시 상세 정보 팝업 구현.

## Phase 4: ML Intelligence (D+5~6)
- [ ] **P4 예측 페이지**: 시리즈 분절 배치 스캔 엔진 (`batch.py`).
- [ ] **리스크 스캐너**: 챔피언 모델 선정 및 계절성 배분 로직 완성.
- [ ] **Warning Marking**: 예측 기반 위험 등급 산출 및 `alerts.json` 저장.

## Phase 5: Final Optimization (D+7)
- [ ] 업로드 시 증분 업데이트 트리거 연결.
- [ ] 전사 통합 테스트 및 UI/UX 비주얼 고도화.
