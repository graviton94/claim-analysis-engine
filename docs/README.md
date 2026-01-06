# 🏆 Advanced Claim Analysis & Early Warning System (v2.0)

> **⚠️ 작업 브랜치**: 모든 작업은 반드시 **[main 브랜치](https://github.com/graviton94/claim-analysis-engine/tree/main)**에서 수행합니다.

## 🎯 Project Vision
단순한 통계 조회를 넘어, **비즈니스 규칙(Rule)**과 **인공지능(ML)**이 결합된 하이브리드 조기 경보 시스템을 구축합니다. 수만 개의 데이터 조합을 스스로 감시하고 이상 징후를 선제적으로 보고합니다.

## 🖥️ Page Navigation
1. **수동 업로드**: CSV/Excel 표준화 적재 및 연/월 파티셔닝.
2. **통합 요약 (Summary)**: 전사 추이 및 고위험 클레임 감지 리스트(Rule+ML).
3. **플랜트별 상세 분석**: 동적 피벗, **이상치(Outlier) 감지**, **Lag(제조~접수 시차) 분석**.
4. **예측 엔진 관리**: 전수 시리즈 스캔, 챔피언 모델 업데이트 및 정확도 레포트.
5. **매출 수량 관리**: PPM 산출을 위한 플랜트별 매출 데이터 관리.
6. **감지 대상 관리**: 사용자 정의 위험 조건(Rule) 설정 및 P2 연동.

## 📂 Directory Structure
```bash
claim-prediction-system/
├── core/
│   ├── engine/           # ML 모델 및 배치 엔진 (SARIMAX, CatBoost, LSTM)
│   ├── etl.py            # 54개 필드 추출 및 정제
│   └── storage.py        # Parquet 파티셔닝 및 시리즈 분절 저장
├── pages/
│   ├── 1_데이터_업로드.py
│   ├── 2_통합_요약.py     # New: Executive Summary
│   ├── 3_플랜트_분석.py   # Update: Outlier & Lag 분석 추가
│   ├── 4_예측_페이지.py    # New: 엔진 관리 및 리스크 스캐너
│   ├── 5_매출_관리.py
│   └── 6_감지_대상_관리.py # New: 사용자 정의 규칙 설정
├── data/
│   ├── hub/              # Raw 파티션
│   ├── series/           # 분절된 시계열 데이터 마트 (JSON/Parquet)
│   └── results/          # 최종 리스크 마킹 결과 (alerts.json)
└── docs/                 # 설계 문서

```

## 📂 Field def
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2
