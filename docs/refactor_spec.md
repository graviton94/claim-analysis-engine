# **🏗️ Refactoring Technical Specification (v2.0)**

이 문서는 Phase 2.5 통합 및 Phase 2.8 고도화 작업의 기술 명세이다.

## **1\. Core Module Refactoring (✅ Completed)**

### **1.1 core/storage.py: 통합 로더**

* **Function**: load\_and\_filter\_data  
* **Role**: pyarrow.dataset을 활용한 지연 로딩 및 공통 필터링(플랜트/기간/모드) 수행.  
* **Status**: 적용 완료 (app.py, pages/3\_, pages/4\_ 전체 적용).

### **1.2 core/analytics.py: Zero-filling 전처리**

* **Function**: prepare\_risk\_data  
* **Role**: 분석 기준일로부터 과거 24개월 데이터를 강제 주입(Reindex with fill\_value=0)하여 급증 패턴 감지력 확보.  
* **Status**: 적용 완료.

## **2\. Dashboard Enhancement (✅ Completed)**

### **2.1 Color System**

* **Palette**: \#EF151E(Red), \#FF9700(Yellow), \#006ECD(Blue).  
* **Application**: KPI Metric 텍스트, Plotly 라인 차트, Risk Card 테두리 및 배지.

### **2.2 Visualization Logic**

* **Trend Chart**: 실적 라인(mode='lines+markers')과 예측 라인(mode='markers+lines')을 분리하여 시각적 단절(Disconnection) 구현. 과거와 미래의 경계를 명확히 함.  
* **Action Cards**: HTML/CSS 기반의 커스텀 카드 UI 적용. 엑셀 다운로드 및 정밀 분석 Deep Link 버튼 탑재.

## **3\. Future Plan (Phase 3 Spec)**

### **3.1 Predictive Engine Upgrade**

* **Target**: core/engine/trainer.py  
* **Feature**:  
  * Optuna 튜닝 횟수 및 파라미터 범위 확장.  
  * Prophet의 changepoint\_prior\_scale 동적 조정 로직 추가.  
  * 예측 결과의 신뢰구간(Confidence Interval) 시각화.

### **3.2 AI Insight (Natural Language Generation)**

* **Target**: app.py & pages/3\_플랜트\_분석.py  
* **Feature**: Rule-based 텍스트 생성에서 LLM 기반 요약으로 전환 (API Key 연동 필요).
