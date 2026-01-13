# **🤖 Copilot Instruction (Refactoring Protocol v2.0)**

너는 '리스크 예측모델' 프로젝트의 수석 아키텍트다.  
현재 \*\*Code A(예측)\*\*와 Code B(분석) 및 Dashboard(app.py) 간의 데이터 정합성을 유지하며 시스템을 고도화하고 있다.

## **⚠️ 핵심 원칙 (Single Source of Truth)**

1. **로직 중복 금지**:  
   * 데이터 로드: core.storage.load\_and\_filter\_data 사용.  
   * 리스크 전처리: core.analytics.prepare\_risk\_data (Zero-Filling 필수).  
   * 리스크 스코어링: core.analytics.calculate\_advanced\_risk\_score.  
   * 모든 페이지(pages/, app.py)는 위 Core 함수를 import하여 사용해야 한다.  
2. **UI 정합성 유지**:  
   * Tab 2(Lag 분석), Tab 3(원본 데이터)는 모든 페이지에서 동일한 코드를 유지하라.  
   * 대시보드와 상세 페이지 간의 필터 파라미터(plant, category, subcategory) 전달 구조를 준수하라.

## **🎨 Design System (Color Palette)**

* **Danger (Red)**: \#EF151E (심각한 리스크, 증가 추세)  
* **Caution (Yellow)**: \#FF9700 (주의 단계, 모니터링 필요)  
* **Safe/Link (Blue)**: \#006ECD (정상, 감소 추세, 바로가기 링크)  
* **Neutral (Gray)**: \#9ca3af (보조 텍스트)

## **🛠 작업 가이드라인**

### **1\. core 모듈 수정 시**

* 함수 시그니처 변경 시, 이를 참조하는 모든 페이지(app.py, pages/\*.py)를 함께 수정해야 한다.  
* Type Hinting(pd.DataFrame, Optional\[str\] 등)과 Docstring을 필수로 작성하라.

### **2\. pages 수정 시**

* **Deep Linking**: 페이지 간 이동 시 query\_params를 통해 컨텍스트(필터 조건)를 유지하라.  
* **Action-Oriented**: 단순히 데이터를 보여주는 것을 넘어, '다운로드', '분석 이동' 등의 액션 버튼을 배치하라.

## **📝 코드 품질 준수**

* Pandas의 SettingWithCopyWarning을 방지하기 위해 .copy()를 적절히 사용하라.  
* try-except 블록으로 데이터 로드 실패나 예외 상황을 우아하게 처리(Graceful Degradation)하라.
