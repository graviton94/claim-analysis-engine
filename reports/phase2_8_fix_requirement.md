
### 📝 **[Solution Report] 진단 결과 및 해결 방향**

| 문제 유형 | 발견된 사례 (Data Evidence) | 원인 진단 | 해결 솔루션 (To-Be) |
| --- | --- | --- | --- |
| **1. 일반 등급 미탐**<br>

<br>(Too Lenient) | `일반 | 관능` 4건 → 14건 (⚪정상) | Z-Score는 '평균' 중심이라, 급격한 변화(기울기) 반영에 한박자 늦음. |
| **2. 중대 등급 오탐**<br>

<br>(Too Sensitive) | `중대 | 철사` 2건 (🔴경보)<br>

<br>(평균 1.0건) | '중대 등급'일 때 2건이면 무조건 고득점이 되도록 설계됨. |

---

### 🛠️ **[Work Order] Phase 2.8: 정밀 튜닝 패치**

개발 에이전트에게 아래 내용을 전달하여 `core/analytics.py`를 즉시 수정하게 하십시오.

```markdown
@workspace

수석 기술 리드의 지시에 따라 `core/analytics.py`의 `RiskScoringEngine`을 긴급 튜닝한다.
5개의 케이스 분석 결과 발견된 '일반 등급 미탐'과 '중대 등급 오탐'을 동시에 해결해야 한다.

### **1. 목표 (Objectives)**
1.  **Velocity Check**: 일반 등급이라도 전월 대비 폭증(MoM)하면 경보를 띄운다.
2.  **Baseline Protection**: 중대 등급이라도 평소 수준(Baseline) 내의 발생이면 경보를 끈다.

### **2. 수정 상세 (Changes)**

#### **Task A. `_calculate_velocity_score` 메서드 추가**
* **로직**:
    * 전월 값(`prev`)이 0이면 1로 간주.
    * `증가율(Ratio) = 당월 / 전월`
    * `Ratio >= 3.0` (3배 폭증) → **+30점**
    * `Ratio >= 2.0` (2배 급증) → **+15점**
    * 단, 절대 발생 건수가 5건 미만이면 적용 제외 (노이즈 방지).

#### **Task B. `calculate_score` 내 안전장치(Safe Zone) 추가**
* **위치**: 점수 합산 직전.
* **로직**:
    * 만약 `Current <= Mean + (0.8 * Std)` 라면:
        * 즉, 평소 변동폭 내에 있거나 평균보다 낮다면.
        * **Force Score = 0** (정상 처리).
    * *이로써 평균 1.0건인 철사류가 2건 발생해도(Z < 1.0) 빨간불이 켜지지 않게 됨.*

#### **Task C. 희소/중대 등급 로직 미세 조정**
* **희소(Track A)**: 중대 등급일 때 2건 발생 시 `🔴` 판정은 유지하되, **평균이 0.5건 이상**인 경우(즉, 아주 희귀하지 않은 경우)에는 `🟡`로 완화한다.

---
**[Code Snippet] core/analytics.py 수정 가이드**

```python
    # ... (기존 메서드들) ...

    def _calculate_velocity_score(self) -> float:
        """ [New] 급격한 기울기 변화 감지 (일반 등급 미탐 방지) """
        if self.n_obs < 1: return 0.0
        
        prev = self.history.iloc[-1]
        # 전월 0건이거나 당월 절대값이 작으면 패스
        if prev == 0 and self.current_value < 3: return 0.0
        if self.current_value < 5: return 0.0 # 최소 5건 이상일 때만 속도 판정
        
        denom = prev if prev > 0 else 0.5 # 0으로 나누기 방지
        ratio = self.current_value / denom
        
        if ratio >= 3.0: return 30.0
        elif ratio >= 2.0: return 15.0
        return 0.0

    def calculate_score(self) -> Dict:
        # ... (초기화 및 Cold Start 로직 유지) ...

        # [Logic Check 1] Safe Zone (안전지대) - 중대 등급 오탐 방지
        # 현재 값이 '평균 + 0.8표준편차' 이내라면 무조건 정상
        # (단, 3건 이상 급증한 경우는 예외)
        safe_threshold = self.mean + (0.8 * self.std)
        if self.current_value <= safe_threshold and self.current_value < 3:
             return {"score": 0, "status": "⚪", "reason": "정상 범위(Safe)"}

        # ... (Track A/B 분기 로직) ...
        
        if self.is_sparse:
             # ... (기존 희소 로직) ...
             # [Tuning] 중대 등급 2건일 때, 평균이 0.5 이상이면(덜 희귀하면) 🟡로 완화
             if self.current_value == 2 and self.is_critical and self.mean >= 0.5:
                 final_status = "🟡"

        else: # Dense
             # ... (기존 Z-Score, Nelson, EWMA 계산) ...
             
             # [New] Velocity Score 추가
             velocity_score = self._calculate_velocity_score()
             
             total_score = base_score + nelson_score + ewma_score + velocity_score
             
             # ... (나머지 합산 로직) ...

```