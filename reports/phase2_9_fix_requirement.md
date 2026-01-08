
---

# 📑 [작업 발주서] UI/UX Unification: Risk Radar 고도화

### 1. 작업 개요

* **목표**: `app.py` 내 **'Risk Radar (실시간 감지)'** 섹션의 UI 디자인을 **'주요 점검필요 LOT'** 섹션과 동일한 **Modern Card Style**로 통일한다.
* **제약 사항**:
* LOT 섹션은 100% HTML로 구현되었으나, Risk Radar는 **'🔍 분석' 버튼(Python Logic Trigger)**이 포함되어야 한다.
* 따라서 **HTML(디자인) + Streamlit Widget(기능)**이 혼합된 **하이브리드 레이아웃**을 적용해야 한다.



### 2. 디자인 규격 (Design Spec)

기존에 정의된 Global CSS 클래스를 재사용하여 시각적 일관성을 확보한다.

| 구분 | 적용 클래스 (CSS) | 내용 |
| --- | --- | --- |
| **카드 컨테이너** | `.card-container` (유사 효과) | 흰색 배경, 그림자, 라운드 처리 |
| **타이틀** | `.card-header`, `.card-title` | 플랜트명 강조, 점수 배치 |
| **메타 정보** | `.card-meta`, `.badge` | 유형, 등급, 건수 뱃지 스타일링 |
| **메시지 박스** | `.card-message` | 회색 배경 박스에 진단/추이 정보 표시 |
| **액션 버튼** | `.btn-download` | 엑셀 다운로드 버튼 스타일 통일 |

### 3. 구현 지침 (Implementation Guide)

개발자는 `app.py`의 **Risk Radar 렌더링 루프**를 아래 **하이브리드 구조**로 전면 재작성하시오.

#### **Step 1. 레이아웃 구조 변경**

* 기존: `with st.container(border=True):` (Native 방식)
* 변경: **`st.columns([8, 2])`** 를 사용하여 좌측(정보)과 우측(액션)을 명확히 분리.

#### **Step 2. 좌측 정보 패널 (Information Column)**

* **방식**: `st.markdown(..., unsafe_allow_html=True)` 사용.
* **내용**: LOT 섹션의 HTML 구조를 차용하되, Risk Radar의 데이터(`점수`, `진단`, `추이`)를 바인딩.
* **Header**: `플랜트명` + `점수` (Color Class 적용: `.text-red` / `.text-yellow`)
* **Meta**: `유형` + `등급 뱃지` + `건수 뱃지`
* **Message**: `진단 내용` + `최근 추이` + `마지막 발생일`



#### **Step 3. 우측 액션 패널 (Action Column)**

* **구성 요소 1 (엑셀)**: `st.markdown`을 이용해 LOT와 동일한 **HTML `<a>` 태그 버튼** 구현. (`class='btn-download'` 적용 필수)
* **구성 요소 2 (분석)**: Streamlit **`st.button`** 위젯 유지. (단, `use_container_width=True` 적용하여 꽉 차게 배치)

---

### 4. 코드 예시 (Reference Code)

개발 에이전트는 아래 코드 패턴을 그대로 차용하여 `render_risk_column` 함수를 재작성하시오.

```python
# [참고] Risk Radar 내부 렌더링 로직 변경안

# ... (데이터 준비 및 base64 인코딩 부분 동일) ...

# --- UI 렌더링 시작 ---
# 카드 간 구분을 위한 컨테이너 (CSS 스타일링을 위해 div로 감싸거나 markdown hr 활용)
st.markdown("""<div style="padding: 12px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 10px;">""", unsafe_allow_html=True)

# 8:2 비율로 정보와 액션 분리
c_info, c_action = st.columns([0.75, 0.25])

# [Left] 정보 패널 (HTML Design)
with c_info:
    badge_color = "badge-red" if color_class == "text-red" else "badge-yellow"
    score_color = "#dc2626" if color_class == "text-red" else "#d97706"
    
    html_content = f"""
    <div style="display: flex; flex-direction: column; gap: 6px;">
        <div class="card-header" style="margin-bottom: 0;">
            <div class="card-title" style="font-size: 1.1rem;">🏭 {row['플랜트']}</div>
            <div style="font-weight: 800; font-size: 1.2rem; color: {score_color};">{int(row['점수'])}점</div>
        </div>
        <div class="card-meta">
            <span class="badge badge-gray">{row['유형']}</span>
            <span class="badge {badge_color}">{row['등급']}</span>
            <span class="badge badge-count">당월 {int(row['건수'])}건</span>
        </div>
        <div class="card-message" style="margin-top: 6px;">
            <div style="margin-bottom: 2px;">💡 <b>{row['진단']}</b></div>
            <div style="font-size: 0.8em; color: #6b7280;">
                📉 추이: {row['Trend_Str']} <span style="margin: 0 4px; color: #d1d5db;">|</span> 📅 {row['Last_Date']}
            </div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

# [Right] 액션 패널 (Hybrid)
with c_action:
    # 1. 엑셀 다운로드 (HTML Style Button)
    st.markdown(f"""
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
       download="Risk_{row['플랜트']}.xlsx" class="btn-download" style="margin-bottom: 8px;">
       📥 엑셀
    </a>
    """, unsafe_allow_html=True)
    
    # 2. 분석 실행 (Native Widget)
    # 버튼 스타일을 CSS로 강제 조정하여 HTML 버튼과 높이/너비 등을 맞춤
    if st.button("🔍 분석", key=f"btn_{color_class}_{idx}", use_container_width=True):
        st.session_state['trigger_analysis'] = True
        st.session_state['target_plant'] = row['플랜트']
        # ... (이동 로직 동일) ...

st.markdown("</div>", unsafe_allow_html=True) # 카드 닫기

```