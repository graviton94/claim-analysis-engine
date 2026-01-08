import re
from pathlib import Path
p = Path(r"c:\claim-analysis-engine\app.py")
s = p.read_text(encoding='utf-8')

# Replace .lot-info CSS rule
s_new = re.sub(r"\.lot-info\s*\{[^}]*\}", ".lot-info { color: #111827; font-size: 0.95rem; line-height: 1.6; }", s, count=1)

# Replace format_trend_with_highlight function
pattern = re.compile(r"def\s+format_trend_with_highlight\(.*?\)\s*:\n(.*?)\n# --- 0\. 페이지 설정 ---", re.S)
new_func = '''def format_trend_with_highlight(trend_str):
    """추이 문자열에서 마지막 숫자를 굵게 강조(검은색)

    Returns the sequence with the final value wrapped in <strong> tags (black).
    """
    if not trend_str or trend_str == '-':
        return "추이 정보 없음"

    # "1 → 2 → 3 → 4 → 5 → 6" 형식 파싱
    parts = trend_str.split(' → ')
    if len(parts) <= 1:
        return f"{trend_str}"

    # 마지막 부분을 제외한 나머지
    normal_parts = parts[:-1]
    last_part = parts[-1]

    # 마지막 숫자를 굵게(검은색)로 표시
    normal_text = ' → '.join(normal_parts)
    highlighted_text = f'<strong style="color: #111827;">{last_part}</strong>'

    if normal_parts:
        return f'{normal_text} → {highlighted_text}'
    else:
        return highlighted_text

# --- 0. 페이지 설정 ---'''

if pattern.search(s_new):
    s_new = pattern.sub(new_func, s_new, count=1)
else:
    print('format_trend_with_highlight pattern not found')

p.write_text(s_new, encoding='utf-8')
print('patched app.py')
