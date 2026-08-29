"""마크다운 분량·가독성 감사 — 정규식 체크포인트로는 셀 단위 문장을 셀 수 없어 별도 스크립트로 둔다.

사용:
    python .claude/scripts/markdown_budget.py "Practice05_ML_General_Tips.ipynb"
    python .claude/scripts/markdown_budget.py            # 과목 전체
    python .claude/scripts/markdown_budget.py --stats    # 기준을 다시 잴 때

검사 항목:
    words     셀 전체 단어 수 (표 마크업 포함)
    prose     표·헤더·수식을 뺀 산문 단어 수 — 줄글이 늘어난 것은 이 숫자가 잡는다
    run-on    한 줄에 두 문장 이상 (문장마다 줄을 바꾼다)
    para      한 문단이 세 문장을 넘음

기준의 근거와 이유는 CLAUDE.md "설명 분량" 을 볼 것.
"""
import glob
import json
import os
import re
import sys
import statistics

WORDS_HARD = 120        # 셀 전체 단어 수, 어떤 마크다운 셀도 넘지 않는다
WORDS_SOFT = 80         # 넘으면 표로 바꾸거나 쪼갤 것
TITLE_CAP = 150         # 노트북 첫 셀(제목 + 개요)만 예외
SUMMARY_CAP = 100       # Summary 는 본문 요약이므로 더 짧아야 한다

PROSE_HARD = 80         # 표를 뺀 산문만 센 단어 수
PROSE_SOFT = 60         # 넘으면 불릿·표로 바꾸거나 버릴 것
PARA_SENTENCES = 3      # 한 문단(빈 줄로 구분되는 렌더링 단위)의 문장 수

FENCE = re.compile(r'```.*?```', re.S)
SKIP_LINE = ('|', '#', '---', '$$', '<!--')
ABBREVIATIONS = ('e.g.', 'i.e.', 'vs.', 'etc.', 'cf.', 'approx.', 'Fig.', 'Eq.', 'No.')
# 글자·숫자·닫는 괄호·수식·백틱으로 끝난 뒤 공백을 두고 대문자로 시작하면 새 문장으로 본다.
# 대문자를 포함하는 이유는 "than plain SGD. Its argument" 처럼 약어로 끝나는 문장이 흔하기 때문이다.
SENTENCE_BREAK = re.compile(r'(?<=[A-Za-z0-9\)\]$`])[.!?]["\')\]`]?\s+(?=[A-Z$`*\[(])')


def strip_verbatim(source):
    # 코드펜스와 여러 줄 $$ 수식은 산문이 아니다. 줄 수를 유지하려고 빈 줄로 바꾼다
    out, in_fence, in_math = [], False, False
    for line in source.splitlines():
        # 인용 안의 코드펜스(`> ```python`)도 코드다. 마커를 떼고 판정한다
        stripped = re.sub(r'^\s*>\s?', '', line).strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
            out.append('')
            continue
        if not in_fence and stripped.count('$$') % 2 == 1:
            in_math = not in_math
            out.append('')
            continue
        out.append('' if (in_fence or in_math) else line)
    return '\n'.join(out)


def prose_lines(source):
    # 렌더링되는 산문 줄만 남긴다 (표·헤더·수식·코드펜스 제외, 불릿과 인용은 포함)
    return [line for line in strip_verbatim(source).splitlines()
            if line.strip() and not line.strip().startswith(SKIP_LINE)]


def paragraphs(source):
    # 빈 줄로 구분되는 렌더링 단위. 불릿·인용·번호 목록은 항목마다 줄이 서므로 문단 검사에서 뺀다
    source = strip_verbatim(source)
    for block in re.split(r'\n\s*\n', source):
        lines = [l for l in block.splitlines()
                 if l.strip() and not l.strip().startswith(SKIP_LINE)]
        if lines and not re.match(r'\s*([-*>]|\d+\.)\s', lines[0]):
            yield lines


LIST_MARKER = re.compile(r'^\s*([-*>]\s+|\d+\.\s+)+')
# 연습문제·과제 문항은 지시문이다. 줄이면 과제 자체가 바뀌므로 산문 상한에서 뺀다
EXERCISE = re.compile(r'^#{1,4}\s*(Exercise|Review Exercise|Problem)\b', re.M | re.I)


def count_sentences(text):
    text = LIST_MARKER.sub('', text)          # "1. Store ..." 의 목록 번호는 문장 끝이 아니다
    for abbreviation in ABBREVIATIONS:
        text = text.replace(abbreviation, abbreviation.replace('.', '\x00'))
    return len(SENTENCE_BREAK.findall(text)) + 1


def cap_for(markdown_index, source):
    if markdown_index == 0:
        return TITLE_CAP, 'title'
    if re.search(r'^#{1,3}\s*Summary', source.strip(), re.M | re.I):
        return SUMMARY_CAP, 'summary'
    return WORDS_HARD, 'body'


def first_heading(source):
    for line in source.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('---'):
            return stripped[:58]
    return ''


def audit(path):
    notebook = json.load(open(path, encoding='utf-8'))
    findings = []
    markdown_index = 0
    for cell_number, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'markdown':
            continue
        source = ''.join(cell.get('source', ''))
        cap, kind = cap_for(markdown_index, source)
        markdown_index += 1
        heading = first_heading(source)

        words = len(source.split())
        if words > cap:
            findings.append(('OVER', cell_number, f'{words} words (cap {cap}, {kind})', heading))
        elif words > WORDS_SOFT and kind == 'body':
            findings.append(('soft', cell_number, f'{words} words (soft {WORDS_SOFT})', heading))

        prose = sum(len(line.split()) for line in prose_lines(source))
        if EXERCISE.search(source):
            prose = 0        # 문제 지시문은 설명이 아니다. 전체 단어 수 상한만 적용한다
        if prose > PROSE_HARD:
            findings.append(('OVER', cell_number, f'{prose} prose words (cap {PROSE_HARD})', heading))
        elif prose > PROSE_SOFT:
            findings.append(('soft', cell_number, f'{prose} prose words (soft {PROSE_SOFT})', heading))

        for line in prose_lines(source):
            if count_sentences(line) > 1:
                findings.append(('OVER', cell_number, 'run-on line', line.strip()[:58]))

        for block in paragraphs(source):
            sentences = count_sentences(' '.join(block))
            if sentences > PARA_SENTENCES:
                findings.append(('OVER', cell_number,
                                 f'{sentences} sentences in one paragraph', block[0].strip()[:58]))
    return findings


def stats(paths):
    # 기준을 다시 잴 때 쓴다. 캡을 옮기려면 이 숫자를 CLAUDE.md 에 함께 적을 것
    per_cell, per_paragraph, run_on, lines = [], [], 0, 0
    for path in paths:
        notebook = json.load(open(path, encoding='utf-8'))
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') != 'markdown':
                continue
            source = ''.join(cell.get('source', ''))
            prose = sum(len(line.split()) for line in prose_lines(source))
            if prose:
                per_cell.append(prose)
            for line in prose_lines(source):
                lines += 1
                run_on += count_sentences(line) > 1
            for block in paragraphs(source):
                per_paragraph.append(count_sentences(' '.join(block)))

    def percentile(values, point):
        values = sorted(values)
        return values[min(len(values) - 1, int(len(values) * point / 100))]

    print(f'markdown cells with prose : {len(per_cell)}')
    print(f'prose words per cell      : median {statistics.median(per_cell):.0f}, '
          f'p70 {percentile(per_cell, 70)}, p90 {percentile(per_cell, 90)}, max {max(per_cell)}')
    print(f'sentences per paragraph   : median {statistics.median(per_paragraph):.0f}, '
          f'p90 {percentile(per_paragraph, 90)}, max {max(per_paragraph)}')
    print(f'run-on lines              : {run_on} of {lines} prose lines ({run_on / lines:.0%})')


def main(paths):
    total_over = 0
    for path in paths:
        findings = audit(path)
        over = [f for f in findings if f[0] == 'OVER']
        total_over += len(over)
        name = os.path.basename(path)
        if not findings:
            print(f'{name}: OK')
            continue
        print(f'{name}: {len(over)} over cap, {len(findings) - len(over)} above soft cap')
        for level, cell_number, what, context in findings:
            mark = '  [!]' if level == 'OVER' else '   - '
            print(f'{mark} cell {cell_number:3d}  {what:34s} {context}')
        print()
    return 1 if total_over else 0


if __name__ == '__main__':
    arguments = sys.argv[1:]
    targets = [a for a in arguments if not a.startswith('--')] or \
        sorted(glob.glob('Practice*.ipynb')) + sorted(glob.glob('HW/**/*.ipynb', recursive=True))
    if '--stats' in arguments:
        stats(targets)
        sys.exit(0)
    sys.exit(main(targets))
