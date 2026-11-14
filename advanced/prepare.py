"""Advanced challenge: OpenStax Introductory Statistics 2e, Chapters 1-5 (PDF pages 17-346).

Run from repo root:
    python advanced/prepare.py download                 # fetch PDF, check SHA-256
    python advanced/prepare.py extract                  # data/pages.jsonl + data/manifest.json
    python advanced/prepare.py vision MODEL [PAGES]     # describe figure pages; appends to data/visual.jsonl
    python advanced/prepare.py chunks                   # data/chunks.json (page text + latest visual description)
PAGES is a comma list (e.g. 118,124) or a file with one; default = every page with a figure not yet described by MODEL.
"""
import base64
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF 1.26.5
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PDF = DATA / 'raw' / 'introductory-statistics-2e.pdf'
URL = 'https://assets.openstax.org/oscms-prodcms/media/documents/introductory-statistics-2e_-_WEB.pdf'
SHA256 = 'a79a953b75e570030c74ea6f700b8bd70f41dc8c01bf4e9618e6a0adba3a382a'
DOC_ID = 'openstax-intro-stats-2e'
FIRST_CHAPTER, LAST_CHAPTER = 1, 5
PRICES = {'gpt-4.1-mini': (0.40, 1.60), 'gpt-4.1': (2.0, 8.0)}  # USD per 1M input/output tokens (OpenAI list price)
load_dotenv(ROOT.parent / 'starter' / '.env')

VISION_PROMPT = """This is PDF page {page} of the textbook "Introductory Statistics 2e" (OpenStax).
Describe every figure, chart, graph, diagram, calculator screen or photo in the attached crops so that a reader who cannot see the page can answer questions about it.
For each visual give: its label (e.g. "Figure 2.5") if printed, kind (bar graph, histogram, dot plot, box plot, scatter plot, line graph, pie chart, tree diagram, Venn diagram, density curve, calculator screen, table image, photo, other),
title, axis titles with units and tick labels, legend, and every number or label printed in the visual.
Write "unreadable" rather than inventing text you cannot see. Mark purely decorative photos as decorative and do not describe them further.
Mark every value you estimate (not printed) with "≈".
Rules for charts:
- Dot plots: for every axis value list the exact number of X marks or dots above it ("3: 5 X"); then give the total.
- Scatter plots: list every point left to right as (≈x, ≈y), then give the total number of points.
- Histograms / bar graphs: list every bar left to right with its interval or label and its height; if the y-axis has no numbers, give the height in gridlines. Then rank the bars from tallest to shortest; say "equal" only if the tops are at the same gridline.
- Box plots: minimum, Q1 (left edge of the box), median (line inside), Q3 (right edge), maximum. Ticks may be unevenly spaced: interpolate between the two nearest tick labels. No left whisker means Q1 = minimum; no right whisker means Q3 = maximum.
- Line graphs: list every point in order and say whether each step goes up, down or stays level.
- Shaded-area graphs (uniform, exponential, normal ...): the full x-range of the curve or rectangle, its height if labeled, the exact shaded interval and every label.
The page text below only helps match labels and captions; do not summarize it.
Answer as compact markdown, one "### <label or 'Unlabeled visual N'>" section per visual.

Page text:
{text}"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope(doc):
    """First and last PDF page of chapters FIRST_CHAPTER..LAST_CHAPTER, from the PDF outline."""
    starts = {int(m.group(1)): p for lvl, t, p in doc.get_toc() if lvl == 1 and (m := re.match(r'Chapter (\d+)', t))}
    return starts[FIRST_CHAPTER], starts[LAST_CHAPTER + 1] - 1


def merge(rects):
    """Merge overlapping/nearby boxes so one figure made of several images becomes one crop."""
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if (rects[i] + (-10, -10, 10, 10)).intersects(rects[j]):
                    rects[i] |= rects.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(rects, key=lambda r: (r.y0, r.x0))


def extract():
    doc = fitz.open(PDF)
    first, last = scope(doc)
    toc = [(lvl, t, p) for lvl, t, p in doc.get_toc() if lvl <= 2]
    pages, skipped = [], []
    for n in range(first, last + 1):
        page = doc[n - 1]
        text = page.get_text()
        area = abs(page.rect)
        # Figures in this book are raster images; tiny ones are icons (Try It arrows, calculator logo).
        boxes = merge([i['bbox'] for i in page.get_image_info() if abs(fitz.Rect(i['bbox'])) >= 0.01 * area])
        if len(text.strip()) < 200 and not boxes:
            skipped.append({'page': n, 'reason': 'under 200 characters and no figure'})
            continue
        chapter = [t for lvl, t, p in toc if lvl == 1 and p <= n][-1]
        section = ([t for lvl, t, p in toc if lvl == 2 and p <= n] or [None])[-1]
        pages.append({'page': n, 'printed_page': page.get_label() or None, 'chapter': chapter, 'section': section,
                      'text': text, 'figure_labels': sorted(set(re.findall(r'Figure \d+\.\d+', text))),
                      'table_labels': sorted(set(re.findall(r'Table \d+\.\d+', text))),
                      'table_boxes': [list(t.bbox) for t in page.find_tables().tables],
                      'visual_boxes': [list(b) for b in boxes]})
    with open(DATA / 'pages.jsonl', 'w', encoding='utf-8') as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    manifest = {'document': 'Introductory Statistics 2e (OpenStax, 2023), CC BY-NC-SA 4.0', 'source': URL, 'sha256': SHA256,
                'scope': f'Chapters {FIRST_CHAPTER}-{LAST_CHAPTER}, PDF pages {first}-{last}',
                'pages_in_scope': last - first + 1, 'pages_indexed': len(pages),
                'pages_with_figures': sum(bool(p['visual_boxes']) for p in pages),
                'figures': sum(len(p['visual_boxes']) for p in pages), 'skipped_pages': skipped,
                'tools': {'pymupdf': fitz.VersionBind}}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: v for k, v in manifest.items() if k != 'skipped_pages'}, 'skipped', len(skipped))


def load_visual():
    """Latest description per page (later lines win, so a re-run never deletes the earlier one)."""
    path = DATA / 'visual.jsonl'
    return {v['page']: v for v in map(json.loads, open(path, encoding='utf-8'))} if path.exists() else {}


def vision(model, only=None):
    pages = [json.loads(l) for l in open(DATA / 'pages.jsonl', encoding='utf-8')]
    done = {p for p, v in load_visual().items() if v['model'] == model}
    todo = [p for p in pages if p['visual_boxes'] and (p['page'] in only if only else p['page'] not in done)]
    client = OpenAI(timeout=120, max_retries=2)
    doc = fitz.open(PDF)
    (DATA / 'pages').mkdir(exist_ok=True)
    tin = tout = 0
    print(f'vision {model}: {len(todo)} pages', flush=True)
    with open(DATA / 'visual.jsonl', 'a', encoding='utf-8') as f:
        for k, p in enumerate(todo, 1):
            page = doc[p['page'] - 1]
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(DATA / 'pages' / f"p{p['page']:03d}.png")  # for checking
            content = [{'type': 'text', 'text': VISION_PROMPT.format(page=p['page'], text=' '.join(p['text'].split())[:2500])}]
            for box in p['visual_boxes'][:4]:
                clip = (fitz.Rect(box) + (-25, -25, 25, 25)) & page.rect  # margin keeps axis titles and caption
                zoom = min(3.0, 1200 / max(clip.width, clip.height))
                b64 = base64.b64encode(page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).tobytes('png')).decode()
                content.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'}})
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(model=model, temperature=0, messages=[{'role': 'user', 'content': content}])
                    break
                except Exception as e:  # transient API error: retry, then fail loudly
                    if attempt == 2:
                        raise
                    print('retry', p['page'], e, flush=True)
                    time.sleep(5)
            tin += r.usage.prompt_tokens
            tout += r.usage.completion_tokens
            f.write(json.dumps({'page': p['page'], 'model': model, 'description': r.choices[0].message.content,
                                'prompt_tokens': r.usage.prompt_tokens, 'completion_tokens': r.usage.completion_tokens},
                               ensure_ascii=False) + '\n')
            f.flush()
            pi, po = PRICES[model]
            print(f"{k}/{len(todo)} p{p['page']} cost so far ${(tin * pi + tout * po) / 1e6:.4f}", flush=True)
    pi, po = PRICES[model]
    print(f'vision {model} tokens in={tin} out={tout} cost=${(tin * pi + tout * po) / 1e6:.4f}', flush=True)


def chunks():
    pages = [json.loads(l) for l in open(DATA / 'pages.jsonl', encoding='utf-8')]
    vis = load_visual()
    out = []
    for p in pages:
        meta = {'chapter': p['chapter'], 'section': p['section'], 'printed_page': p['printed_page'],
                'figure_labels': p['figure_labels'], 'table_labels': p['table_labels'], 'table_boxes': p['table_boxes']}
        out.append({'id': f"{DOC_ID}:p{p['page']}:text", 'source_id': DOC_ID, 'source': URL, 'page': p['page'],
                    'text': p['text'], 'metadata': {**meta, 'modality': 'text'}})
        if p['page'] in vis:
            v = vis[p['page']]
            out.append({'id': f"{DOC_ID}:p{p['page']}:visual", 'source_id': DOC_ID, 'source': URL, 'page': p['page'],
                        'text': f"[Visual description of page {p['page']} generated by {v['model']}; values marked ≈ are "
                                f"estimates; verify against the page image]\n{v['description']}",
                        'metadata': {**meta, 'modality': 'visual', 'visual_boxes': p['visual_boxes'],
                                     'page_image': f"advanced/data/pages/p{p['page']:03d}.png"}})
    (DATA / 'chunks.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f"chunks: {len(out)} ({sum(c['metadata']['modality'] == 'visual' for c in out)} visual)")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'download':
        PDF.parent.mkdir(parents=True, exist_ok=True)
        if not PDF.exists():
            urllib.request.urlretrieve(URL, PDF)
        assert sha(PDF) == SHA256, 'PDF changed: snapshot differs from the one used for the benchmark'
        print('PDF OK', PDF)
    elif cmd == 'extract':
        extract()
    elif cmd == 'vision':
        arg = sys.argv[3] if len(sys.argv) > 3 else ''
        arg = Path(arg).read_text().strip() if arg and Path(arg).exists() else arg
        vision(sys.argv[2], {int(x) for x in arg.split(',')} if arg else None)
    elif cmd == 'chunks':
        chunks()
