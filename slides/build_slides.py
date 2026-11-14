from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


OUT = Path(__file__).with_name("Day07_Agentic_RAG_Demo_NguyenNgocToan.pptx")
prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)

# Palette: dark technical deck with high-contrast accents.
NAVY = "081D36"
PANEL = "102B4A"
TEAL = "35D0BA"
BLUE = "4DA3FF"
GOLD = "F5B642"
RED = "EE6A6A"
WHITE = "F7FAFC"
MUTED = "AFC1D4"
GREEN = "65D69E"


def rgb(value):
    return RGBColor.from_string(value)


def shape(slide, x, y, w, h, fill, radius=False, line=None):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    item = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    item.fill.solid()
    item.fill.fore_color.rgb = rgb(fill)
    item.line.color.rgb = rgb(line or fill)
    return item


def text(slide, value, x, y, w, h, size=18, color=WHITE, bold=False,
         align=PP_ALIGN.LEFT, font="Aptos", valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def line(slide, x1, y1, x2, y2, color=TEAL, width=2):
    item = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    item.line.color.rgb = rgb(color)
    item.line.width = Pt(width)
    return item


def base(slide, section, number):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(NAVY)
    shape(slide, 0.0, 0.0, 13.333, 0.08, TEAL)
    text(slide, section.upper(), 0.55, 0.28, 5.0, 0.28, 9, TEAL, True)
    text(slide, f"DAY 7 · AGENTIC RAG     {number:02d}", 9.4, 7.08, 3.35, 0.22, 8, MUTED, False, PP_ALIGN.RIGHT)


def title(slide, title_text, subtitle=None):
    text(slide, title_text, 0.55, 0.65, 12.2, 0.7, 28, WHITE, True)
    if subtitle:
        text(slide, subtitle, 0.58, 1.34, 12.0, 0.36, 12, MUTED)


def card(slide, heading, body, x, y, w, h, accent=TEAL, body_size=15):
    shape(slide, x, y, w, h, PANEL, True)
    shape(slide, x, y, 0.06, h, accent)
    text(slide, heading, x + 0.24, y + 0.17, w - 0.42, 0.33, 14, accent, True)
    text(slide, body, x + 0.24, y + 0.58, w - 0.42, h - 0.7, body_size, WHITE, False, valign=MSO_ANCHOR.TOP)


def pill(slide, value, x, y, w, color=TEAL):
    shape(slide, x, y, w, 0.36, color, True)
    text(slide, value, x + 0.05, y + 0.02, w - 0.1, 0.28, 10, NAVY, True, PP_ALIGN.CENTER)


def new_slide(section, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    base(slide, section, number)
    return slide


# 01 — Title
slide = new_slide("Báo cáo Lab", 1)
text(slide, "DAY 7", 0.72, 0.83, 2.2, 0.5, 16, TEAL, True)
text(slide, "Agentic RAG", 0.7, 1.35, 8.5, 0.85, 38, WHITE, True)
text(slide, "Xây dựng agent truy xuất có bằng chứng, citation\nvà biết từ chối khi thiếu thông tin", 0.74, 2.35, 7.5, 0.85, 18, MUTED)
shape(slide, 0.74, 3.52, 1.05, 0.07, GOLD)
text(slide, "Nguyễn Ngọc Toàn", 0.74, 4.03, 4.4, 0.36, 18, WHITE, True)
text(slide, "MSSV: 2A202603405", 0.74, 4.42, 4.4, 0.28, 13, MUTED)
shape(slide, 0.68, 5.06, 3.2, 0.44, TEAL, True)
text(slide, "Phạm vi demo: phần cơ bản", 0.74, 5.1, 3.05, 0.36, 12, NAVY, True, PP_ALIGN.CENTER)

# Right-side visual: question → retrieval → evidence → answer
for label, x, color in [("Câu hỏi", 8.55, BLUE), ("Truy xuất", 9.85, TEAL), ("Bằng chứng", 11.15, GOLD)]:
    shape(slide, x, 2.18, 1.02, 1.05, PANEL, True, color)
    text(slide, label, x + 0.06, 2.54, 0.9, 0.24, 10, color, True, PP_ALIGN.CENTER)
line(slide, 9.57, 2.7, 9.83, 2.7, MUTED, 2)
line(slide, 10.87, 2.7, 11.13, 2.7, MUTED, 2)
shape(slide, 9.0, 4.05, 3.55, 1.2, PANEL, True)
text(slide, "Trả lời có citation\nhoặc abstain đúng lúc", 9.22, 4.33, 3.1, 0.55, 17, WHITE, True, PP_ALIGN.CENTER)
line(slide, 11.66, 3.26, 10.77, 4.02, TEAL, 2)

# 02 — Scope
slide = new_slide("Mục tiêu", 2)
title(slide, "Bài toán & phạm vi thực hiện", "Tận dụng index Day 6 để trả lời câu hỏi về tài liệu Python có kiểm chứng nguồn.")
card(slide, "01 · Kết nối Day 6", "203 chunk từ OpenStax Introduction to Python Programming.\nEmbedding: text-embedding-3-small (1536 chiều).\nVector store: PostgreSQL + pgvector.", 0.62, 2.05, 3.9, 2.25, BLUE)
card(slide, "02 · So sánh retrieval", "Đánh giá dense, BM25 + filter và hybrid RRF + filter trên cùng tập dev.\nKhông coi thay đổi top-k đơn thuần là phương pháp mới.", 4.72, 2.05, 3.9, 2.25, TEAL)
card(slide, "03 · Agent có grounding", "Model thật gọi tool search, trả lời kèm citation hợp lệ; nếu evidence không đủ thì abstain.\nTheo dõi trace, context, thời gian và chi phí.", 8.82, 2.05, 3.9, 2.25, GOLD)
shape(slide, 0.62, 4.9, 12.1, 0.95, "0D3857", True)
text(slide, "Tiêu chí thành công: chọn cấu hình trên dev, sau đó mới đánh giá test một lần.", 0.95, 5.17, 11.4, 0.34, 17, WHITE, True, PP_ALIGN.CENTER)

# 03 — Data & benchmark
slide = new_slide("Dữ liệu", 3)
title(slide, "Dữ liệu và benchmark", "Corpus giới hạn ở các chương Loops, Lists và Dictionaries của OpenStax.")
card(slide, "Corpus", "59 trang nội dung\n203 chunk strategy B\n≤ 500 ký tự / chunk\nID gồm nguồn · trang · vị trí", 0.62, 2.02, 3.22, 2.3, BLUE)
card(slide, "Dev", "18 câu có đáp án\n+ 3 câu thiếu evidence\nDùng để thử nghiệm\nvà chốt cấu hình", 4.0, 2.02, 3.22, 2.3, TEAL)
card(slide, "Test", "6 câu có đáp án\n+ 2 câu thiếu evidence\nChạy sau khi chốt\nđể đo tổng quát hóa", 7.38, 2.02, 3.22, 2.3, GOLD)
card(slide, "Cách chấm", "Evidence phải có trong chunk\nđúng nguồn/trang. Citation\nphải nằm trong context và\nthực sự hỗ trợ câu trả lời.", 10.76, 2.02, 1.95, 2.3, RED, 12)
text(slide, "Pipeline dữ liệu", 0.7, 4.84, 2.0, 0.28, 13, TEAL, True)
for label, x, width, color in [("Trang sách", 0.7, 1.55, BLUE), ("Chunks", 3.0, 1.55, TEAL), ("Embeddings", 5.3, 1.7, GOLD), ("pgvector", 7.75, 1.55, TEAL), ("Evidence", 10.05, 1.55, GREEN)]:
    shape(slide, x, 5.35, width, 0.7, PANEL, True, color)
    text(slide, label, x, 5.55, width, 0.22, 12, WHITE, True, PP_ALIGN.CENTER)
for x in [2.32, 4.62, 7.1, 9.62]:
    line(slide, x, 5.7, x + 0.55, 5.7, MUTED, 2)

# 04 — Architecture
slide = new_slide("Kiến trúc", 4)
title(slide, "Kiến trúc Agentic RAG", "Agent quyết định cách tìm; câu trả lời chỉ dùng context đã thu thập.")
steps = [
    ("1", "Câu hỏi", "Người dùng gửi query", BLUE),
    ("2", "Agent", "gpt-4o-mini\nđiều phối tool", TEAL),
    ("3", "search(query)", "dense / BM25 / hybrid\n+ chapter filter", GOLD),
    ("4", "Context", "dedup · rank\nbudget 1500 ký tự", TEAL),
    ("5", "Kết quả", "answer + citation\nhoặc abstain", GREEN),
]
for idx, (no, heading, body, color) in enumerate(steps):
    x = 0.53 + idx * 2.55
    shape(slide, x, 2.25, 2.08, 2.0, PANEL, True, color)
    shape(slide, x + 0.17, 2.45, 0.4, 0.4, color, True)
    text(slide, no, x + 0.17, 2.51, 0.4, 0.2, 11, NAVY, True, PP_ALIGN.CENTER)
    text(slide, heading, x + 0.16, 3.05, 1.75, 0.27, 14, WHITE, True, PP_ALIGN.CENTER)
    text(slide, body, x + 0.16, 3.46, 1.75, 0.55, 11, MUTED, False, PP_ALIGN.CENTER)
    if idx < 4:
        line(slide, x + 2.1, 3.25, x + 2.52, 3.25, MUTED, 2)
card(slide, "Kiểm soát agent", "Tối đa 3 lần search · bắt buộc search trước khi abstain · dừng khi trả lời/từ chối, hết lượt hoặc lặp query · provider/DB lỗi trả về status: error.", 0.73, 5.15, 11.85, 0.95, RED, 14)

# 05 — Retrieval methods
slide = new_slide("Thử nghiệm", 5)
title(slide, "Ba phương pháp retrieval được so sánh", "Cùng corpus, cùng benchmark dev, top-3 kết quả; khác nhau ở cách truy xuất/xếp hạng.")
card(slide, "Dense · R0", "Query embedding → cosine similarity trong pgvector.\n\nƯu điểm: nhận diện ngữ nghĩa.\nHạn chế: bỏ sót thuật ngữ đặc thù, câu hỏi khác từ vựng sách.", 0.62, 2.0, 3.86, 3.25, BLUE, 14)
card(slide, "Lexical · R2", "BM25 + filter chapter người dùng cung cấp.\n\nƯu điểm: bắt tốt get(), items(), từ khóa chính xác.\nHạn chế: kém linh hoạt khi câu hỏi diễn đạt khác tài liệu.", 4.74, 2.0, 3.86, 3.25, TEAL, 14)
card(slide, "Hybrid · R3/R5", "Dense + BM25, hợp nhất bằng RRF (k=60), 20 candidate mỗi bên; sau đó filter chapter.\n\nMục tiêu: kết hợp tín hiệu ngữ nghĩa và từ khóa.", 8.86, 2.0, 3.86, 3.25, GOLD, 14)
shape(slide, 0.62, 5.77, 12.1, 0.55, "0D3857", True)
text(slide, "R5 = Hybrid + context theo rank + dừng query lặp + budget 1500 ký tự", 0.8, 5.92, 11.7, 0.22, 14, WHITE, True, PP_ALIGN.CENTER)

# 06 — Dev results
slide = new_slide("Kết quả Dev", 6)
title(slide, "Dev: hybrid + filter đem evidence vào context tốt nhất", "Chỉ số chính là Context Evidence trên 18 câu có đáp án; số cao hơn là tốt hơn.")
rows = [
    ("R0", "Dense", 6, "4,72 s", "$0,00515", BLUE),
    ("R2", "BM25 + filter", 12, "3,49 s", "$0,00533", TEAL),
    ("R3", "Hybrid + filter", 15, "3,67 s", "$0,00474", GOLD),
    ("R5", "R3 + rank + stop-repeat", 15, "3,35 s", "$0,00419", GREEN),
]
text(slide, "Cấu hình", 0.78, 1.9, 2.7, 0.25, 11, MUTED, True)
text(slide, "Context Evidence", 3.65, 1.9, 5.0, 0.25, 11, MUTED, True)
text(slide, "Trung vị", 9.25, 1.9, 1.1, 0.25, 11, MUTED, True, PP_ALIGN.CENTER)
text(slide, "Chi phí / 21 câu", 10.6, 1.9, 1.5, 0.25, 11, MUTED, True, PP_ALIGN.CENTER)
for i, (run, method, score, seconds, cost, color) in enumerate(rows):
    y = 2.28 + i * 0.84
    shape(slide, 0.62, y, 12.1, 0.64, PANEL, True)
    text(slide, run, 0.84, y + 0.15, 0.45, 0.24, 13, color, True)
    text(slide, method, 1.35, y + 0.15, 2.15, 0.24, 13, WHITE, i == 3)
    shape(slide, 3.7, y + 0.18, 4.7, 0.27, "183B59", True)
    shape(slide, 3.7, y + 0.18, 4.7 * score / 18, 0.27, color, True)
    text(slide, f"{score}/18", 8.55, y + 0.14, 0.55, 0.26, 13, WHITE, True)
    text(slide, seconds, 9.2, y + 0.15, 1.2, 0.24, 12, WHITE, False, PP_ALIGN.CENTER)
    text(slide, cost, 10.55, y + 0.15, 1.65, 0.24, 12, WHITE, False, PP_ALIGN.CENTER)
shape(slide, 0.62, 5.9, 12.1, 0.62, "123C4B", True)
text(slide, "Lựa chọn R5: cùng 15/18 evidence như R3/R4 nhưng nhanh nhất, rẻ nhất và context ngắn hơn 23%.", 0.85, 6.08, 11.65, 0.23, 13, WHITE, True, PP_ALIGN.CENTER)

# 07 — Agent & controls
slide = new_slide("Agent", 7)
title(slide, "Quy tắc grounding và xử lý lỗi", "Thiết kế nhằm tránh trả lời khi evidence chưa đủ và giảm vòng lặp tìm kiếm.")
card(slide, "Ba tool", "search(query)\nanswer(answer, citations)\nabstain(reason)", 0.62, 2.0, 2.9, 2.42, BLUE, 16)
card(slide, "Context an toàn", "Gộp các search, bỏ trùng, xếp theo rank; mỗi đoạn kèm ID/nguồn/trang. Citation bắt buộc có trong context.", 3.77, 2.0, 2.9, 2.42, TEAL, 15)
card(slide, "Điều kiện dừng", "Tối đa 3 search. Bắt buộc search ở bước đầu. Dừng khi model answer/abstain, hết lượt, hoặc query bị lặp.", 6.92, 2.0, 2.9, 2.42, GOLD, 15)
card(slide, "Lỗi & trace", "Lưu tool call, query, context, kết quả và citation. Lỗi provider/DB không tạo câu trả lời: status = error.", 10.07, 2.0, 2.65, 2.42, RED, 14)
shape(slide, 0.62, 5.05, 12.1, 0.9, PANEL, True)
text(slide, "Bài học từ dev", 0.9, 5.25, 2.1, 0.23, 13, TEAL, True)
text(slide, "Dense thiếu evidence chủ yếu vì không tìm thấy chunk đúng, không phải vì context bị cắt. Hybrid + filter tăng Context Evidence từ 6/18 lên 15/18.", 2.85, 5.18, 9.45, 0.35, 14, WHITE)

# 08 — Demo cases
slide = new_slide("Demo", 8)
title(slide, "Hai tình huống demo", "Một câu có evidence và một câu phải từ chối vì corpus không có thông tin cần thiết.")
card(slide, "Case 1 · q22 · Có evidence", "Query: iterating with items() loop variable\n\nTool: search(...) → p248:p1 (rank 1)\n\nAnswer: “Vòng lặp nhận cả key và value cho mỗi item.” [p248:p1]\n\nKết quả: grounded 2/2, citation hỗ trợ đúng.", 0.62, 2.0, 5.85, 3.72, GREEN, 15)
card(slide, "Case 2 · u04 · Thiếu evidence", "Query: dictionary method setdefault() return when key is missing\n\nTool: search(...) → chỉ tìm được chunk về get(); lần 2 lặp query nên dừng.\n\nAnswer: “Không đủ bằng chứng để trả lời.”\n\nKết quả: abstention đúng, không bịa thông tin.", 6.87, 2.0, 5.85, 3.72, RED, 15)
pill(slide, "Có nguồn · có citation", 1.0, 6.1, 2.25, GREEN)
pill(slide, "Không có nguồn · abstain", 8.05, 6.1, 2.55, RED)

# 09 — Test & conclusion
slide = new_slide("Kết luận", 9)
title(slide, "Kết quả test & kết luận", "R5 được chốt trên dev trước khi chạy test; baseline R0 chạy để đối chiếu.")
card(slide, "R5 trên test", "Evidence trong context\n6/6 câu có đáp án\n\nCitation hợp lệ\n8/8\n\nAbstention đúng\n2/2", 0.62, 2.0, 3.15, 3.15, GREEN, 16)
card(slide, "So với baseline R0", "Context Evidence\n4/6 → 6/6\n\nGrounded answer TB\n1,50/2 → 2,00/2\n\nTừ chối nhầm\n1/6 → 0/6", 4.0, 2.0, 3.15, 3.15, BLUE, 16)
card(slide, "Thông điệp chính", "Hybrid RRF + filter không chỉ tăng khả năng tìm evidence mà còn giúp agent trả lời có nguồn, ít từ chối nhầm hơn và tiết kiệm chi phí nhờ chặn search lặp.", 7.38, 2.0, 5.34, 3.15, GOLD, 17)
shape(slide, 0.62, 5.72, 12.1, 0.64, "0D3857", True)
text(slide, "Tái lập: cd starter  →  python agent.py --split test --method hybrid_rrf --filter --budget 1500 --context-order rank --stop-repeat", 0.82, 5.92, 11.7, 0.22, 10, WHITE, False, PP_ALIGN.CENTER, "Consolas")

prs.core_properties.title = "Day 7 Agentic RAG — Demo"
prs.core_properties.subject = "LAB Day 7 — phần cơ bản"
prs.core_properties.author = "Nguyễn Ngọc Toàn"
OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)
