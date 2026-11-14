# Báo cáo Day 7 — Nguyễn Ngọc Toàn (2A202603405)

> Nhãn chấm tay (grounded answer, citation, abstention, context 0/1/2) là **bản nháp do Claude (AI) chấm**, học viên duyệt lại; cột `grader` trong mỗi `review.csv` ghi rõ. Chi tiết thí nghiệm: [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).

## Dữ liệu và benchmark

- **Nguồn, quyền, số trang, số chunk:** OpenStax *Introduction to Python Programming*, chương 5 Loops, 9 Lists, 10 Dictionaries; CC BY-NC-SA 4.0; 59 trang nội dung (manifest và cách tải lại ở repo Day 6). Dùng lại 203 chunk strategy B của Day 6 (theo đoạn, ≤500 ký tự), ID `openstax-intro-python:p<trang>:p<n>`.
- **Vector store, embedding:** PostgreSQL 16 + pgvector 0.8.7 (Docker Day 6, `127.0.0.1:55433`), bảng `chunks`, exact cosine. OpenAI `text-embedding-3-small`, 1536 chiều; `starter/day6_store.py` báo lỗi nếu model hoặc số chiều của query khác index.
- **Agent model, cách chạy, split:** `gpt-4o-mini`, temperature 0, tool calling. Lệnh chạy ở `starter/README.md`. Benchmark: dev 18 có đáp án + 3 thiếu evidence; test 6 + 2 (24 câu có đáp án lấy từ Day 6, đã chia trước khi chạy; 5 câu thiếu evidence mới ở `data/unanswerable.json`).
- **Có/thiếu evidence, tiêu chí chấm:** câu có đáp án có gold `{source_id, page, evidence}` (gồm cả evidence thay thế); hit = evidence nằm trọn trong chunk đúng nguồn/trang, không phân biệt khoảng trắng. Câu thiếu evidence gần chủ đề (do-while, độ phức tạp sort(), defaultdict, setdefault(), enumerate()); kiểm tra bằng grep toàn bộ `pages.jsonl` (0 kết quả) và đọc các trang liên quan. Grounded answer 0/1/2, citation phải nằm trong context **và** hỗ trợ khẳng định.

## Retrieval và thử nghiệm

Retrieval thuần (top-3, budget 1500 ký tự, `starter/results/retrieval-{dev,test}/`) và agent (`starter/results/exp-{dev,test}/`). Thời gian là trung vị mỗi câu; chi phí agent là tổng của split (LLM + embedding).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (R0) | dense pgvector, top-3, không filter, budget 1500, thứ tự đến | baseline | dev | 6/18 (33%) | 10/18 (56%) | agent 6/18 (33%) | 4,72 s / $0,0052 |
| Phương pháp 2 (R2) | BM25 + filter chapter, top-3, budget 3000 | lexical bắt từ khóa đặc thù (`get()`, `items()`) | dev | 8/18 (44%) | 12/18 (67%) | agent 12/18 (67%) | 3,49 s / $0,0053 |
| Phương pháp 3 (R3) | hybrid RRF (dense + BM25, k=60, 20 candidate mỗi bên) + filter, budget 3000 | dense và BM25 bắt các câu khác nhau | dev | 7/18 (39%) | 16/18 (89%) | agent 15/18 (83%) | 3,67 s / $0,0047 |
| Cải thiện 1 (R1) | R0 + context xếp theo rank qua các lần search | chunk tốt ở search sau bị cắt khi hết budget | dev | 6/18 | 10/18 | agent 7/18 (39%) | 4,48 s / $0,0052 |
| Cải thiện 2 (R4/R5) | R3 + xếp theo rank + dừng khi lặp query; R5 budget 1500 | bỏ search lặp; budget nhỏ không mất evidence | dev | 7/18 | 16/18 | agent 15/18 (83%) | R5: 3,35 s / $0,0042 |
| Cấu hình đã chọn (R5) | như trên | chốt trên dev | test | 5/6 (83%) | 6/6 (100%) | agent 6/6 (100%) | 4,12 s / $0,0017 |
| Baseline (R0) | như trên | | test | 3/6 (50%) | 4/6 (67%) | agent 4/6 (67%) | 4,86 s / $0,0020 |

Hit@1/Hit@3 là retrieval một lượt với query gốc (bảng đầy đủ 7 cấu hình, gồm TF-IDF, ở `docs/EXPERIMENTS.md`). Context Evidence Rate là context cuối của agent (query do model viết lại, có thể nhiều lượt). R2 là phương pháp mới, R3 vừa là phương pháp mới vừa là cải thiện; R1, R4, R5 là cấu hình cải thiện.

**Hai lỗi dev:**
1. *Từ chối nhầm 15/18 ở lần chạy agent đầu.* Nguyên nhân: model gọi `abstain` trước khi search (q06, q09, q12, q15) và ghi citation mà thiếu `[id]` trong câu trả lời nên khung kiểm tra loại (q10, q13, q14). Sửa: bắt buộc search ở bước đầu; nối marker còn thiếu và ghi log. Sau: 0/18 từ chối nhầm, abstention 3/3 giữ nguyên.
2. *Dense agent thiếu evidence (R0 6/18).* Giả thuyết R1 (thứ tự context) **gần như sai**: chỉ 2/12 câu có gold trong kết quả search nhưng bị cắt; 10/12 dense không hề tìm thấy. R1 chỉ 6→7 (trong mức nhiễu). Đổi sang hybrid + filter: 15/18.

**Lý do chọn R5** (tiêu chí công bố trước, ở đầu `docs/EXPERIMENTS.md`): R3/R4/R5 hòa về Context Evidence (15/18), từ chối nhầm (0/18), abstention (3/3) và câu trả lời; R5 nhanh và rẻ nhất, context nhỏ hơn 23%.

## Agent và đánh giá

- **Tool, lượt gọi, dừng, context, lỗi:** 3 tool `search(query)`, `answer(answer, citations)`, `abstain(reason)`; tối đa 3 search, top-3 mỗi lần; bước đầu bắt buộc search; dừng khi model trả lời/từ chối, hết lượt hoặc (R5) lặp query. Context: gộp các lần search, bỏ trùng, xếp theo rank, cắt theo 1500 ký tự, mỗi đoạn kèm ID/nguồn/trang; nội dung tài liệu được khai báo là dữ liệu, không phải chỉ dẫn. Filter chapter do người dùng cung cấp được hệ thống áp dụng, model không tự đặt. Citation phải nằm trong context, không thì từ chối. Lỗi provider/DB ghi `status: error` (không có lần nào).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| R5 / dev | 21/21 | 1: 19, 2: 1, 3: 1 | 0 / 3 / 15 — 1,83 | 21/22 | 18/18 | 3/3 | 0/18 |
| R0 / test | 8/8 | 1: 5, 3: 3 | 1 / 1 / 4 — 1,50 | 6/8 | 5/5 | 2/2 | 1/6 |
| R5 / test | 8/8 | 1: 7, 3: 1 | 0 / 0 / 6 — 2,00 | 8/8 | 6/6 | 2/2 | 0/6 |

Context 0/1/2 trên 6 câu test có đáp án: **R0** 2: 4 (q19, q20, q21, q24), 1: 1 (q23, chỉ có nested list, thiếu dictionary), 0: 1 (q22); **R5** 2: 6.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q22 (R5, test) | có evidence | `search("iterating with items() loop variable")` → p248:p1 hạng 1 | "the loop variable receives both the key and value for each item" [p248:p1] | 2 — đúng nguyên văn p248:p1. R0 dense từ chối nhầm sau 3 search |
| u04 (R5, test) | thiếu evidence | `search("dictionary method setdefault() return when key is missing")` → các chunk về get(); lần 2 lặp query → dừng | "Không đủ bằng chứng để trả lời." | đúng — không có setdefault trong corpus, không bịa |
| q23 (R0, test) | có evidence, lỗi | `search("nested data type")` không filter → nested list p229 + tiêu đề | trả lời chung list + dictionary [p221:p4][p229:p3][p251:p3] | 1 — câu về nested dictionary không có trong context; 2/3 citation không hỗ trợ. R5 có filter ch10 → 2 |
| q15 (R5, dev) | có evidence, lỗi | context chỉ có trắc nghiệm `{}` (p239:p3) | "{} hoặc dict()" [p239:p3] | 1 — `dict()` là kiến thức ngoài; kiểm tra ID không phát hiện |

- **Trace, nhãn, lệnh, thời gian, chi phí:** `starter/results/exp-{dev,test}/<run>/{traces,summary,reviews,agent_metrics_draft}.json` và `review.csv`. Toàn bộ thí nghiệm dev + test tốn khoảng $0,04.
- **Trước/sau, đánh đổi, giới hạn:** Hybrid + filter tăng Context Evidence dev 6→15/18 và test 4→6/6, đồng thời nhanh và rẻ hơn dense vì ít phải search lại. Lỗi còn lại là model thêm chi tiết ngoài context (q07, q15, q11 trên dev); cần đọc tay, metric tự động không bắt được. Giới hạn: test chỉ 8 câu nên 100% không có nghĩa thống kê; một cuốn sách, ba chương; filter chapter là input giả định; model không hoàn toàn tất định (±1 câu giữa các lần chạy); câu Day 6 test từng được chạy một lần ở Day 6 nhưng không dùng để chỉnh Day 7; nhãn do AI chấm nháp có nguy cơ thiên lệch.

## Thử thách nâng cao (nếu thực hiện)

Chi tiết, cách chọn benchmark, các cấu hình và lệnh tái lập: [advanced/README.md](../../advanced/README.md).

- **Tài liệu, nguồn, quyền, số trang, phiên bản:** OpenStax *Introductory Statistics 2e* (2023), CC BY-NC-SA 4.0. Dùng Chương 1–5, tức PDF trang 17–346 (330 trang); SHA-256 `a79a953b…382a`, tải từ assets.openstax.org. Index 328 trang (bỏ 2 trang trống).
- **Cách xử lý:**
  - Text từng trang bằng PyMuPDF 1.26.5 (bảng giữ trong text, bbox bảng ở metadata).
  - 113 trang có hình raster: crop từng hình. `gpt-4.1-mini` mô tả, sau đó `gpt-4.1` đọc lại 69 trang biểu đồ. Mô tả lưu thành chunk `:visual` kèm bbox và ảnh trang.
  - Vòng 2 (cấu hình chính thức) dùng **Gemini API free tier**:
    - embedding `gemini-embedding-001` (1536 chiều) trên pgvector Day 6, strategy `ADVG`;
    - hybrid RRF với BM25;
    - **tra cứu theo nhãn** Figure/Table nêu trong câu hỏi;
    - mỗi trang mang cả chunk text lẫn chunk visual;
    - lúc trả lời, **gửi kèm ảnh crop hình gốc** cho `gemini-3.5-flash-lite`.
  - Chưa xử lý: công thức dạng ảnh; mô tả vision có thể sai, nên kiểm chứng bằng ảnh gốc.
- **Benchmark, model, cấu hình:**
  - 78 câu: dev 15+3; dev2 24+6 (là test E1 cũ, đã xem nên chuyển sang dev); test 5 gồm 24+6 câu mới, khóa trước vòng 2 (SHA-256 trong `advanced/benchmark/questions.sha256`).
  - Hình, bảng và trang text được chọn bằng lấy mẫu hệ thống. Evidence text kiểm tra nguyên văn bằng script; evidence hình đối chiếu trên ảnh trang gốc.
  - Chốt G2L trên 48 câu dev + dev2 (47/48) trước khi chạy test 5 một lần.

| Nhóm test 5 | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | gồm 2 bảng trải hai trang |
| Ảnh, sơ đồ | 5/5 | 100% | |
| Biểu đồ | 5/5 | 100% | |
| Multi-hop | 2/4 | 50% | n22 khẳng định đáp án lời giải khi trang lời giải không có trong context; n24 cộng sai (78 thay vì 75) |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 6/6 | 100% | |
| **Toàn bộ** | **28/30** | **93,3%** | **đạt ngưỡng 90%** |

- **Trace, nhãn, chi phí:** `advanced/results/test-G2L/{traces,labels,grade_summary,summary}.json`. 0 lỗi hệ thống; 0 từ chối nhầm; 3,2 s/câu; $0 (Gemini free tier). Vòng 1 với OpenAI tốn $0,98 (vision $0,64 vẫn được dùng lại ở vòng 2). Nhãn test 5 do Claude chấm nháp (đối chiếu trang gốc), học viên Nguyễn Ngọc Toàn đã duyệt cả 30 câu, không sửa nhãn nào.
- **Lịch sử, minh bạch:** cấu hình E1 (OpenAI `gpt-4.1-mini`, cổng kiểm chứng) đạt 23/30 trên test cũ. Lỗi của E1 là cổng kiểm chứng chặn phép tính đúng, bảng trải hai trang, vision đọc sai và câu trả lời dài thêm ý ngoài nguồn. Vòng 2 sửa bằng cơ chế chung (tra cứu theo nhãn, ảnh gốc lúc trả lời, trả lời ngắn, bỏ cổng) và đánh giá trên test 5 mới.
- **Giới hạn:**
  - Người soạn benchmark cũng là người dựng pipeline (Claude).
  - Free tier giới hạn request/ngày.
  - Một số câu thiếu evidence được trả lời bằng nội dung "không có thông tin" thay vì tool `abstain`; chấm tay tính là đạt theo ADVANCED.md.
