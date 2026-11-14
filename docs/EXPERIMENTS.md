# Thử nghiệm Day 7

## Tiêu chí chọn cấu hình (chốt 2026-11-14, trước khi chạy các cấu hình cải thiện)

Chỉ dùng **dev** (18 câu có đáp án + 3 câu thiếu evidence), theo thứ tự:
1. Context Evidence Rate của agent cao nhất (chênh ≤ 1 câu coi là hòa).
2. Từ chối nhầm ít hơn, và abstention trên câu thiếu evidence không giảm.
3. Điểm grounded answer (chấm tay) cao hơn.
4. Thời gian trung vị / chi phí thấp hơn.

Test chỉ chạy một lần cho baseline và cấu hình đã chọn sau khi chốt.

## Ghi chú nhãn

- q11: lúc đầu nghi gold thiếu trang 225 (`p225:p1` nói sort() xếp ascending). Đọc lại: đáp án gold là điều kiện *"all elements of the list are of the same type"*, chỉ có ở trang 235; `p225:p1` không nêu điều kiện này. **Nhãn đúng, giữ nguyên.**

## 1. Retrieval trên dev (18 câu có đáp án, top-3, budget 1500 ký tự)

Cùng corpus (203 chunk strategy B), cùng index pgvector, cùng định nghĩa hit (evidence nằm trọn trong chunk đúng nguồn/trang, không phân biệt khoảng trắng). Nguồn: `starter/results/retrieval-dev/`.

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence | MRR@3 | ms trung vị | $ |
|---|---|---|---|---|---|---|
| tfidf | 7/18 | 9/18 | 9/18 | 0,435 | 0,6 | 0 |
| bm25 | 7/18 | 11/18 | 10/18 | 0,481 | 0,6 | 0 |
| bm25+filter | 8/18 | 12/18 | 11/18 | 0,537 | 0,6 | 0 |
| dense | 6/18 | 10/18 | 10/18 | 0,435 | 224 | 0,000005 |
| dense+filter | 6/18 | 12/18 | 12/18 | 0,491 | 209 | 0,000005 |
| hybrid_rrf | 6/18 | 14/18 | 12/18 | 0,528 | 205 | 0,000005 |
| hybrid_rrf+filter | 7/18 | 16/18 | 14/18 | 0,611 | 199 | 0,000005 |

## 2. Agent trên dev (gpt-4o-mini, temperature 0, tối đa 3 search, top-3/search)

Nguồn: `starter/results/exp-dev/R*/`. Chi phí = LLM + embedding query, tổng cho 21 câu.

| Run | Cấu hình | Giả thuyết | Context Evidence | Từ chối nhầm | Abstention đúng | Phân bố số search | s trung vị | $ / 21 câu | Context TB (ký tự) |
|---|---|---|---|---|---|---|---|---|---|
| R0 | dense, budget 1500, thứ tự đến | baseline | 6/18 | 4/18 | 3/3 | 1:13, 2:3, 3:5 | 4,72 | 0,00515 | 1141 |
| R1 | R0 + context theo rank | chunk tốt ở search sau bị loại khi hết budget | 7/18 | 5/18 | 3/3 | 1:12, 2:3, 3:6 | 4,48 | 0,00516 | 1118 |
| R2 | bm25+filter, budget 3000 | phương pháp lexical | 12/18 | 1/18 | 3/3 | 1:16, 2:2, 3:3 | 3,49 | 0,00533 | 1653 |
| R3 | hybrid_rrf+filter, budget 3000 | kết hợp dense + lexical | 15/18 | 0/18 | 3/3 | 1:19, 3:2 | 3,67 | 0,00474 | 1525 |
| R4 | R3 + rank + dừng khi lặp query | bỏ search lặp vô ích | 15/18 | 0/18 | 3/3 | 1:19, 2:1, 3:1 | 3,71 | 0,00463 | 1567 |
| **R5** | R4 với budget 1500 | budget nhỏ hơn không mất evidence | **15/18** | **0/18** | **3/3** | 1:19, 2:1, 3:1 | **3,35** | **0,00419** | **1181** |

Chấm tay (bản nháp do Claude chấm, **học viên cần duyệt**: `R5.../review.csv`, tổng hợp `agent_metrics_draft.json`): grounded answer R5 = 15×2, 3×1, 0×0, trung bình 1,83/2; citation hỗ trợ 21/22; 18/18 câu trả lời có citation. R3/R4 chỉ khác R5 ở citation phụ (q02, q03, q05, q09, q12 có thêm 1 citation), status và chunk chứa gold giống nhau.

Ghi chú: model không hoàn toàn tất định dù temperature 0; chênh 1 câu giữa các lần chạy (R0 cũ 7/18 → lần chạy lại 6/18) được coi là nhiễu.

## 3. Phân tích lỗi dev

**Lỗi 1 — từ chối nhầm hàng loạt (đã sửa).** Lần chạy hybrid đầu tiên: từ chối nhầm 15/18. Đọc trace: (a) q06, q09, q12, q15 gọi `abstain` khi chưa search; (b) q10, q13, q14 có evidence trong context nhưng model ghi citation mà không có `[id]` trong câu trả lời, nên `run_agent` loại. Sửa: bắt buộc `search` ở bước đầu; nối marker `[id]` còn thiếu (có log `citation_markers_appended`; ID vẫn phải nằm trong context). Sau sửa: từ chối nhầm 0/18, Context Evidence 15/18, abstention 3/3 không đổi.

**Lỗi 2 — dense agent thiếu evidence trong context (R0 6/18).** Giả thuyết R1: `pack_context` ưu tiên chunk của search đầu nên chunk tốt ở search sau bị loại. Đo: trong 12 câu R0 thiếu evidence, chỉ 2 câu có gold trong kết quả search nhưng bị loại khỏi context; 10 câu **chưa bao giờ được dense tìm thấy**. R1 chỉ tăng 6→7 (trong mức nhiễu). Kết luận: giả thuyết sai phần lớn, nguyên nhân chính là dense retrieval (từ vựng câu hỏi khác sách, chunk ngắn), không phải cách xếp context. Thay phương pháp sang hybrid+filter (R3) tăng lên 15/18.

**Lỗi 3 — search lặp cùng query.** u02, u03 (thiếu evidence) search 3 lần với cùng query, kết quả giống nhau. R4/R5 dừng và từ chối khi query lặp: tổng số search của u02+u03 giảm từ 6 xuống 5, abstention vẫn 3/3, chi phí giảm ~2%. Lợi ích nhỏ; chủ yếu chặn vòng lặp vô ích.

**Lỗi còn lại (chưa sửa):** q15 nêu `dict()` không có trong context (chunk `p239:p3` chỉ có trắc nghiệm `{}`); q07 thêm "if an index is provided" ngoài nguồn; q11 thiếu điều kiện "same type" vì gold chunk p235 không được lấy. Đây là lỗi model dùng kiến thức ngoài, mà Context Evidence và kiểm tra ID không phát hiện được.

## 4. Lựa chọn (theo tiêu chí ở đầu file)

1. Context Evidence: R3 = R4 = R5 = 15/18 (hòa), cao hơn R2 (12) và R0/R1 (6–7).
2. Từ chối nhầm 0/18 và abstention 3/3 ở cả ba (hòa).
3. Grounded answer: chấm R5 = 1,83/2; R3/R4 có cùng status và chunk evidence, chỉ thêm citation phụ (hòa trong giới hạn đã kiểm tra).
4. Thời gian/chi phí: R5 nhanh nhất (3,35 s) và rẻ nhất ($0,00419), context nhỏ hơn 23%.

**Cấu hình chốt: R5** = hybrid_rrf (dense Day 6 + BM25, RRF k=60, 20 candidate mỗi bên) + filter chapter do người dùng cung cấp, top-3/search, tối đa 3 search, context theo rank, budget 1500 ký tự, dừng khi lặp query, gpt-4o-mini temperature 0. **Baseline cho test: R0** (dense, budget 1500).

```bash
cd starter
PYTHONUTF8=1 python agent.py --split test --method dense --budget 1500 --output results/exp-test/R0-dense-b1500
PYTHONUTF8=1 python agent.py --split test --method hybrid_rrf --filter --budget 1500 --context-order rank --stop-repeat --output results/exp-test/R5-hybridf-b1500-rank-stop
```
