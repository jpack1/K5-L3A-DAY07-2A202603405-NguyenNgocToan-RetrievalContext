# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.

## Kết nối Day 6 (bài làm)

| Thành phần | Giá trị |
|---|---|
| Corpus | OpenStax *Introduction to Python Programming* ch5/9/10, 59 trang, CC BY-NC-SA 4.0 (manifest ở repo Day 6) |
| Chunk | Day 6 strategy **B** (by_paragraph ≤500 ký tự, cấu hình C đã chọn ở Day 6 = B + filter `chapter`): 203 chunk, ID `openstax-intro-python:p<page>:p<n>` |
| Embedding | OpenAI `text-embedding-3-small`, 1536 chiều, query/document embed giống nhau, không prefix |
| Vector store | PostgreSQL 16 + pgvector 0.8.7 (Docker Day 6, `127.0.0.1:55433`), bảng `chunks`, exact cosine `<=>` |
| Agent model | `gpt-4o-mini` (tool calling) |

```bash
pip install -r starter/requirements.txt
# Copy starter/.env.example -> starter/.env, điền OPENAI_API_KEY
python starter/day6_store.py "How do I leave an infinite loop?"   # kiểm tra model/số chiều + dense top-3
python starter/prepare_day6.py                                       # -> starter/data/{chunks,questions,cases}.json
```

`day6_store.py` báo lỗi nếu model của index khác `EMBEDDING_MODEL` hoặc số chiều query khác index. `prepare_day6.py` chỉ xuất text/metadata (vector ở lại pgvector), gộp `alt_evidence` của Day 6 vào `gold` và thêm câu thiếu evidence từ `data/unanswerable.json`. Benchmark: dev 18 có đáp án + 3 thiếu evidence; test 6 + 2. Hit khớp evidence không phân biệt khoảng trắng, giống quy tắc Day 6. Trên Windows chạy test với `PYTHONUTF8=1`.

## Retrieval (bài làm)

`starter/retrieval.py`: `tfidf` (starter), `bm25` (k1=1.5, b=0.75, tách từ `\w+` casefold), `dense` (pgvector Day 6), `hybrid_rrf` (RRF k=60 trên top-20 dense + top-20 BM25; dùng rank nên không cần chuẩn hóa score). Biến thể `+filter` dùng `filter.chapter` do người dùng cung cấp trong câu hỏi. Context = top-3 qua `pack_context`, budget 1500 ký tự.

```bash
cd starter
PYTHONUTF8=1 python retrieval.py --chunks data/chunks.json --questions data/questions.json --split dev --output results/retrieval-dev
```

## Agent (bài làm)

`starter/agent.py`: controller OpenAI (`gpt-4o-mini`, temperature 0, tool calling) gồm 3 tool `search(query)`, `answer(answer, citations)`, `abstain(reason)`; vòng lặp, giới hạn search, kiểm tra citation và từ chối dùng `run_agent` của starter. Bước đầu bắt buộc `search` (không được từ chối khi chưa tìm). Nếu model liệt kê citation mà thiếu `[id]` trong câu trả lời, controller nối marker vào cuối và ghi `citation_markers_appended` trong trace; `run_agent` vẫn kiểm tra ID nằm trong context. Filter `chapter` (do người dùng cung cấp) được hệ thống áp dụng, model không tự đặt filter. Lỗi provider/DB ghi `status: error`, không thay bằng kết quả giả. Gold chỉ đọc sau khi chạy để tính Context Evidence Rate.

```bash
cd starter
PYTHONUTF8=1 python agent.py --split dev --method dense --budget 1500 --output results/exp-dev/R0-dense-b1500
PYTHONUTF8=1 python agent.py --split dev --method hybrid_rrf --filter --budget 1500 --context-order rank --stop-repeat --output results/exp-dev/R5-hybridf-b1500-rank-stop
```
