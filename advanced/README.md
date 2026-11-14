# Thử thách nâng cao — OpenStax *Introductory Statistics 2e*, Chương 1–5

> **Kết quả chính thức: test 5, cấu hình G2L (Gemini free tier) — 28/30 (93,3%), đạt ngưỡng 90%.** Lần test trước (E1, OpenAI) đạt 23/30; test đó được chuyển thành dev2. Test 5 được soạn và khóa mới trước khi chỉnh (xem mục "Vòng 2").

## Tài liệu và index

- **Nguồn:** OpenStax, *Introductory Statistics 2e* (2023), giấy phép CC BY-NC-SA 4.0 (trang bản quyền, PDF p4). Bản PDF web: <https://assets.openstax.org/oscms-prodcms/media/documents/introductory-statistics-2e_-_WEB.pdf>, SHA-256 `a79a953b…3a382a` (kiểm tra trong `prepare.py download`). PDF không được commit (`.gitignore`); tải lại bằng lệnh bên dưới.
- **Phạm vi:** Chương 1–5 (Sampling and Data, Descriptive Statistics, Probability Topics, Discrete Random Variables, Continuous Random Variables), PDF trang 17–346, tức **330 trang**. Ranh giới lấy từ mục lục PDF (đầu Chương 1 đến trước Chương 6). Phạm vi được chọn theo ngân sách vision, trước khi soạn benchmark.
- **Đã index:** 328 trang. Bỏ p76 và p238 (trang trống: dưới 200 ký tự và không có hình), danh sách ở `data/manifest.json`. Không bỏ trang homework hay solutions vì chúng chứa bảng, hình và lời giải hữu ích.
- **Pipeline (`prepare.py`):**
  - Mỗi trang có một chunk `:text` (PyMuPDF `get_text`). Bảng giữ nguyên trong text; bbox bảng (`find_tables`) và nhãn bảng/hình lưu ở metadata.
  - Hình trong sách là ảnh raster: lấy các ảnh ≥ 1% diện tích trang, gộp các vùng chồng nhau, crop có lề để giữ trục và caption.
  - 113 trang có 165 hình. Mỗi trang có hình thêm một chunk `:visual` do vision model mô tả, kèm bbox và đường dẫn ảnh trang (`data/pages/pNNN.png`).
  - Prompt vision có luật cho từng loại biểu đồ: đếm từng dấu X, liệt kê từng điểm scatter, xếp hạng cột, đọc box plot bằng nội suy, ghi phạm vi vùng tô, đánh dấu số ước lượng bằng "≈".
- **Vision 2 lượt:** `gpt-4.1-mini` cho cả 113 trang có hình ($0,155). Sau đó `gpt-4.1` đọc lại 69 trang có biểu đồ (dot plot, histogram, bar, box plot, line, scatter, pie, Pareto; chọn bằng từ khóa trong mô tả lượt 1, bỏ ảnh mở chương p179), tốn $0,487. Chunk dùng mô tả mới nhất; `visual.jsonl` giữ cả hai lượt.
- **Embedding:** 441 chunk (328 text + 113 visual), `text-embedding-3-small`, pgvector của Day 6 với strategy `ADV` ($0,0045).
- **Phần không xử lý được:** công thức toán dạng ảnh/MathML bị mất trong text (ví dụ μ = 1/p của phân phối geometric); mô tả do vision tạo ra có thể sai, nhất là khi đếm điểm hoặc ước lượng chiều cao không có số in sẵn. Kiểm chứng bằng ảnh trang gốc.

## Benchmark (`benchmark/questions.json`, khóa trước mọi lần chạy vision/retrieval/agent)

Hiện có 78 câu:

| Split | Id | Có đáp án | Thiếu evidence | Ghi chú |
|---|---|---|---|---|
| dev | d01–d18 | 15 (bảng 3, text 3, biểu đồ 3, ảnh/sơ đồ 3, multi-hop 3) | 3 | khóa ở vòng 1 |
| dev2 | t01–t30 | 24 | 6 | là test E1 của vòng 1; đã xem kết quả nên chuyển sang dev |
| **test** (test 5) | n01–n30 | 24 (bảng 6, text 4, biểu đồ 5, ảnh/sơ đồ 5, multi-hop 4) | 6 | soạn và khóa trước vòng 2, offset lấy mẫu mới |

SHA-256 lúc khóa: vòng 1 `148c5089…bce7f1`; sau khi thêm test 5 xem `benchmark/questions.sha256`. Kiểm tra bằng `python advanced/build_benchmark.py`: số câu tối thiểu, mọi quote text có nguyên văn trên trang, trang bắt buộc đã được index, từ khóa của câu thiếu evidence có 0 kết quả.

**Chọn câu bằng lấy mẫu hệ thống** (để tránh chọn câu dễ):
- **Hình:** 156 hình có thông tin (bỏ ảnh mở chương và ảnh 3.13), lấy mỗi hình thứ 6. Bỏ 2 khung trống (4.10, 5.48). Phân loại thành biểu đồ (có trục, dữ liệu) hoặc ảnh/sơ đồ (Venn, đường cong mật độ có vùng tô, màn hình máy tính), lấy 8 hình đầu mỗi loại, gán split theo mẫu T, D, T, T, D, T, D, T.
- **Bảng:** lấy mỗi bảng thứ 17 trong 160 bảng; bảng trống để học sinh tự điền thì thay bằng bảng có nội dung kế tiếp.
- **Text:** lấy mỗi mục thứ 3 trong 25 mục lý thuyết.
- **Multi-hop:** các hình mẫu còn dư ghép với định nghĩa, luật hoặc đề bài ở trang khác.
- **Test 5** dùng cùng quy tắc nhưng offset khác, chỉ lấy hình/bảng/mục chưa dùng: mỗi hình thứ 6 với offset 5, mỗi bảng thứ 17 với offset 0, các mục lý thuyết thứ 3, 6, 9, … Multi-hop ghép biểu đồ với lời giải hoặc bảng dữ liệu ở trang khác. Câu thiếu evidence: geometric mean, law of total probability, multinomial, Weibull, relative risk, và n30 hỏi một nhóm không có trong Figure 1.7.
- **Thiếu evidence:** chủ đề gần nhưng không có trong Chương 1–5 (coefficient of variation, kurtosis, negative binomial, Bayes, snowball sampling, trimmed mean, gamma, ogive). Riêng t30 hỏi Year 14 trong Table 1.27 (chỉ có Year 1–13), có bẫy là Table 1.17 ở p45 có hàng Year 14.

Đáp án hình/biểu đồ được đọc trên ảnh render trang gốc, **không** dựa vào mô tả vision. Benchmark do Claude (AI) soạn sau khi đọc trang gốc; học viên cần đối chiếu lại. Giới hạn: người soạn benchmark cũng là người dựng pipeline. Cấu hình được chọn chỉ trên dev, và mô tả vision của hình thuộc test không được xem trước khi chạy test.

## Cấu hình và lựa chọn (dev)

Agent `gpt-4.1-mini` (`starter/agent.py`), chạy qua `run.sh` với luật grounding 1–8. Cấu hình **E1** dùng lại cấu hình agent đã ổn định ở các thử nghiệm trước:
- hybrid RRF (dense pgvector `ADV` + BM25 có bigram), top-12, context 24.000 ký tự xếp theo rank;
- tối đa 4 search; search đầu là câu hỏi gốc; cần ≥ 2 search trước khi từ chối;
- khi model lặp query thì trả lời từ context hiện có; bỏ citation nằm ngoài context;
- cổng kiểm chứng v2 (lenient) rồi đến cổng v1.

| Cấu hình | dev (18 câu) | Context Evidence | Lỗi | s trung vị | $ |
|---|---|---|---|---|---|
| **E1** | **17/18** (bảng 3/3, text 3/3, biểu đồ 3/3, ảnh 3/3, multi-hop 2/3, thiếu evidence 3/3) | 15/15 | d13 từ chối nhầm: bản nháp đúng nhưng trang chứa luật 1.5×IQR (p100) không vào context, cổng v1 chặn | 4,9 | 0,131 |

Nhãn: `results/dev-E1/labels.json`. **Chốt E1 mà không chỉnh thêm.** Ngân sách ($1) chỉ đủ cho một lần dev và một lần test; với 17/18 trên dev, không có thay đổi nào đáng thử với rủi ro chỉnh quá khớp. Test chạy **một lần**:

```bash
cd starter
bash ../advanced/run.sh test E1 --method hybrid_rrf --top-k 12 --budget 24000 --max-searches 4 --first-original --min-searches 2 --bm25-bigrams --gate2 --gate2-lenient --gate --repeat-answer --drop-invalid-citations
```

## Kết quả test cũ (E1, chạy một lần): **23/30 (76,7%) — chưa đạt; test này sau đó chuyển thành dev2**

Nhãn từng câu và lý do: `results/test-E1/labels.json`, tổng hợp ở `grade_summary.json`, trace (retrieval, context, tool call, câu trả lời, citation, thời gian, chi phí) ở `traces.json`. Chấm đạt/không đạt bằng cách đối chiếu text và ảnh trang gốc. Đúng một phần hoặc có khẳng định không có trong nguồn đều tính không đạt.

| Nhóm | Đạt / tổng | Lỗi |
|---|---|---|
| Bảng | 4/6 | t02 từ chối nhầm; t04 sai bảng |
| Biểu đồ | 4/5 | t10 vision đếm sai dot plot |
| Ảnh, sơ đồ | 5/5 | |
| Text | 2/4 | t17, t20 thêm khẳng định ngoài nguồn |
| Multi-hop | 2/4 | t22 giá trị ≈ sai; t24 từ chối nhầm |
| Thiếu bằng chứng | 6/6 | gồm t30 (bẫy Year 14 của bảng khác) |
| **Toàn bộ** | **23/30 (76,7%)** | 25/30 nếu chấp nhận t17, t20; vẫn dưới ngưỡng |

0 lỗi hệ thống; 2 từ chối nhầm; trung vị 4,4 s/câu; test $0,207.

Phân tích lỗi:
- **Cổng kiểm chứng v1 quá chặt (t02, t24):**
  - t02: bản nháp đúng (12 và tổng 38 cộng từ các ô trên hai trang) nhưng cổng cho rằng nguồn "không nêu tổng".
  - t24: trước đó retrieval đã lấy nhầm Figure 5.12 thay vì 5.42, vì chunk hình p330 không vào context.
  - Trên dev, d13 cũng bị chặn như vậy. Đây là mẫu lỗi lặp lại: cổng chặn phép tính dựa trên các giá trị có trong nguồn.
- **Bảng trải hai trang (t04):** chỉ trang sau (p168) vào context, nên agent đọc nhầm Table 2.86 nằm cùng trang.
- **Vision (t10, t22):** gpt-4.1 đếm sai số dấu X; trả lời 0.26 thay vì 0.25 dù cột chạm đúng gridline 0.25.
- **Trả lời dài, thêm ý ngoài nguồn (t17, t20):** luật 6 (xóa câu không có trong nguồn) và cổng v2 lenient không chặn được.

**So với dev (17/18):** dev chỉ có 18 câu nên không đại diện được các lỗi trên. Từ chối nhầm và thêm ý ngoài nguồn xuất hiện nhiều hơn trên test. Không chỉnh tiếp trên test này; muốn cải thiện thì phải tạo test mới.

## Chi phí

Vision $0,642 (gpt-4.1-mini $0,155 + gpt-4.1 $0,487), embedding $0,0045, agent dev $0,131, test $0,207. **Tổng $0,98.**

## Tái lập

```bash
python advanced/prepare.py download && python advanced/prepare.py extract
python advanced/build_benchmark.py
python advanced/prepare.py vision gpt-4.1-mini
python advanced/prepare.py vision gpt-4.1 <danh sách trang biểu đồ, xem results/vision_chart_pages.txt>
python advanced/prepare.py chunks && python advanced/index.py   # cần Docker Day 6 đang chạy
cd starter && bash ../advanced/run.sh dev E1 <flags như trên>
```

## Vòng 2 (API miễn phí): G2L — chốt trước khi chạy test 5

API OpenAI hết ngân sách nên vòng 2 dùng **Gemini API free tier**: `gemini-3.5-flash-lite` cho agent và `gemini-embedding-001` (1536 chiều) cho embedding, lưu ở strategy `ADVG`, chi phí $0. Test E1 đã bị xem kết quả nên được chuyển thành **dev2**. **Test 5** (30 câu mới, id `n01–n30`, lấy mẫu hệ thống với offset mới từ các hình/bảng/mục chưa dùng) được soạn và khóa trước mọi thay đổi ở vòng này; SHA-256 ghi ở `benchmark/questions.sha256`.

`advanced/rag.py` (cấu hình G2L):
- **Tra cứu theo nhãn:** câu hỏi nêu "Figure/Table X.Y" thì lấy mọi trang có nhãn đó. Bảng trải hai trang được lấy đủ; nhãn được phân tích từ câu hỏi, không dùng gold.
- **Hybrid RRF** dense ADVG + BM25 bigram, top-8; mỗi trang mang theo cả chunk text và chunk visual của nó.
- **Ảnh crop hình gốc** của các trang được tra theo nhãn được gửi kèm lời gọi trả lời; khi ảnh và mô tả vision mâu thuẫn, model được dặn tin ảnh.
- Tool calling `search`/`answer`/`abstain`; tối đa 1 search thêm; bước cuối chỉ được chọn answer/abstain. Trả lời tối đa 3 câu, mỗi khẳng định có citation; chỉ giữ citation nằm trong context.

| Cấu hình | dev (18) | dev2 (30) | Lỗi |
|---|---|---|---|
| G1 (`gemini-3.5-flash`) | 17/17 câu chạy được (d18 bị quota ngày 20 request) | — | — |
| G1L (`gemini-3.5-flash-lite`) | 17/18 | — | d03: model lặp lại search rồi kết thúc bằng từ chối |
| **G2L** = G1L + bước cuối không có search | **18/18** | **29/30** | t10: model ghi tổng 11 dấu X (ảnh gốc 10) |

Chọn **G2L** vì đạt 47/48 trên 48 câu dev; flash-lite nhanh (khoảng 3 s/câu) và có quota ngày đủ cho test. Test 5 chạy **một lần**:

```bash
python advanced/rag.py --split test --name G2L --model gemini-3.5-flash-lite --final-no-search
```

### Kết quả test 5 (G2L, chạy một lần): **28/30 (93,3%) — đạt ngưỡng 90%**

Nhãn từng câu và lý do: `results/test-G2L/labels.json`, tổng hợp ở `grade_summary.json`, trace (nhãn được tra, kết quả retrieval kèm rank/score, context, trang có ảnh gửi kèm, tool call, câu trả lời, citation, thời gian, token) ở `traces.json`. Chấm nghiêm như vòng trước: đúng một phần, citation thiếu trang bắt buộc, hoặc khẳng định không có trong nguồn đều tính không đạt.

| Nhóm | Đạt / tổng | Lỗi |
|---|---|---|
| Bảng | 6/6 | |
| Biểu đồ | 5/5 | |
| Ảnh, sơ đồ | 5/5 | |
| Text | 4/4 | |
| Multi-hop | 2/4 | n22: nói "lời giải chọn c (41%)" khi trang lời giải p176 không có trong context; n24: cộng sai (ghi 78, đúng là 75) |
| Thiếu bằng chứng | 6/6 | 2 câu abstain, 4 câu nêu rõ "nguồn không có thông tin" và không thêm khẳng định |
| **Toàn bộ** | **28/30 (93,3%)** | |

0 lỗi hệ thống; 0 từ chối nhầm; 43 lời gọi model; trung vị 3,2 s/câu; chi phí $0 (Gemini free tier).

**So với E1 (23/30):**
- Tra cứu theo nhãn sửa được lỗi bảng trải hai trang và lỗi lấy nhầm hình.
- Ảnh gốc gửi kèm giúp đọc đúng biểu đồ (biểu đồ 5/5, ảnh/sơ đồ 5/5).
- Bỏ cổng kiểm chứng, giới hạn câu trả lời 3 câu và bước cuối chỉ được answer/abstain giúp loại hết từ chối nhầm và ý thừa.

**Lỗi còn lại:** số học sai (n24); khẳng định về trang không được lấy (n22); đếm sai trên dev2 (t10).

**Giới hạn:**
- Benchmark do Claude soạn, và Claude cũng dựng pipeline; cấu hình chỉ chọn trên dev + dev2 (48 câu).
- Câu thiếu bằng chứng đôi khi được trả lời bằng `answer` có nội dung "không có thông tin" thay vì tool `abstain`, nên số đếm tự động `abstention_unanswerable` trong `summary.json` thấp hơn kết quả chấm tay.
- Free tier giới hạn khoảng 20 request/ngày cho các model flash, và 30.000 token/phút cho embedding.

### Tái lập vòng 2

```bash
python advanced/build_benchmark.py
python advanced/index.py --gemini        # strategy ADVG, cần GEMINI_API_KEY và Docker Day 6
python advanced/rag.py --split dev  --name G2L --model gemini-3.5-flash-lite --final-no-search
python advanced/rag.py --split dev2 --name G2L --model gemini-3.5-flash-lite --final-no-search
python advanced/rag.py --split test --name G2L --model gemini-3.5-flash-lite --final-no-search
```
Mô tả vision (`data/visual.jsonl`) được tạo ở vòng 1 bằng OpenAI ($0,64); vòng 2 dùng lại, không gọi lại vision.

## Kết quả chính thức

**Test 5 với cấu hình G2L: 28/30 (93,3%) ≥ 90%.** Lịch sử để minh bạch: E1 (OpenAI) đạt 23/30 trên test cũ; test đó đã được chuyển thành dev2 trước khi chỉnh, và test 5 được soạn và khóa mới trước vòng 2. Tổng chi phí: vòng 1 $0,98 (OpenAI), vòng 2 $0.
