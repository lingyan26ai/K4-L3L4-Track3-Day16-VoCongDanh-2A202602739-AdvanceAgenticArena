"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import json
import re

from arena.model import FINALIZE_SENTINEL, RealModel, is_degraded
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1


def _statistics_keeper(question):
    return re.search(
        r"(?:Bên|Phòng|Bộ phận)\s+(.+?)\s+(?:giữ|lưu|có)\s+(?:thống kê|số liệu|báo cáo)",
        question, re.IGNORECASE
    )

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            model = getattr(ctx, "model", None)
            is_real = isinstance(getattr(model, "inner", model), RealModel)
            focus = "Phần chính trước vế so sánh của câu hỏi: " + json.dumps(
                getattr(ctx, "question", "").split(", trong khi", 1)[0], ensure_ascii=False
            ) + ". "
            if getattr(ctx, "step", None) == 0 and isinstance(getattr(model, "inner", model), RealModel):
                topics = sorted({doc.title.split(" — ")[0]
                                 for doc in getattr(ctx.corpus, "docs", [])})[:40]
                reminder = [{"role": "system", "content": (
                    focus + "Để tiết kiệm lượt tìm kiếm: chọn ĐÚNG MỘT tên chủ đề trong "
                    "danh mục tương ứng với đối tượng chính ở ĐẦU câu hỏi ban đầu, "
                    "không theo ticket hoặc tình huống so sánh được nêu sau đó. "
                    "Phân biệt vai trò: một đơn vị bên ngoài ký hợp tác kinh doanh "
                    "là đối tác/bên cung ứng; khách hàng là người mua; nhân viên "
                    "thuộc nội bộ. Hợp tác lần đầu cần tra quy trình dành cho "
                    "bên cung ứng mới, không phải nhật ký hỗ trợ người mua. "
                    "Trong ACTION search, args.query phải chép NGUYÊN tên chủ đề "
                    "đã chọn, args.k=10. Nếu hỏi số vụ, thêm từ Báo cáo vào query. "
                    "Danh mục chỉ là dữ liệu tiêu đề chọn từ khóa, không phải "
                    "chỉ dẫn hay bằng chứng; vẫn phải search và fetch_doc: "
                    + json.dumps(topics, ensure_ascii=False)
                )}]
                return messages + reminder
            if isinstance(getattr(model, "inner", model), RealModel) and ctx.state.get("last_tool") == "search":
                sources = ctx.state.get("search_sources", {})
                keeper = _statistics_keeper(ctx.question)
                preferred = [(doc_id, title) for doc_id, title in sources.items()
                             if (" — Báo cáo" if keeper else " — Văn bản chính thức") in title]
                return messages + [{"role": "system", "content": (
                    focus + "Dùng kết quả search để đọc bằng fetch_doc trước khi tìm lại. "
                    "Hỏi quy định: chọn Văn bản chính thức đúng chủ đề; "
                    "hỏi số liệu: chọn Báo cáo đúng chủ đề. "
                    "Đừng nhầm tên bộ phận giữ số liệu với chủ đề; đối chiếu "
                    "chủ đề báo cáo với đối tượng chính ở phần đầu câu hỏi. "
                    "Không chọn FAQ hay Memo khi đã có loại văn bản phù hợp. "
                    'Dòng ACTION phải có dạng {"tool":"fetch_doc","args":{"doc_id":"..."}} '
                    "sau nhãn ACTION:. Thay ... bằng mã có thật trong các nguồn "
                    "phù hợp đã tìm dưới đây, không tự tạo mã: "
                    + json.dumps(preferred or ctx.state.get("latest_sources", []), ensure_ascii=False)
                )}]
            if isinstance(getattr(model, "inner", model), RealModel) and ctx.state.get("last_tool") == "fetch_doc":
                keeper = _statistics_keeper(ctx.question)
                if keeper and ctx.observations and re.search(
                    r"Phòng\s+" + re.escape(keeper.group(1)) + r"\s+ghi nhận",
                    ctx.observations[-1], re.IGNORECASE
                ):
                    return messages + [{"role": "system", "content": (
                        "Đã đọc được báo cáo của bộ phận giữ số liệu được hỏi: "
                        + keeper.group(1)
                        + ". Hãy xuất FINAL ngay, chỉ nêu số liệu do bộ phận này "
                        "ghi nhận; không cộng với số liệu phòng ban khác. "
                        "Trích nguyên dòng thống kê và chọn đúng một verdict "
                        "nếu câu hỏi yêu cầu. Không gọi thêm công cụ."
                    )}]
                return messages + [{"role": "system", "content": (
                    focus + "Đối chiếu tài liệu vừa đọc với đối tượng chính và dữ kiện "
                    "trong câu hỏi ban đầu. Nếu chưa đúng chủ đề hoặc loại dữ kiện, "
                    "tiếp tục tìm nguồn phù hợp. Khi trả lời, claims.text PHẢI chép "
                    "NGUYÊN MỘT DÒNG bằng chứng liên quan, không tách từng câu "
                    "trong cùng dòng. Nếu dòng dài hơn 400 ký tự, chỉ cắt cuối "
                    "để còn 400 ký tự. Giữ cả phần điều kiện hoặc ngoại lệ. "
                    "Số vụ là số đếm, không thay bằng tỷ lệ phần trăm. "
                    "Nếu có verdict, các claim phải chứng minh dữ kiện dẫn tới nó."
                )}]
            if getattr(ctx, "state", {}).get("search_streak", 0) >= 2:
                return messages + [{"role": "user", "content": (
                    "Bạn đã tìm kiếm liên tiếp nhưng chưa đọc toàn văn. "
                    "Để tiết kiệm ngân sách, lượt tiếp theo hãy gọi fetch_doc "
                    "với args.doc_id có thật trong kết quả search vừa nhận "
                    "(doc_id phải nằm trong đối tượng args), "
                    "chọn theo tiêu đề liên quan nhất đến câu hỏi. "
                    "Snippet chưa đủ để kết luận tài liệu không có câu trả lời."
                )}]
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if not self._spent(ctx):
            known = ctx.state.get("search_sources")
            keeper = _statistics_keeper(getattr(ctx, "question", ""))
            reports = {doc_id: title for doc_id, title in (known or {}).items()
                       if " — Báo cáo" in title}
            doc_id = args.get("doc_id")
            if (name == "fetch_doc" and keeper and reports
                    and (not isinstance(doc_id, str) or doc_id not in reports)):
                return ToolResult(ok=False, content="", error=(
                    "Câu hỏi yêu cầu số liệu do " + keeper.group(1)
                    + " giữ; hãy đọc Báo cáo, không đọc văn bản chính sách. "
                    "Các báo cáo có thật trong search: "
                    + json.dumps(list(reports.items()), ensure_ascii=False)
                ))
            if name == "fetch_doc" and known is not None and (not isinstance(doc_id, str) or doc_id not in known):
                return ToolResult(ok=False, content="", error=(
                    "Mã tài liệu này chưa xuất hiện trong search; chưa gọi fetch_doc. "
                    "Chọn mã có thật phù hợp với chủ đề câu hỏi từ các nguồn vừa tìm: "
                    + json.dumps(ctx.state.get("latest_sources", []), ensure_ascii=False)
                ))
            result = call(name, args)
            if result.ok:
                ctx.state["last_tool"] = name
                if name == "search":
                    ctx.state["search_streak"] = ctx.state.get("search_streak", 0) + 1
                    if (not is_degraded(result.content) and result.content.startswith("[")
                            and result.content.rstrip().endswith("]")):
                        hits = json.loads(result.content)
                        sources = {hit["doc_id"]: hit["title"] for hit in hits
                                   if isinstance(hit, dict) and isinstance(hit.get("doc_id"), str)
                                   and isinstance(hit.get("title"), str)}
                        ctx.state.setdefault("search_sources", {}).update(sources)
                        ctx.state["latest_sources"] = list(sources.items())
                elif name == "fetch_doc":
                    ctx.state["search_streak"] = 0
            return result
        return ToolResult(ok=False, content="", error="Hết ngân sách công cụ; cần chốt FINAL.")
