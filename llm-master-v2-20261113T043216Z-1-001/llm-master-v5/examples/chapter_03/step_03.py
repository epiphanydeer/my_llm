"""3.3 Step 3: 在 Step 2 基础上升级 ingest()，新增带重叠、标点吸附与字符偏移校验的滑窗切片器"""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    version: str
    effective_date: date
    expire_date: date
    roles: tuple[str, ...] = ("employee",)


@dataclass(frozen=True)
class Chunk:
    """【Step 3 升级】新增 start_char / end_char，支持前端高亮精准溯源。"""
    chunk_id: str
    doc_id: str
    version: str
    content: str
    # [必记设计决策-偏移溯源] 必须记录 start_char / end_char 字符坐标，支撑前端原文高亮与可逆校验
    start_char: int
    end_char: int
    effective_date: date
    expire_date: date
    roles: tuple[str, ...]


class SlidingWindowChunker:
    """【Step 3 新增】支持重叠、标点自然吸附与可逆偏移断言的切分器。"""
    def __init__(self, chunk_size: int = 40, chunk_overlap: int = 10, snap_punct: bool = True) -> None:
        if chunk_size <= 0 or not (0 <= chunk_overlap < chunk_size):
            raise ValueError("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.snap_punct = snap_punct
        self._puncts = ("。", "！", "？", "；", "\n")

    def split_document(self, doc: Document) -> list[Chunk]:
        text = doc.content
        n = len(text)
        if n == 0:
            return []
        chunks: list[Chunk] = []
        start, idx = 0, 0
        while start < n:
            end = min(start + self.chunk_size, n)
            # [必背手撕-标点吸附] 在窗口尾部 40% 区间 rfind 自然标点，避免硬切截断完整语义句
            if self.snap_punct and end < n:
                # 在窗口尾部 40% 范围内寻找最后一个自然标点，防止切断完整句子
                search_left = max(start + int(self.chunk_size * 0.6), start + 1)
                snapped = max((text.rfind(p, search_left, end) for p in self._puncts), default=-1)
                if snapped != -1:
                    end = snapped + 1
            piece = text[start:end]
            # [必记设计决策-可逆不变量] 切片与原文档区间必须严格恒等，禁止因提前清洗破坏字符坐标
            assert doc.content[start:end] == piece, "偏移切片必须与原文档区间严格恒等"
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c{idx:02d}",
                doc_id=doc.doc_id,
                version=doc.version,
                content=piece,
                start_char=start,
                end_char=end,
                effective_date=doc.effective_date,
                expire_date=doc.expire_date,
                roles=doc.roles,
            ))
            if end == n:
                break
            # [必背手撕-防死循环] 重叠回退时若 next_start <= start 必须强制推进到 end，防止步长退化死循环
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end  # 防止步长退化引发死循环
            idx += 1
        return chunks


if __name__ == "__main__":
    raw = "未经通风检测合格，严禁进入有限空间作业。违者予以停职处分。进入前必须佩戴气体检测仪。"
    doc = Document("DOC_SPACE", "受限空间", raw, "v1", date(2026, 1, 1), date(2027, 12, 31))
    chunker = SlidingWindowChunker(chunk_size=28, chunk_overlap=8, snap_punct=True)
    res = chunker.split_document(doc)
    assert len(res) >= 2
    assert res[0].content.endswith("。"), f"标点吸附应使首块在句号处收尾，实际: {res[0].content}"
    assert res[1].start_char == res[0].end_char - 8, "第二块起点必须保留 overlap 重叠"
    assert all(doc.content[c.start_char:c.end_char] == c.content for c in res)
    print("Step 3 self-test passed!")
