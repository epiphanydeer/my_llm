"""3.8 Step 8: 在 Step 3/7 基础上新增父子层级扩展（Edge-placement）与 Python AST 代码切片（带语法错误降级）"""
import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    qualified_name: str
    start_line: int  # 含装饰器
    end_line: int
    content: str


def ast_chunk_python(source: str, fallback_window: int = 20) -> list[CodeChunk]:
    """【Step 8 新增】基于 AST 提取函数/类（含装饰器行号），语法错误时自动降级滑窗。"""
    if not source.strip():
        return []
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            CodeChunk(f"fallback_L{i+1}", i + 1, min(i + fallback_window, len(lines)), "".join(lines[i : i + fallback_window]))
            for i in range(0, len(lines), fallback_window)
        ]

    chunks: list[CodeChunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decs = getattr(node, "decorator_list", [])
            start = min([d.lineno for d in decs] + [node.lineno])
            end = node.end_lineno or node.lineno
            chunks.append(CodeChunk(node.name, start, end, "".join(lines[start - 1 : end])))
    return chunks


def expand_parent_and_reorder(
    ranked_child_ids: list[str],
    child_to_parent: dict[str, str],
    parent_texts: dict[str, str],
    max_parents: int = 3,
) -> list[str]:
    """先按 Child 精排顺序去重映射为 Parent，再做首尾倒置摆放（破除 Lost in the Middle）。"""
    parents: list[str] = []
    seen: set[str] = set()
    for cid in ranked_child_ids:
        pid = child_to_parent[cid]
        if pid not in seen:
            seen.add(pid)
            parents.append(parent_texts[pid])
        if len(parents) >= max_parents:
            break
    # Edge-placement: 最相关放最前，第二相关放最后（紧贴 Question），其余放中间
    if len(parents) >= 3:
        return [parents[0]] + parents[2:] + [parents[1]]
    return parents


if __name__ == "__main__":
    code = "@login_required\ndef verify_token(tok: str) -> bool:\n    return bool(tok)\n"
    res = ast_chunk_python(code)
    assert len(res) == 1 and res[0].start_line == 1 and "@login_required" in res[0].content
    assert len(ast_chunk_python("def broken(:\n  pass")) == 1, "语法错误必须优雅降级而非崩溃"

    ordered = expand_parent_and_reorder(["c1", "c2", "c3"], {"c1": "p1", "c2": "p2", "c3": "p3"}, {"p1": "P1", "p2": "P2", "p3": "P3"})
    assert ordered == ["P1", "P3", "P2"], f"首尾倒置摆放应为 [P1, P3, P2]，实际: {ordered}"
    print("Step 8 self-test passed!")
