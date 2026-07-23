"""
python run_index.py --root ./my_kg
GraphRAG Index Runner — 本地源码直接调用
==========================================
直接调用项目 packages/ 下的 graphrag 源码构建索引，
支持 VS Code 断点调试、日志保存、阶段计时、进度条。

用法:
    python run_index.py --root ./my_kg
    python run_index.py --root ./my_kg --method fast
    python run_index.py --root ./my_kg --verbose

日志文件: run_index_<YYYYMMDD_HHMMSS>.log（保存在当前工作目录）
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import sys
import time
from pathlib import Path

# Fix Windows GBK console encoding issues
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _now() -> str:
    """返回格式化的当前时间戳（毫秒精度）。"""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _elapsed(start: float) -> str:
    """返回从 start 到现在的耗时字符串。"""
    secs = time.time() - start
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"+{secs:6.2f}s"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"+{h}h{m:02d}m{s:02d}s"
    return f"+{m}m{s:02d}s"


def _setup_local_packages() -> None:
    """将本地 packages/ 下的所有子包加入 sys.path。"""
    project_root = Path(__file__).resolve().parent
    packages_dir = project_root / "packages"
    if packages_dir.is_dir():
        for pkg in sorted(packages_dir.iterdir()):
            pkg_str = str(pkg)
            if pkg_str not in sys.path:
                sys.path.insert(0, pkg_str)


def _parse_args(argv: list[str]) -> tuple[str, str, bool]:
    """解析命令行参数，返回 (root_dir, method, verbose)。"""
    root_dir = "."
    method = "standard"
    verbose = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--root" and i + 1 < len(argv):
            root_dir = argv[i + 1]
            i += 2
        elif arg == "--method" and i + 1 < len(argv):
            method = argv[i + 1]
            i += 2
        elif arg == "--verbose":
            verbose = True
            i += 1
        else:
            i += 1
    return root_dir, method, verbose


# ---------------------------------------------------------------------------
# Tee 写入器：同时输出到终端和日志文件
# ---------------------------------------------------------------------------

class TeeWriter:
    """同时写入终端和日志文件的 stream 包装器。"""

    def __init__(self, original, log_path: Path):
        self._original = original
        self._log = open(log_path, "a", encoding="utf-8")

    def write(self, data):
        self._original.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        return self._original.encoding

    def close(self):
        self._log.close()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    start_time = time.time()
    start_dt = _dt.datetime.now()

    # 解析参数
    root_dir, method, verbose = _parse_args(argv)
    root_path = Path(root_dir).resolve()

    # 创建日志文件
    log_name = f"run_index_{start_dt.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = Path.cwd() / log_name

    # 设置本地 packages 到 sys.path
    _setup_local_packages()

    # 导入 graphrag（使用本地源码）
    from graphrag.api import build_index
    from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
    from graphrag.config.enums import IndexingMethod
    from graphrag.config.load_config import load_config

    # Tee：同时输出到终端和日志文件
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeWriter(original_stdout, log_file)
    tee_stderr = TeeWriter(original_stderr, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        # 输出头部信息
        header = (
            f"{'=' * 60}\n"
            f"  GraphRAG Index Runner（本地源码）\n"
            f"{'=' * 60}\n"
            f"  开始时间 : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  工作目录 : {Path.cwd()}\n"
            f"  Root目录 : {root_path}\n"
            f"  索引方法 : {method}\n"
            f"  Python   : {sys.executable}\n"
            f"  日志文件 : {log_file}\n"
            f"{'=' * 60}\n"
        )
        print(header, end="")

        # 加载配置
        config = load_config(root_dir=root_path)
        method_enum = IndexingMethod(method)

        # 自定义 callback：继承 ConsoleWorkflowCallbacks，增加阶段耗时追踪
        stage_times: list[tuple[str, float, float]] = []
        current_stage_name: str | None = None
        current_stage_start: float = 0

        class TimingCallbacks(ConsoleWorkflowCallbacks):
            """继承 ConsoleWorkflowCallbacks，增加阶段耗时追踪。"""

            def workflow_start(self, name: str, instance: object) -> None:
                nonlocal current_stage_name, current_stage_start
                current_stage_name = name
                current_stage_start = time.time()
                print(f"\n▶ [{_now()}] 阶段开始: {name}")
                super().workflow_start(name, instance)

            def workflow_end(self, name: str, instance: object) -> None:
                nonlocal current_stage_name, current_stage_start
                end = time.time()
                if current_stage_name == name:
                    dur = end - current_stage_start
                    stage_times.append((name, current_stage_start, end))
                    print(f"✓ [{_now()}] 阶段完成: {name}  (耗时 {dur:.2f}s)")
                super().workflow_end(name, instance)

        timing_cb = TimingCallbacks(verbose=verbose)

        # 执行索引
        outputs = asyncio.run(
            build_index(
                config=config,
                method=method_enum,
                is_update_run=False,
                callbacks=[timing_cb],
                verbose=verbose,
            )
        )
        elapsed = time.time() - start_time

        # 统计结果
        errors = [o for o in outputs if o.error is not None]

        # 输出摘要
        summary = (
            f"\n{'=' * 60}\n"
            f"  运行摘要\n"
            f"{'=' * 60}\n"
            f"  结束时间 : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  总耗时   : {_elapsed(start_time)[1:].strip()} ({elapsed:.2f}s)\n"
            f"  退出码   : {1 if errors else 0}\n"
        )

        if stage_times:
            summary += f"\n  各阶段耗时:\n"
            max_name_len = max(len(name) for name, _, _ in stage_times)
            for name, s, e in stage_times:
                dur = e - s
                summary += f"    {name:<{max_name_len}}  {dur:8.2f}s\n"

        if errors:
            summary += f"\n  错误 ({len(errors)}):\n"
            for err in errors:
                summary += f"    - {err.workflow}: {err.error}\n"

        summary += f"{'=' * 60}\n"
        print(summary, end="")

        return 1 if errors else 0

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()
        print(f"\n完整日志已保存至: {log_file}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
