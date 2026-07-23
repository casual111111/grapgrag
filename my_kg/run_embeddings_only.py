"""只重新跑向量编码（embedding）步骤，不重新构建图谱。

用法：
    cd my_kg
    python run_embeddings_only.py

前提：output/ 目录中已有 entities.parquet 和 text_units.parquet。
"""

import asyncio
import sys
from pathlib import Path

# 确保 graphrag 包在 path 中（如果用的是 editable install 可以删掉这两行）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "packages" / "graphrag"))

from graphrag.config.load_config import load_config
from graphrag.api.index import build_index
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks


async def main():
    root_dir = Path(__file__).resolve().parent

    # 加载 settings.yaml + .env
    config = load_config(root_dir)

    # 覆盖 workflows，只跑 embedding 这一步
    config.workflows = ["generate_text_embeddings"]

    embed_model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    print("开始重新生成向量编码...")
    print(f"  embedding model: {embed_model_config.model}")
    print(f"  embed fields: {config.embed_text.names}")
    print(f"  output dir: {config.output_storage.base_dir}")
    print()

    results = await build_index(
        config=config,
        callbacks=[NoopWorkflowCallbacks()],
    )

    for r in results:
        if r.error:
            print(f"[ERROR] {r.workflow}: {r.error}")
        else:
            print(f"[OK] {r.workflow} 完成")

    print("\n向量编码完成！")


if __name__ == "__main__":
    asyncio.run(main())
