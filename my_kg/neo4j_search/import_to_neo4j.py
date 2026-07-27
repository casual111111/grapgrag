"""将 GraphRAG 的 parquet 数据 + embedding 导入 Neo4j。

用法：
    cd my_kg
    python neo4j_search/import_to_neo4j.py

前提：Neo4j 容器已通过 start_neo4j.py 启动。

数据模型：
    节点:
        (e:Entity {id, title, type, description, frequency, degree, embedding})
        (t:TextUnit {id, text, document_id, n_tokens})
        (d:Document {id, title, text})

    关系:
        (e1)-[r:RELATES_TO {id, description, weight}]->(e2)
        (e)-[:MENTIONED_IN]->(t)
        (t)-[:FROM_DOCUMENT]->(d)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

# ─── 配置 ────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_test"

# 数据路径（相对于 my_kg/）
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
LANCEDB_DIR = OUTPUT_DIR / "lancedb"

BATCH_SIZE = 200  # 每批导入数量


def load_data() -> dict:
    """加载所有 parquet 文件和 embedding 向量。"""
    print("[1/5] 加载 parquet 数据...")

    entities = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    relationships = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
    text_units = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    documents = pd.read_parquet(OUTPUT_DIR / "documents.parquet")

    print(f"  entities:      {len(entities)}")
    print(f"  relationships: {len(relationships)}")
    print(f"  text_units:    {len(text_units)}")
    print(f"  documents:     {len(documents)}")

    # 加载 embedding
    print("[2/5] 加载 embedding 向量...")
    try:
        import lancedb
        db = lancedb.connect(str(LANCEDB_DIR))
        emb_table = db.open_table("entity_description")
        emb_df = emb_table.to_pandas()[["id", "vector"]]
        # 转为 dict: entity_id -> vector list
        embeddings = dict(zip(emb_df["id"], emb_df["vector"]))
        print(f"  embeddings: {len(embeddings)} (维度: {len(next(iter(embeddings.values())))})")
    except Exception as e:
        print(f"  [WARN] 无法加载 embedding: {e}")
        print("  将跳过 embedding 导入（向量检索不可用）")
        embeddings = {}

    return {
        "entities": entities,
        "relationships": relationships,
        "text_units": text_units,
        "documents": documents,
        "embeddings": embeddings,
    }


def create_constraints(driver):
    """创建唯一性约束和索引。"""
    print("[3/5] 创建约束和索引...")

    constraints = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT textunit_id IF NOT EXISTS FOR (t:TextUnit) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    ]

    with driver.session() as session:
        for constraint in constraints:
            session.run(constraint)
        print("  约束创建完成")

    # 创建向量索引（需要 Neo4j 5.x）
    _create_vector_index(driver)


def _create_vector_index(driver):
    """创建 entity embedding 的向量索引。"""
    try:
        with driver.session() as session:
            # 先检查索引是否存在
            result = session.run("SHOW INDEXES YIELD name WHERE name = 'entity_embedding'")
            if result.single():
                print("  向量索引已存在，跳过")
                return

            session.run("""
                CREATE INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
            """)
            # Neo4j 5.x 向量索引需要用 db.index.vector.createNodes
            # 但先尝试简单方式
        print("  向量索引创建完成")
    except Exception as e:
        print(f"  [WARN] 向量索引创建失败: {e}")
        print("  将使用向量手动计算相似度")


def import_entities(driver, entities: pd.DataFrame, embeddings: dict):
    """批量导入实体节点。"""
    print(f"[4/5] 导入实体 ({len(entities)} 个)...")

    # 准备批量数据
    records = []
    for _, row in entities.iterrows():
        record = {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "type": str(row.get("type", "")),
            "description": str(row.get("description", "")),
            "frequency": int(row.get("frequency", 0)),
            "degree": int(row.get("degree", 0)),
            "text_unit_ids": [str(x) for x in row.get("text_unit_ids", [])],
        }
        # 添加 embedding（如果存在）
        entity_id = str(row["id"])
        if entity_id in embeddings:
            emb = embeddings[entity_id]
            # 转为 Python list of float（neo4j 需要）
            record["embedding"] = [float(x) for x in emb]
        records.append(record)

    # 分批导入
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS record
                MERGE (e:Entity {id: record.id})
                SET e.title = record.title,
                    e.type = record.type,
                    e.description = record.description,
                    e.frequency = record.frequency,
                    e.degree = record.degree,
                    e.text_unit_ids = record.text_unit_ids,
                    e.embedding = record.embedding
            """, batch=batch)

            done = min(i + BATCH_SIZE, len(records))
            print(f"  进度: {done}/{len(records)}", end="\r")

    print(f"  实体导入完成 OK")


def import_relationships(driver, relationships: pd.DataFrame):
    """批量导入关系。"""
    print(f"[5/5] 导入关系 ({len(relationships)} 条)...")

    # 构建 entity title -> id 的映射（关系用 title 做 source/target）
    # 注意：parquet 中 relationship 的 source/target 是 entity title
    records = []
    for _, row in relationships.iterrows():
        records.append({
            "id": str(row["id"]),
            "source_title": str(row["source"]),
            "target_title": str(row["target"]),
            "description": str(row.get("description", "")),
            "weight": float(row.get("weight", 1.0)),
            "combined_degree": int(row.get("combined_degree", 0)),
            "text_unit_ids": [str(x) for x in row.get("text_unit_ids", [])],
        })

    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS record
                MATCH (source:Entity {title: record.source_title})
                MATCH (target:Entity {title: record.target_title})
                MERGE (source)-[r:RELATES_TO {id: record.id}]->(target)
                SET r.description = record.description,
                    r.weight = record.weight,
                    r.combined_degree = record.combined_degree,
                    r.text_unit_ids = record.text_unit_ids
            """, batch=batch)

            done = min(i + BATCH_SIZE, len(records))
            print(f"  进度: {done}/{len(records)}", end="\r")

    print(f"  关系导入完成 OK")


def import_text_units(driver, text_units: pd.DataFrame):
    """批量导入文本块。"""
    print(f"导入文本块 ({len(text_units)} 个)...")

    records = []
    for _, row in text_units.iterrows():
        records.append({
            "id": str(row["id"]),
            "text": str(row.get("text", "")),
            "document_id": str(row.get("document_id", "")),
            "n_tokens": int(row.get("n_tokens", 0)),
            "entity_ids": [str(x) for x in row.get("entity_ids", [])],
            "relationship_ids": [str(x) for x in row.get("relationship_ids", [])],
        })

    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS record
                MERGE (t:TextUnit {id: record.id})
                SET t.text = record.text,
                    t.document_id = record.document_id,
                    t.n_tokens = record.n_tokens,
                    t.entity_ids = record.entity_ids,
                    t.relationship_ids = record.relationship_ids
            """, batch=batch)

    print(f"  文本块导入完成 OK")


def import_documents(driver, documents: pd.DataFrame):
    """批量导入文档。"""
    print(f"导入文档 ({len(documents)} 个)...")

    records = []
    for _, row in documents.iterrows():
        records.append({
            "id": str(row["id"]),
            "title": str(row.get("title", "")),
            "text": str(row.get("text", "")),
        })

    with driver.session() as session:
        session.run("""
            UNWIND $records AS record
            MERGE (d:Document {id: record.id})
            SET d.title = record.title,
                d.text = record.text
        """, records=records)

    print(f"  文档导入完成 OK")


def create_edges_to_text_units(driver, entities: pd.DataFrame, text_units: pd.DataFrame):
    """创建 Entity -> TextUnit 的 MENTIONED_IN 关系。"""
    print("创建 Entity-TextUnit 关联...")

    # 从 entities 的 text_unit_ids 提取关联
    edges = []
    for _, row in entities.iterrows():
        entity_id = str(row["id"])
        for tu_id in row.get("text_unit_ids", []):
            edges.append({"entity_id": entity_id, "text_unit_id": str(tu_id)})

    with driver.session() as session:
        for i in range(0, len(edges), BATCH_SIZE):
            batch = edges[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS edge
                MATCH (e:Entity {id: edge.entity_id})
                MATCH (t:TextUnit {id: edge.text_unit_id})
                MERGE (e)-[:MENTIONED_IN]->(t)
            """, batch=batch)

    print(f"  Entity-TextUnit 关联完成 ({len(edges)} 条) OK")


def create_edges_to_documents(driver, text_units: pd.DataFrame):
    """创建 TextUnit -> Document 的 FROM_DOCUMENT 关系。"""
    print("创建 TextUnit-Document 关联...")

    edges = []
    for _, row in text_units.iterrows():
        tu_id = str(row["id"])
        doc_id = str(row.get("document_id", ""))
        if doc_id:
            edges.append({"text_unit_id": tu_id, "document_id": doc_id})

    with driver.session() as session:
        for i in range(0, len(edges), BATCH_SIZE):
            batch = edges[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS edge
                MATCH (t:TextUnit {id: edge.text_unit_id})
                MATCH (d:Document {id: edge.document_id})
                MERGE (t)-[:FROM_DOCUMENT]->(d)
            """, batch=batch)

    print(f"  TextUnit-Document 关联完成 ({len(edges)} 条) OK")


def create_vector_index_proper(driver, has_embeddings: bool):
    """使用正确的 Neo4j 5.x API 创建向量索引。"""
    if not has_embeddings:
        print("  无 embedding 数据，跳过向量索引创建")
        return

    print("创建向量索引...")
    with driver.session() as session:
        try:
            # Neo4j 5.x 创建向量索引的正确方式
            session.run("""
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            print("  向量索引创建完成（需要等几秒生效）")
        except Exception as e:
            print(f"  [WARN] 向量索引创建失败: {e}")


def print_stats(driver):
    """打印导入统计信息。"""
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label")
        print("\n节点统计:")
        for record in result:
            print(f"  {record['label']}: {record['count']}")

        result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type")
        print("关系统计:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")


def main():
    print("=" * 60)
    print("GraphRAG → Neo4j 数据导入")
    print("=" * 60)

    # 连接 Neo4j
    print(f"\n连接 Neo4j: {NEO4J_URI}")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("连接成功 OK\n")
    except Exception as e:
        print(f"\n[ERROR] 无法连接 Neo4j: {e}")
        print("请先运行 start_neo4j.py 启动容器")
        sys.exit(1)

    try:
        # 加载数据
        data = load_data()
        has_embeddings = len(data["embeddings"]) > 0

        # 清空已有数据（可选）
        print("\n清空已有数据...")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("  已清空 OK")

        # 创建约束
        create_constraints(driver)

        # 导入数据
        import_entities(driver, data["entities"], data["embeddings"])
        import_relationships(driver, data["relationships"])
        import_text_units(driver, data["text_units"])
        import_documents(driver, data["documents"])

        # 创建边
        create_edges_to_text_units(driver, data["entities"], data["text_units"])
        create_edges_to_documents(driver, data["text_units"])

        # 创建向量索引
        create_vector_index_proper(driver, has_embeddings)

        # 打印统计
        print_stats(driver)

        print("\n" + "=" * 60)
        print("导入完成！")
        print("=" * 60)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
