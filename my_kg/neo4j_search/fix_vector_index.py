"""修复 Neo4j 向量索引。"""
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j_test"))

with driver.session() as s:
    # 删除旧的错误索引
    s.run("DROP INDEX entity_embedding IF EXISTS")
    print("Dropped old index")

    # 创建正确的向量索引
    query = """
        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: 'cosine'
        }}
    """
    s.run(query)
    print("Created vector index")

    # 验证
    result = s.run("SHOW INDEXES WHERE name = 'entity_embedding'")
    for r in result:
        print(dict(r))

# 等待索引生效
import time
print("\nWaiting for index to come online...")
time.sleep(5)

with driver.session() as s:
    result = s.run("SHOW INDEXES WHERE name = 'entity_embedding'")
    for r in result:
        d = dict(r)
        print(f"  state: {d.get('state', 'unknown')}")

driver.close()
print("Done")
