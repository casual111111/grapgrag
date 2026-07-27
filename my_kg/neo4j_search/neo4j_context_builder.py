"""基于 Neo4j 的 GraphRAG LocalContextBuilder 实现。

支持：
    - 向量检索入口实体（利用 Neo4j 向量索引）
    - 可配置跳数的多跳图扩展（Cypher）
    - 返回实体、关系、文档（兼容 GraphRAG 的 ContextBuilderResult）

用法：
    # 独立使用
    builder = Neo4jLocalContextBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4j_test",
        text_embedder=your_embedder,
    )
    result = builder.build_context("用户注册流程", num_hops=2, top_k_entities=10)

    # 集成到 GraphRAG LocalSearch
    from graphrag.query.structured_search.local_search.search import LocalSearch
    search = LocalSearch(
        model=chat_model,
        context_builder=builder,
    )
    result = await search.search("用户注册流程")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

# 尝试导入 GraphRAG 的接口，如果不可用则使用兼容的 fallback
try:
    from graphrag.query.context_builder.builders import LocalContextBuilder
    from graphrag.query.context_builder.builders import ContextBuilderResult
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    # Fallback 定义
    @dataclass
    class ContextBuilderResult:
        context_chunks: str | list[str]
        context_records: dict[str, Any] = None
        llm_calls: int = 0
        prompt_tokens: int = 0
        output_tokens: int = 0

    class LocalContextBuilder:
        """Fallback base class when GraphRAG is not available."""
        def build_context(self, query: str, **kwargs) -> ContextBuilderResult:
            raise NotImplementedError


logger = logging.getLogger(__name__)


class Neo4jLocalContextBuilder(LocalContextBuilder):
    """基于 Neo4j 的 LocalContextBuilder，支持多跳图检索。"""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j_test",
        text_embedder: Any = None,  # GraphRAG 的 LLMEmbedding 或任何有 embed() 方法的对象
        tokenizer: Any = None,
        embedding_dimension: int = 1024,
    ):
        """
        初始化 Neo4j context builder。

        Args:
            neo4j_uri: Neo4j Bolt URI
            neo4j_user: 用户名
            neo4j_password: 密码
            text_embedder: embedding 模型（需要有 embed() 或 embed_batch() 方法）
            tokenizer: 分词器（用于计算 token 数）
            embedding_dimension: 向量维度（默认 1024）
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.text_embedder = text_embedder
        self.tokenizer = tokenizer
        self.embedding_dimension = embedding_dimension

        # 验证连接
        try:
            self.driver.verify_connectivity()
            logger.info(f"Neo4j 连接成功: {neo4j_uri}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise

    def close(self):
        """关闭 Neo4j 连接。"""
        self.driver.close()

    def build_context(
        self,
        query: str,
        conversation_history: Any = None,
        *,
        # === 和 local search 对齐的核心参数 ===
        top_k_entities: int = 10,          # 向量检索入口实体数
        top_k_relationships: int = 18,     # 关系预算（= top_k × seed_count）
        max_context_tokens: int = 24000,   # 上下文 token 上限
        # === 各部分 token 占比 ===
        entity_relationship_prop: float = 0.35,  # 实体+关系合计占比
        text_unit_prop: float = 0.65,            # 文本块原文占比
        # === Neo4j 特有参数 ===
        num_hops: int = 2,                 # 多跳扩展跳数
        min_relationship_weight: float = 0.0,
        **kwargs,
    ) -> ContextBuilderResult:
        """
        构建检索上下文。

        Args:
            query: 查询文本
            conversation_history: 对话历史（暂未使用）
            num_hops: 多跳扩展的跳数（默认 2）
            top_k_entities: 向量检索返回的入口实体数量（默认 10）
            max_entities_after_expand: 多跳扩展后保留的最大实体数（默认 30）
            top_k_relationships: 每个实体最多保留的关系数（默认 10）
            max_context_tokens: 上下文最大 token 数（默认 24000）
            min_relationship_weight: 关系权重过滤阈值（默认 0，不过滤）
            include_documents: 是否包含关联文档（默认 True）

        Returns:
            ContextBuilderResult，包含:
                - context_chunks: 格式化后的文本
                - context_records: dict with "entities", "relationships", "sources" DataFrames
        """
        # Step 1: 对 query 做向量编码
        query_vector = self._embed_query(query)

        # Step 2: 用向量相似度在 Neo4j 中检索 top-k 入口实体
        seed_entities = self._vector_search_entities(query_vector, top_k_entities)

        if not seed_entities:
            logger.warning("未找到匹配的入口实体")
            return ContextBuilderResult(
                context_chunks="未找到相关信息",
                context_records={"entities": pd.DataFrame(), "relationships": pd.DataFrame(), "sources": pd.DataFrame()},
            )

        seed_entity_ids = [e["id"] for e in seed_entities]

        # Step 3: 多跳图扩展（获取所有可达的实体和关系）
        all_entities, all_relationships = self._multi_hop_expand(
            seed_entity_ids, num_hops, min_relationship_weight
        )

        # Step 4: 对实体按相关性排序（种子实体优先，然后按 degree 排序）
        # 不截断 — 所有扩展到的实体都保留，让 LLM 自己判断
        seed_id_set = set(seed_entity_ids)
        for e in all_entities:
            e["_is_seed"] = 1 if e["id"] in seed_id_set else 0
        all_entities.sort(key=lambda e: (e["_is_seed"], e.get("degree", 0), e.get("frequency", 0)), reverse=True)
        entities = all_entities  # 保留全部

        # Step 5: 过滤关系 — 只保留两端都在选中实体中的，按权重排，预算控制
        relationships = self._filter_relationships(
            all_relationships, entities, top_k_per_entity=top_k_relationships
        )

        # Step 6: 获取关联的文本块（text units），不是完整文档
        # 参考 local search：从实体和关系的 text_unit_ids 收集，去重后返回文本块原文
        text_units = self._get_related_text_units(entities, relationships)

        # 清理内部字段
        for e in entities:
            e.pop("_is_seed", None)

        # Step 7: 格式化输出（按比例分配 token 预算）
        context_chunks, truncation_info = self._format_context(
            entities, relationships, text_units,
            max_context_tokens,
            entity_relationship_prop, text_unit_prop,
        )
        context_records = self._build_records(entities, relationships, text_units)
        # 添加截断信息到 context_records
        context_records["_truncation_info"] = truncation_info

        return ContextBuilderResult(
            context_chunks=context_chunks,
            context_records=context_records,
        )

    def _embed_query(self, query: str) -> list[float]:
        """获取 query 的 embedding 向量。
        使用 GraphRAG 的 LLMEmbedding 接口: text_embedder.embedding(input=[query]).first_embedding
        """
        if self.text_embedder is None:
            raise ValueError("未配置 text_embedder，无法进行向量编码。请通过 --use-embedding 运行。")

        # GraphRAG 的 LLMEmbedding 接口
        response = self.text_embedder.embedding(input=[query])
        return response.first_embedding

    def _vector_search_entities(self, query_vector: list[float], top_k: int) -> list[dict]:
        """使用 Neo4j 向量索引检索最相关的实体。"""
        with self.driver.session() as session:
            try:
                # 使用向量索引查询
                result = session.run("""
                    CALL db.index.vector.queryNodes('entity_embedding', $top_k, $query_vector)
                    YIELD node AS entity, score
                    RETURN entity.id AS id,
                           entity.title AS title,
                           entity.type AS type,
                           entity.description AS description,
                           entity.frequency AS frequency,
                           entity.degree AS degree,
                           score AS score
                """, top_k=top_k, query_vector=query_vector)

                entities = [dict(record) for record in result]
                return entities

            except Exception as e:
                # 如果向量索引不可用，退回到关键词匹配
                logger.warning(f"向量检索失败，退回关键词匹配: {e}")
                return self._keyword_search_entities(session, top_k)

    def _keyword_search_entities(self, session, top_k: int) -> list[dict]:
        """关键词搜索实体（fallback）——返回 degree 最高的实体。"""
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.id AS id,
                   e.title AS title,
                   e.type AS type,
                   e.description AS description,
                   e.frequency AS frequency,
                   e.degree AS degree,
                   1.0 AS score
            ORDER BY e.degree DESC
            LIMIT $top_k
        """, top_k=top_k)
        return [dict(record) for record in result]

    def _keyword_search(self, session, query: str, top_k: int) -> list[dict]:
        """根据 query 文本做关键词匹配的实体检索。"""
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.title) CONTAINS toLower($search_text)
               OR toLower(e.description) CONTAINS toLower($search_text)
            RETURN e.id AS id,
                   e.title AS title,
                   e.type AS type,
                   e.description AS description,
                   e.frequency AS frequency,
                   e.degree AS degree,
                   1.0 AS score
            ORDER BY e.degree DESC
            LIMIT $top_k
        """, search_text=query, top_k=top_k)
        entities = [dict(record) for record in result]

        # 如果关键词匹配不到足够结果，补充 degree 最高的
        if len(entities) < top_k:
            existing_ids = {e["id"] for e in entities}
            result2 = session.run("""
                MATCH (e:Entity)
                WHERE NOT e.id IN $existing_ids
                RETURN e.id AS id,
                       e.title AS title,
                       e.type AS type,
                       e.description AS description,
                       e.frequency AS frequency,
                       e.degree AS degree,
                       1.0 AS score
                ORDER BY e.degree DESC
                LIMIT $top_k
            """, existing_ids=list(existing_ids), top_k=top_k - len(entities))
            entities.extend(dict(record) for record in result2)

        return entities

    def _multi_hop_expand(
        self,
        seed_entity_ids: list[str],
        num_hops: int,
        min_weight: float,
    ) -> tuple[list[dict], list[dict]]:
        """
        从种子实体出发，进行多跳图扩展。

        Returns:
            (entities, relationships)
        """
        with self.driver.session() as session:
            # 构建多跳查询
            # 使用可变长度路径 (1..num_hops)
            query = """
                MATCH (seed:Entity)
                WHERE seed.id IN $seed_ids

                MATCH path = (seed)-[*1..$num_hops]-(neighbor:Entity)

                // 收集路径上的所有实体
                WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS all_nodes

                // 展开实体列表
                UNWIND all_nodes AS entity
                WITH DISTINCT entity

                // 获取实体信息
                WITH entity
                RETURN entity.id AS id,
                       entity.title AS title,
                       entity.type AS type,
                       entity.description AS description,
                       entity.frequency AS frequency,
                       entity.degree AS degree,
                       entity.text_unit_ids AS text_unit_ids
            """

            # 注意：Neo4j Cypher 中可变长度路径的参数不能直接用 $param
            # 需要用字符串拼接或者 APOC
            # 这里用简单方式：直接拼接到查询中
            query = query.replace("$num_hops", str(num_hops))

            result = session.run(query, seed_ids=seed_entity_ids)
            entities = [dict(record) for record in result]

            # 获取这些实体之间的关系
            entity_ids = [e["id"] for e in entities]
            rel_query = """
                MATCH (e1:Entity)-[r:RELATES_TO]-(e2:Entity)
                WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
                AND r.weight >= $min_weight
                RETURN DISTINCT
                    r.id AS id,
                    e1.title AS source,
                    e2.title AS target,
                    r.description AS description,
                    r.weight AS weight,
                    r.combined_degree AS combined_degree,
                    r.text_unit_ids AS text_unit_ids
            """
            result = session.run(rel_query, entity_ids=entity_ids, min_weight=min_weight)
            relationships = [dict(record) for record in result]

            return entities, relationships

    def _filter_relationships(
        self,
        all_relationships: list[dict],
        kept_entities: list[dict],
        top_k_per_entity: int,
    ) -> list[dict]:
        """过滤关系，参考 GraphRAG local search 的 _filter_relationships 逻辑。

        top_k_per_entity: 每个实体最多保留的关系数（和 local search 含义一致）。
        全局预算 = top_k_per_entity * len(seed_entities)。

        1. 只保留两端都在 kept_entities 中的关系（用 title 匹配）
        2. 区分 in-network（两端都是种子实体）和 out-network
        3. in-network 全部保留
        4. out-network 按权重排序，全局预算 = top_k_per_entity * seed_count
        """
        # 建立 title 集合
        kept_titles = {e["title"] for e in kept_entities}
        seed_titles = {e["title"] for e in kept_entities if e.get("_is_seed")}

        # Step 1: 只保留两端都在选中实体中的关系
        filtered = [
            r for r in all_relationships
            if r.get("source", "") in kept_titles and r.get("target", "") in kept_titles
        ]

        # Step 2: 区分 in-network 和 out-network
        in_network = []
        out_network = []
        for r in filtered:
            src = r.get("source", "")
            tgt = r.get("target", "")
            if src in seed_titles and tgt in seed_titles:
                in_network.append(r)
            else:
                out_network.append(r)

        # Step 3: in-network 按权重降序排序，每个实体最多 top_k_per_entity 条
        in_network.sort(key=lambda r: r.get("weight", 0), reverse=True)
        in_entity_count: dict[str, int] = {}
        in_budgeted = []
        for r in in_network:
            source = r.get("source", "")
            target = r.get("target", "")
            src_count = in_entity_count.get(source, 0)
            tgt_count = in_entity_count.get(target, 0)
            if src_count < top_k_per_entity or tgt_count < top_k_per_entity:
                in_budgeted.append(r)
                in_entity_count[source] = src_count + 1
                in_entity_count[target] = tgt_count + 1

        # Step 4: out-network 每个实体最多 top_k_per_entity/4 条关系
        out_budget = max(top_k_per_entity // 4, 1)
        entity_rel_count: dict[str, int] = {}
        budget_result = []

        for r in out_network:
            source = r.get("source", "")
            target = r.get("target", "")
            src_count = entity_rel_count.get(source, 0)
            tgt_count = entity_rel_count.get(target, 0)

            # 只要 source 或 target 任一方还有预算，就保留
            if src_count < out_budget or tgt_count < out_budget:
                budget_result.append(r)
                entity_rel_count[source] = src_count + 1
                entity_rel_count[target] = tgt_count + 1

        return in_budgeted + budget_result

    def _get_related_text_units(
        self,
        entities: list[dict],
        relationships: list[dict],
    ) -> list[dict]:
        """获取与实体/关系关联的文本块（text units），不是完整文档。
        参考 local search 的做法：从实体和关系的 text_unit_ids 收集，去重后返回文本块原文。
        """
        # 收集所有 text_unit_ids
        text_unit_ids = set()
        for e in entities:
            for tu_id in (e.get("text_unit_ids") or []):
                text_unit_ids.add(str(tu_id))
        for r in relationships:
            for tu_id in (r.get("text_unit_ids") or []):
                text_unit_ids.add(str(tu_id))

        if not text_unit_ids:
            return []

        with self.driver.session() as session:
            # 直接返回文本块本身（包含原文），不跳到文档
            result = session.run("""
                MATCH (t:TextUnit)
                WHERE t.id IN $text_unit_ids
                RETURN
                    t.id AS id,
                    t.text AS text,
                    t.document_id AS document_id,
                    t.n_tokens AS n_tokens,
                    t.entity_ids AS entity_ids
            """, text_unit_ids=list(text_unit_ids))

            text_units = [dict(record) for record in result]

        # 按关联实体数量排序（关联越多的 text unit 越重要），参考 local search
        # 计算每个 text unit 被多少个选中实体/关系引用
        tu_ref_count: dict[str, int] = {}
        for tu in text_units:
            tu_id = tu["id"]
            count = 0
            for e in entities:
                if tu_id in [str(x) for x in (e.get("text_unit_ids") or [])]:
                    count += 1
            for r in relationships:
                if tu_id in [str(x) for x in (r.get("text_unit_ids") or [])]:
                    count += 1
            tu_ref_count[tu_id] = count

        text_units.sort(key=lambda tu: tu_ref_count.get(tu["id"], 0), reverse=True)
        return text_units

    def _format_context(
        self,
        entities: list[dict],
        relationships: list[dict],
        text_units: list[dict],
        max_tokens: int,
        entity_relationship_prop: float = 0.35,
        text_unit_prop: float = 0.65,
    ) -> tuple[str, dict]:
        """将检索结果格式化为文本上下文，按比例分配 token 预算。
        实体+关系共享 entity_relationship_prop 的预算，文本块占 text_unit_prop。

        Returns:
            (formatted_text, truncation_info):
                - formatted_text: 格式化后的上下文文本
                - truncation_info: 截断信息字典，包含哪些部分被截断了
        """
        # 各部分的 token 预算（中文约 2 char/token）
        er_budget = int(max_tokens * entity_relationship_prop)
        tu_budget = int(max_tokens * text_unit_prop)

        # 实体和关系各占 er_budget 的一半
        entity_budget_chars = er_budget  # 先用总字符预算
        rel_budget_chars = er_budget

        sections = []

        # 截断信息
        truncation_info = {
            "entity_truncated": False,
            "entity_included": 0,
            "entity_total": len(entities),
            "relationship_truncated": False,
            "relationship_included": 0,
            "relationship_total": len(relationships),
            "text_unit_truncated": False,
            "text_unit_included": 0,
            "text_unit_total": len(text_units),
        }

        # 1. 实体部分（按预算截断）
        entity_text = ""
        if entities:
            entity_text = "## 相关实体\n\n"
            entity_chars = 0
            entity_max_chars = er_budget  # 实体先用 er_budget 的一半
            entity_count = 0
            for e in entities:
                block = f"**{e.get('title', 'N/A')}**"
                if e.get("type"):
                    block = f"({e['type']}){block}"
                block += "\n"
                if e.get("description"):
                    desc = str(e["description"])
                    block += f"{desc}\n\n"
                if entity_chars + len(block) > entity_max_chars:
                    break
                entity_text += block
                entity_chars += len(block)
                entity_count += 1
            sections.append(entity_text)
            truncation_info["entity_included"] = entity_count
            if entity_count < len(entities):
                truncation_info["entity_truncated"] = True

        # 2. 关系部分（用剩余的 er 预算）
        if relationships:
            # 计算实体实际用了多少，剩余给关系
            entity_used_chars = len(entity_text)
            rel_max_chars = max(er_budget * 2 - entity_used_chars, 0)

            rel_text = "## 相关关系\n\n"
            rel_chars = 0
            rel_count = 0
            for r in relationships:
                block = f"**{r.get('source', '?')}** -> **{r.get('target', '?')}**\n"
                if r.get("description"):
                    block += f"{r['description']}\n"
                block += "\n"
                if rel_chars + len(block) > rel_max_chars:
                    break
                rel_text += block
                rel_chars += len(block)
                rel_count += 1
            sections.append(rel_text)
            truncation_info["relationship_included"] = rel_count
            if rel_count < len(relationships):
                truncation_info["relationship_truncated"] = True

        # 3. 文本块部分（按预算截断）
        if text_units:
            tu_text = "## 相关原文\n\n"
            tu_chars = 0
            tu_max_chars = tu_budget * 2
            tu_count = 0
            for tu in text_units:
                text = str(tu.get("text", ""))
                if not text:
                    continue
                block = f"{text}\n\n"
                if tu_chars + len(block) > tu_max_chars:
                    break
                tu_text += block
                tu_chars += len(block)
                tu_count += 1
            sections.append(tu_text)
            truncation_info["text_unit_included"] = tu_count
            if tu_count < len(text_units):
                truncation_info["text_unit_truncated"] = True

        result = "\n".join(sections)

        # 超限警告
        estimated_tokens = len(result) // 2
        if estimated_tokens > max_tokens:
            logger.warning(
                f"上下文超限！估算 {estimated_tokens} tokens > 上限 {max_tokens} tokens。"
                f"建议增大 max_context_tokens 或减小 num_hops / top_k_entities。"
            )

        return result, truncation_info

    def _build_records(
        self,
        entities: list[dict],
        relationships: list[dict],
        text_units: list[dict],
    ) -> dict[str, pd.DataFrame]:
        """构建结构化记录（DataFrame 格式，兼容 GraphRAG）。"""
        records = {}

        if entities:
            records["entities"] = pd.DataFrame(entities)
        else:
            records["entities"] = pd.DataFrame(columns=["id", "title", "type", "description", "frequency", "degree"])

        if relationships:
            records["relationships"] = pd.DataFrame(relationships)
        else:
            records["relationships"] = pd.DataFrame(columns=["id", "source", "target", "description", "weight"])

        if text_units:
            records["sources"] = pd.DataFrame(text_units)
        else:
            records["sources"] = pd.DataFrame(columns=["id", "text", "document_id", "n_tokens"])

        return records


def quick_search(
    query: str,
    num_hops: int = 2,
    top_k_entities: int = 10,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "neo4j_test",
) -> ContextBuilderResult:
    """
    快速搜索（不需要 embedding，使用关键词匹配）。

    用法：
        result = quick_search("用户注册")
        print(result.context_chunks)
    """
    builder = Neo4jLocalContextBuilder(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        text_embedder=None,  # 不使用向量检索
    )

    try:
        # 使用关键词搜索 fallback
        with builder.driver.session() as session:
            # 直接用关键词匹配实体
            cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.title) CONTAINS toLower($search_text)
                   OR toLower(e.description) CONTAINS toLower($search_text)
                RETURN e.id AS id,
                       e.title AS title,
                       e.type AS type,
                       e.description AS description,
                       e.frequency AS frequency,
                       e.degree AS degree,
                       1.0 AS score
                ORDER BY e.degree DESC
                LIMIT $top_k
            """
            result = session.run(cypher, search_text=query, top_k=top_k_entities)

            seed_entities = [dict(record) for record in result]

        if not seed_entities:
            return ContextBuilderResult(
                context_chunks=f"未找到与 '{query}' 相关的实体",
                context_records={},
            )

        seed_ids = [e["id"] for e in seed_entities]

        # 多跳扩展
        entities, relationships = builder._multi_hop_expand(seed_ids, num_hops, 0.0)

        # 获取文档
        documents = builder._get_related_documents(entities, relationships)

        # 格式化
        context_chunks = builder._format_context(entities, relationships, documents, 24000)
        context_records = builder._build_records(entities, relationships, documents)

        return ContextBuilderResult(
            context_chunks=context_chunks,
            context_records=context_records,
        )

    finally:
        builder.close()
