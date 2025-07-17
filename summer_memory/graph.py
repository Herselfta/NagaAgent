import json
from py2neo import Graph, Node, Relationship
import logging
import sys
import os

# 添加项目根目录到路径，以便导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from config import GRAG_NEO4J_URI, GRAG_NEO4J_USER, GRAG_NEO4J_PASSWORD, GRAG_NEO4J_DATABASE, GRAG_ENABLED
    NEO4J_URI = GRAG_NEO4J_URI
    NEO4J_USER = GRAG_NEO4J_USER
    NEO4J_PASSWORD = GRAG_NEO4J_PASSWORD
    NEO4J_DATABASE = GRAG_NEO4J_DATABASE
except ImportError:
    # 如果无法导入config，使用环境变量作为备选
    NEO4J_URI = os.getenv("GRAG_NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("GRAG_NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("GRAG_NEO4J_PASSWORD", "hkm27iar")
    NEO4J_DATABASE = os.getenv("GRAG_NEO4J_DATABASE", "testnaga")

logger = logging.getLogger(__name__)
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DATABASE) if GRAG_ENABLED else None
TRIPLES_FILE = "triples.json"


def load_triples():
    try:
        with open(TRIPLES_FILE, 'r', encoding='utf-8') as f:
            return set(tuple(t) for t in json.load(f))
    except FileNotFoundError:
        return set()


def save_triples(triples):
    with open(TRIPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(triples), f, ensure_ascii=False, indent=2)


def store_triples(new_triples):
    all_triples = load_triples()
    all_triples.update(new_triples)  # 集合自动去重

    # 持久化到文件
    save_triples(all_triples)

    # 同步更新Neo4j图谱数据库（仅在GRAG_ENABLED时）
    if graph is not None:
        for head, rel, tail in new_triples:
            if not head or not tail:
                logger.warning(f"跳过无效三元组，head或tail为空: {(head, rel, tail)}")
                continue
            h_node = Node("Entity", name=head)
            t_node = Node("Entity", name=tail)
            r = Relationship(h_node, rel, t_node)
            graph.merge(h_node, "Entity", "name")
            graph.merge(t_node, "Entity", "name")
            graph.merge(r)


def get_all_triples():
    return load_triples()


def query_graph_by_keywords(keywords):
    results = []
    if graph is not None:
        for kw in keywords:
            query = f"""
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE e1.name CONTAINS '{kw}' OR e2.name CONTAINS '{kw}' OR type(r) CONTAINS '{kw}'
            RETURN e1.name, type(r), e2.name
            LIMIT 5
            """
            res = graph.run(query).data()
            for record in res:
                results.append((record['e1.name'], record['type(r)'], record['e2.name']))
    return results
