from __future__ import annotations

import re


_SKILL_RULES = [
    (r"\bpython3\b|\bpython2\b|\bpy3\b", "Python"),
    (r"\bpython\b", "Python"),
    (r"\bnodejs\b|\bnode\.?js\b", "Node.js"),
    (r"\btypescript\b|\bts\b", "TypeScript"),
    (r"\bjavascript\b|\bjs\b", "JavaScript"),
    (r"\bgolang\b|\bgo lang\b", "Go"),
    (r"\bkubernetes\b|\bk8s\b", "Kubernetes"),
    (r"\bdocker\b", "Docker"),
    (r"\bmysql\b", "MySQL"),
    (r"\bpostgresql\b|\bpostgres\b", "PostgreSQL"),
    (r"\bredis\b", "Redis"),
    (r"\bmongodb\b", "MongoDB"),
    (r"\belasticsearch\b", "Elasticsearch"),
    (r"\bmq\b|\bmessage queue\b|\b消息队列\b", "MessageQueue"),
    (r"\bcelery\b", "Celery"),
    (r"\bnginx\b", "Nginx"),
    (r"\bflask\b", "Flask"),
    (r"\bfastapi\b", "FastAPI"),
    (r"\bdjango\b", "Django"),
    (r"\bgraphql\b", "GraphQL"),
    (r"\bgrpc\b", "gRPC"),
    (r"\brestful\b|\brest api\b|\brest\b", "RESTful API"),
    (r"\blangchain\b", "LangChain"),
    (r"\blanggraph\b", "LangGraph"),
    (r"\bllm\b", "LLM"),
    (r"\brag\b", "RAG"),
    (r"\b向量数据库\b|\bvector ?db\b", "VectorDB"),
    (r"\bchroma\b", "ChromaDB"),
    (r"\bpostgres\b", "PostgreSQL"),
    (r"\bgit\b", "Git"),
    (r"\bci/cd\b|\bci cd\b|\bjenkins\b", "CI/CD"),
    (r"\blinux\b", "Linux"),
    (r"\b微服务\b|\bmicroservice\b", "Microservices"),
    (r"\b高并发\b|\bhigh concurrency\b|\b并发\b", "HighConcurrency"),
    (r"\b分布式\b|\bdistributed\b", "DistributedSystems"),
    (r"\b缓存\b|\bcaching\b", "Caching"),
    (r"\b算法\b|\balgorithm\b", "Algorithms"),
    (r"\b数据结构\b|\bdata structure\b", "DataStructures"),
]


def normalize_skill(name: str) -> str:
    text = str(name).strip()
    lowered = text.lower()
    for pattern, canonical in _SKILL_RULES:
        if re.search(pattern, lowered):
            return canonical
    return text


def normalize_skills(skills) -> list[str]:
    result = []
    seen = set()
    for skill in skills or []:
        canonical = normalize_skill(skill)
        key = canonical.lower()
        if canonical and key not in seen:
            seen.add(key)
            result.append(canonical)
    return result
