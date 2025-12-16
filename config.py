"""
Java技术知识图谱查询模板配置
作者：zjy
创建时间：2024年

该模块定义了Java文档问答系统中用于查询Neo4j图数据库的模板。
每个模板包含槽位（slots）、自然语言问题、Cypher查询语句和答案格式。
"""

# Java技术知识图谱查询模板
# 模板结构说明：
# - slots: 需要从用户问题中提取的实体槽位
# - question: 匹配的自然语言问题模式
# - cypher: Neo4j图数据库查询语句
# - answer: 答案格式化模板
GRAPH_TEMPLATE = {
    # 1. 类/接口定义查询
    'definition': {
        'slots': ['class_or_interface'],
        'question': '什么叫%class_or_interface%? / %class_or_interface%是什么？',
        'cypher': "MATCH (t:Technology) WHERE t.name='%class_or_interface%' RETURN t.description AS RES",
        'answer': '【%class_or_interface%】的定义：%RES%',
    },

    # 2. 使用场景查询
    'use_case': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%一般用在什么场景？/ 什么时候用%class_or_interface%？',
        'cypher': "MATCH (t:Technology)-[:USED_FOR]->(u:UseCase) WHERE t.name='%class_or_interface%' RETURN u.description AS RES",
        'answer': '【%class_or_interface%】的使用场景：%RES%',
    },

    # 3. 方法特性查询
    'method': {
        'slots': ['method_name'],
        'question': '%method_name%方法有什么作用？/ %method_name%的用法？',
        'cypher': "MATCH (m:Method) WHERE m.name='%method_name%' RETURN m.description AS RES",
        'answer': '【%method_name%】方法的作用：%RES%',
    },

    # 4. 反向查询：根据方法找类
    'class_by_method': {
        'slots': ['method_name'],
        'question': '%method_name%是在哪个类中的方法？',
        'cypher': "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name='%method_name%' RETURN c.name AS RES",
        'answer': '【%method_name%】方法位于类：%RES%',
    },

    # 5. 使用教程查询（包含代码示例、配置方法、注意事项）
    'tutorial': {
        'slots': ['class_or_interface'],
        'question': '如何使用%class_or_interface%？/ %class_or_interface%怎么用？',
        'cypher': '''
            MATCH (t:Technology)-[:HAS_TUTORIAL]->(tu:Tutorial)
            WHERE t.name = '%class_or_interface%'
            WITH COLLECT(DISTINCT tu.code_example) AS codeExamples,
                COLLECT(DISTINCT tu.config_method) AS configMethods,
                COLLECT(DISTINCT tu.notes) AS notes
            RETURN SUBSTRING(REDUCE(s = '', x IN codeExamples | s + '、' + x), 1) AS RES1,
                SUBSTRING(REDUCE(s = '', x IN configMethods | s + '、' + x), 1) AS RES2,
                SUBSTRING(REDUCE(s = '', x IN notes | s + '、' + x), 1) AS RES3
            ''',
        'answer': '【%class_or_interface%】的使用方法：\n代码示例：%RES1%\n配置方式：%RES2%\n注意事项：%RES3%',
    },

    # 6. 技术领域查询
    'domain': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%属于哪个技术领域？',
        'cypher': "MATCH (t:Technology)-[:BELONGS_TO]->(d:Domain) WHERE t.name='%class_or_interface%' RETURN d.name AS RES",
        'answer': '【%class_or_interface%】属于技术领域：%RES%',
    },

    # 7. 最佳实践查询
    'best_practice': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%的最佳实践是什么？',
        'cypher': "MATCH (t:Technology)-[:HAS_BEST_PRACTICE]->(bp:BestPractice) WHERE t.name='%class_or_interface%' RETURN bp.description AS RES",
        'answer': '【%class_or_interface%】的最佳实践：%RES%',
    },

    # 8. 使用限制查询
    'limitation': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%有什么限制？/ 什么情况下不适合用%class_or_interface%？',
        'cypher': "MATCH (t:Technology)-[:HAS_LIMITATION]->(l:Limitation) WHERE t.name='%class_or_interface%' RETURN l.description AS RES",
        'answer': '【%class_or_interface%】的使用限制：%RES%',
    },

    # 9. 性能分析查询
    'performance': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%的性能如何？',
        'cypher': "MATCH (t:Technology)-[:HAS_PERFORMANCE]->(p:Performance) WHERE t.name='%class_or_interface%' RETURN p.analysis AS RES",
        'answer': '【%class_or_interface%】的性能分析：%RES%',
    },

    # 10. 版本兼容性查询
    'version': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%支持哪些Java版本？',
        'cypher': "MATCH (t:Technology)-[:SUPPORTS_VERSION]->(v:Version) WHERE t.name='%class_or_interface%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(v.name) | s + '、' + x), 1) AS RES",
        'answer': '【%class_or_interface%】支持的Java版本：%RES%',
    },

    # 11. 相关技术查询
    'related_tech': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%和哪些技术经常一起使用？',
        'cypher': "MATCH (t1:Technology)-[:COMBINED_WITH]->(t2:Technology) WHERE t1.name='%class_or_interface%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(t2.name) | s + '、' + x), 1) AS RES",
        'answer': '【%class_or_interface%】常配合使用的技术：%RES%',
    },

    # 12. 替代方案查询
    'alternative': {
        'slots': ['class_or_interface'],
        'question': '%class_or_interface%的替代方案有哪些？',
        'cypher': "MATCH (t:Technology)-[:ALTERNATIVE_TO]->(alt:Technology) WHERE t.name='%class_or_interface%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(alt.name) | s + '、' + x), 1) AS RES",
        'answer': '【%class_or_interface%】的替代方案：%RES%',
    },
}
