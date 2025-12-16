"""
医疗知识图谱查询模板配置
作者：zjy
创建时间：2024年

该模块定义了医疗问诊机器人中用于查询Neo4j图数据库的模板。
每个模板包含槽位（slots）、自然语言问题、Cypher查询语句和答案格式。
"""

# 医疗知识图谱查询模板
# 模板结构说明：
# - slots: 需要从用户问题中提取的实体槽位
# - question: 匹配的自然语言问题模式
# - cypher: Neo4j图数据库查询语句
# - answer: 答案格式化模板
GRAPH_TEMPLATE = {
    # 1. 疾病定义查询
    'desc': {
        'slots': ['disease'],
        'question': '什么叫%disease%? / %disease%是一种什么病？',
        'cypher': "MATCH (n:Disease) WHERE n.name='%disease%' RETURN n.desc AS RES",
        'answer': '【%disease%】的定义：%RES%',
    },

    # 2. 病因查询
    'cause': {
        'slots': ['disease'],
        'question': '%disease%一般是由什么引起的？/ 什么会导致%disease%？',
        'cypher': "MATCH (n:Disease) WHERE n.name='%disease%' RETURN n.cause AS RES",
        'answer': '【%disease%】的病因：%RES%',
    },

    # 3. 疾病症状查询
    'disease_symptom': {
        'slots': ['disease'],
        'question': '%disease%会有哪些症状？/ %disease%有哪些临床表现？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_SYMPTOM]->(m) WHERE n.name='%disease%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(m.name) | s + '、' + x), 1) AS RES",
        'answer': '【%disease%】的症状：%RES%',
    },

    # 4. 症状反向查询（根据症状找疾病）
    'symptom': {
        'slots': ['symptom'],
        'question': '%symptom%可能是得了什么病？',
        'cypher': "MATCH (n)-[:DISEASE_SYMPTOM]->(m:Symptom) WHERE m.name='%symptom%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(n.name) | s + '、' + x), 1) AS RES",
        'answer': '可能出现【%symptom%】症状的疾病：%RES%',
    },

    # 5. 治疗方案查询（包含治疗方式、药物、推荐食物）
    'cure_way': {
        'slots': ['disease'],
        'question': '%disease%吃什么药好得快？/ %disease%怎么治？',
        'cypher': '''
            MATCH (n:Disease)-[:DISEASE_CUREWAY]->(m1),
                (n:Disease)-[:DISEASE_DRUG]->(m2),
                (n:Disease)-[:DISEASE_DO_EAT]->(m3)
            WHERE n.name = '%disease%'
            WITH COLLECT(DISTINCT m1.name) AS m1Names,
                COLLECT(DISTINCT m2.name) AS m2Names,
                COLLECT(DISTINCT m3.name) AS m3Names
            RETURN SUBSTRING(REDUCE(s = '', x IN m1Names | s + '、' + x), 1) AS RES1,
                SUBSTRING(REDUCE(s = '', x IN m2Names | s + '、' + x), 1) AS RES2,
                SUBSTRING(REDUCE(s = '', x IN m3Names | s + '、' + x), 1) AS RES3
            ''',
        'answer': '【%disease%】的治疗方法：%RES1%。\n可用药物：%RES2%。\n推荐食物：%RES3%',
    },

    # 6. 就诊科室查询
    'cure_department': {
        'slots': ['disease'],
        'question': '得了%disease%去医院挂什么科室的号？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_DEPARTMENT]->(m) WHERE n.name='%disease%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(m.name) | s + '、' + x), 1) AS RES",
        'answer': '【%disease%】的就诊科室：%RES%',
    },

    # 7. 疾病预防查询
    'prevent': {
        'slots': ['disease'],
        'question': '%disease%要怎么预防？',
        'cypher': "MATCH (n:Disease) WHERE n.name='%disease%' RETURN n.prevent AS RES",
        'answer': '【%disease%】的预防方法：%RES%',
    },

    # 8. 饮食禁忌查询
    'not_eat': {
        'slots': ['disease'],
        'question': '%disease%换着有什么禁忌？/ %disease%不能吃什么？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_NOT_EAT]->(m) WHERE n.name='%disease%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(m.name) | s + '、' + x), 1) AS RES",
        'answer': '【%disease%】的患者不能吃的食物：%RES%',
    },

    # 9. 检查项目查询
    'check': {
        'slots': ['disease'],
        'question': '%disease%要做哪些检查？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_CHECK]->(m) WHERE n.name='%disease%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(m.name) | s + '、' + x), 1) AS RES",
        'answer': '【%disease%】的检查项目：%RES%',
    },

    # 10. 治愈率查询
    'cured_prob': {
        'slots': ['disease'],
        'question': '%disease%能治好吗？/ %disease%治好的几率有多大？',
        'cypher': "MATCH (n:Disease) WHERE n.name='%disease%' RETURN n.cured_prob AS RES",
        'answer': '【%disease%】的治愈率：%RES%',
    },

    # 11. 并发症查询
    'acompany': {
        'slots': ['disease'],
        'question': '%disease%的并发症有哪些？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_ACOMPANY]->(m) WHERE n.name='%disease%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(m.name) | s + '、' + x), 1) AS RES",
        'answer': '【%disease%】的并发症：%RES%',
    },

    # 12. 药物适应症查询
    'indications': {
        'slots': ['drug'],
        'question': '%drug%能治那些病？',
        'cypher': "MATCH (n:Disease)-[:DISEASE_DRUG]->(m:Drug) WHERE m.name='%drug%' RETURN SUBSTRING(REDUCE(s = '', x IN COLLECT(n.name) | s + '、' + x), 1) AS RES",
        'answer': '【%drug%】能治疗的疾病有：%RES%',
    },
}