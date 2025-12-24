"""
Java技术知识图谱导入脚本
作者：zjy

该脚本向Neo4j图数据库导入Java核心技术知识，
包括Spring Boot、MyBatis、Hibernate等主流框架的基础信息。

运行方式：
1. 启动Neo4j服务
2. 打开浏览器访问 http://localhost:7474
3. 在Neo4j Browser中执行此脚本
"""

// 清空现有数据（可选）
MATCH (n) DETACH DELETE n;

// 1. 创建Java技术节点
CREATE (spring:Technology {
    name: 'Spring Boot',
    description: 'Spring Boot是由Pivotal团队提供的全新框架，其设计目的是用来简化新Spring应用的初始搭建以及开发过程。该框架使用了特定的方式来进行配置，从而使开发人员不再需要定义样板化的配置。',
    category: 'Web框架',
    version: '3.1.0',
    created_date: datetime()
});

CREATE (mybatis:Technology {
    name: 'MyBatis',
    description: 'MyBatis是支持定制化SQL、存储过程以及高级映射的优秀持久层框架。MyBatis避免了几乎所有的JDBC代码和手动设置参数以及获取结果集。',
    category: '持久层框架',
    version: '3.0.0',
    created_date: datetime()
});

CREATE (hibernate:Technology {
    name: 'Hibernate',
    description: 'Hibernate是一个开放源代码的对象关系映射框架，它对JDBC进行了非常轻量级的对象封装，它将POJO与数据库表建立映射关系，是一个全自动的orm框架。',
    category: 'ORM框架',
    version: '6.0.0',
    created_date: datetime()
});

CREATE (jpa:Technology {
    name: 'JPA',
    description: 'Java Persistence API是Sun公司发布的Java EE标准规范，用于对象持久化。它提供了完整的ORM标准，底层实现可以用Hibernate、TopLink等。',
    category: 'ORM标准',
    version: '3.1',
    created_date: datetime()
});

CREATE (springmvc:Technology {
    name: 'Spring MVC',
    description: 'Spring MVC是Spring Framework的一部分，是一个基于Java的实现了MVC设计模式的请求驱动类型的轻量级Web框架。',
    category: 'Web框架',
    version: '6.0.0',
    created_date: datetime()
});

CREATE (jwt:Technology {
    name: 'JWT',
    description: 'JSON Web Token (JWT)是一个开放标准，它定义了一种紧凑的、自包含的方式，用于在各方之间安全地传输信息。',
    category: '安全认证',
    version: '0.11.5',
    created_date: datetime()
});

CREATE (redis:Technology {
    name: 'Redis',
    description: 'Redis是一个开源的内存中数据结构存储，用作数据库、缓存和消息代理。',
    category: '缓存数据库',
    version: '7.0',
    created_date: datetime()
});

CREATE (maven:Technology {
    name: 'Maven',
    description: 'Maven是Apache组织开发的一个项目管理工具，主要用于Java项目的构建、依赖管理和项目信息管理。',
    category: '构建工具',
    version: '3.9.0',
    created_date: datetime()
});

// 2. 创建使用场景节点
CREATE (microservice:UseCase {
    name: '微服务架构',
    description: '将单一应用程序开发为一组小型服务的方法，每个服务运行在自己的进程中，服务间采用轻量级通信机制，可独立部署和扩展。'
});

CREATE (webdev:UseCase {
    name: 'Web开发',
    description: '使用Java技术栈开发Web应用程序，包括前端、后端和数据库的全栈开发。'
});

CREATE (data_access:UseCase {
    name: '数据访问',
    description: '实现应用程序与数据库之间的数据交互，包括CRUD操作、事务管理等。'
});

CREATE (api_dev:UseCase {
    name: 'API开发',
    description: '开发RESTful API或GraphQL接口，为前端或第三方应用提供数据服务。'
});

CREATE (caching:UseCase {
    name: '缓存加速',
    description: '使用缓存技术提升应用性能，减少数据库访问压力。'
});

// 3. 创建技术领域节点
CREATE (web:Domain {
    name: 'Web开发',
    description: '使用Java技术开发Web应用程序的领域'
});

CREATE (persistence:Domain {
    name: '数据持久化',
    description: '实现数据在持久化存储和内存之间的转换'
});

CREATE (security:Domain {
    name: '安全认证',
    description: '处理用户身份验证、授权和数据安全'
});

CREATE (devops:Domain {
    name: 'DevOps',
    description: '开发运维一体化，包括构建、部署、监控等'
});

// 4. 创建最佳实践节点
CREATE (autoconfig:BestPractice {
    name: '自动配置',
    description: 'Spring Boot的核心特性，通过starter依赖和自动配置简化项目搭建。'
});

CREATE (restful:BestPractice {
    name: 'RESTful设计',
    description: '遵循REST架构风格的API设计原则，包括资源定位、HTTP方法使用等。'
});

CREATE (orm_best:BestPractice {
    name: 'ORM最佳实践',
    description: '合理使用ORM框架，避免N+1查询问题，合理配置缓存，及时同步实体状态。'
});

// 5. 创建使用限制节点
CREATE (complexity:Limitation {
    name: '学习曲线陡峭',
    description: 'Spring生态系统庞大，初学者需要掌握大量概念和配置。'
});

CREATE (performance:Limitation {
    name: '性能开销',
    description: '框架层的抽象和动态代理会带来一定的性能开销。'
});

CREATE (vendor:Limitation {
    name: '厂商依赖',
    description: '某些特定框架可能绑定到特定厂商或服务。'
});

// 6. 创建性能分析节点
CREATE (spring_perf:Performance {
    name: 'Spring Boot性能',
    analysis: 'Spring Boot通过自动配置和嵌入式服务器减少了启动时间，Spring Boot 2.x启动时间约2-3秒。合理的Bean配置和AOP使用可以保证良好的运行性能。'
});

CREATE (mybatis_perf:Performance {
    name: 'MyBatis性能',
    analysis: 'MyBatis性能优于Hibernate，因为它允许开发者编写原生SQL，精确控制查询。与Hibernate相比，MyBatis的查询性能可提升20-30%。'
});

CREATE (redis_perf:Performance {
    name: 'Redis性能',
    analysis: 'Redis是内存数据库，QPS可达10万+，是MySQL的10-100倍。适合存储热点数据、会话信息等。'
});

// 7. 创建版本信息节点
CREATE (java8:Version {
    name: 'Java 8',
    description: '引入Lambda表达式、Stream API、Optional等重要特性',
    release_date: '2014-03-18'
});

CREATE (java11:Version {
    name: 'Java 11',
    description: 'LTS版本，移除Java EE模块，引入ZGC',
    release_date: '2018-09-25'
});

CREATE (java17:Version {
    name: 'Java 17',
    description: 'LTS版本，引入sealed类、Pattern Matching等特性',
    release_date: '2021-09-14'
});

CREATE (java21:Version {
    name: 'Java 21',
    description: '最新LTS版本，引入虚拟线程、record patterns等',
    release_date: '2023-09-19'
});

// 8. 创建教程信息节点
CREATE (spring_tutorial:Tutorial {
    name: 'Spring Boot入门教程',
    code_example: '@SpringBootApplication\npublic class Application {\n    public static void main(String[] args) {\n        SpringApplication.run(Application.class, args);\n    }\n}',
    config_method: 'application.yml配置示例：\nserver:\n  port: 8080\nspring:\n  application:\n    name: myapp',
    notes: '注意：Spring Boot会自动配置很多组件，使用前需确认是否需要自定义配置。'
});

CREATE (mybatis_tutorial:Tutorial {
    name: 'MyBatis集成教程',
    code_example: '@Mapper\npublic interface UserMapper {\n    @Select("SELECT * FROM users WHERE id = #{id}")\n    User findById(Long id);\n}',
    config_method: 'mybatis-config.xml配置：\n<configuration>\n  <typeAliases>\n    <package name="com.example.model"/>\n  </typeAliases>\n</configuration>',
    notes: '使用@Mapper注解或配置扫描包来注册Mapper接口。'
});

CREATE (jwt_tutorial:Tutorial {
    name: 'JWT认证教程',
    code_example: 'String token = Jwts.builder()\n    .setSubject(user.getUsername())\n    .setExpiration(new Date(System.currentTimeMillis() + 86400000))\n    .signWith(SecretKeySpec)\n    .compact();',
    config_method: 'application.yml配置JWT密钥：\napp:\n  jwt:\n    secret: mySecretKey\n    expiration: 86400000',
    notes: '密钥长度至少32位，过期时间根据业务需求设置。'
});

// 9. 创建相关技术关系
CREATE (spring)-[:COMBINED_WITH]->(springmvc);
CREATE (spring)-[:COMBINED_WITH]->(jpa);
CREATE (spring)-[:COMBINED_WITH]->(maven);
CREATE (spring)-[:COMBINED_WITH]->(jwt);
CREATE (spring)-[:COMBINED_WITH]->(redis);

CREATE (mybatis)-[:COMBINED_WITH]->(mysql);
CREATE (mybatis)-[:COMBINED_WITH]->(spring);
CREATE (mybatis)-[:COMBINED_WITH]->(maven);

CREATE (jpa)-[:ALTERNATIVE_TO]->(mybatis);
CREATE (mybatis)-[:ALTERNATIVE_TO]->(hibernate);

// 10. 创建使用场景关系
CREATE (spring)-[:USED_FOR]->(microservice);
CREATE (spring)-[:USED_FOR]->(webdev);
CREATE (springmvc)-[:USED_FOR]->(api_dev);
CREATE (mybatis)-[:USED_FOR]->(data_access);
CREATE (redis)-[:USED_FOR]->(caching);
CREATE (jwt)-[:USED_FOR]->(webdev);

// 11. 创建属于领域关系
CREATE (spring)-[:BELONGS_TO]->(web);
CREATE (springmvc)-[:BELONGS_TO]->(web);
CREATE (mybatis)-[:BELONGS_TO]->(persistence);
CREATE (hibernate)-[:BELONGS_TO]->(persistence);
CREATE (jpa)-[:BELONGS_TO]->(persistence);
CREATE (jwt)-[:BELONGS_TO]->(security);
CREATE (maven)-[:BELONGS_TO]->(devops);

// 12. 创建最佳实践关系
CREATE (spring)-[:HAS_BEST_PRACTICE]->(autoconfig);
CREATE (springmvc)-[:HAS_BEST_PRACTICE]->(restful);
CREATE (mybatis)-[:HAS_BEST_PRACTICE]->(orm_best);

// 13. 创建限制关系
CREATE (spring)-[:HAS_LIMITATION]->(complexity);
CREATE (hibernate)-[:HAS_LIMITATION]->(performance);
CREATE (mybatis)-[:HAS_LIMITATION]->(vendor);

// 14. 创建性能关系
CREATE (spring)-[:HAS_PERFORMANCE]->(spring_perf);
CREATE (mybatis)-[:HAS_PERFORMANCE]->(mybatis_perf);
CREATE (redis)-[:HAS_PERFORMANCE]->(redis_perf);

// 15. 创建版本支持关系
CREATE (spring)-[:SUPPORTS_VERSION]->(java8);
CREATE (spring)-[:SUPPORTS_VERSION]->(java11);
CREATE (spring)-[:SUPPORTS_VERSION]->(java17);
CREATE (spring)-[:SUPPORTS_VERSION]->(java21);

CREATE (mybatis)-[:SUPPORTS_VERSION]->(java8);
CREATE (mybatis)-[:SUPPORTS_VERSION]->(java11);
CREATE (mybatis)-[:SUPPORTS_VERSION]->(java17);

// 16. 创建教程关系
CREATE (spring)-[:HAS_TUTORIAL]->(spring_tutorial);
CREATE (mybatis)-[:HAS_TUTORIAL]->(mybatis_tutorial);
CREATE (jwt)-[:HAS_TUTORIAL]->(jwt_tutorial);

// 显示导入结果
RETURN 'Java技术知识图谱导入完成!' AS message, count(*) AS node_count;