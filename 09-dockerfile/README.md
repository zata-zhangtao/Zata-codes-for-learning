# FastAPI Docker 部署演示项目

## 项目简介

本项目演示如何创建一个简单的 FastAPI 后端应用，并使用 Docker 进行容器化部署。通过本项目，你将学习到：

- FastAPI 框架的基本使用
- Docker 容器化技术
- Dockerfile 的编写和优化
- 容器化应用的部署流程

## 技术栈

- **Python 3.11+**: 编程语言
- **FastAPI**: 现代、高性能的 Web 框架
- **Uvicorn**: ASGI 服务器
- **Docker**: 容器化平台
- **Docker Compose**: 多容器应用编排（可选）

## 项目结构

```
09-dockerfile/
├── README.md              # 项目文档
├── main.py               # FastAPI 应用主文件
├── requirements.txt      # Python 依赖包
├── Dockerfile           # Docker 镜像构建文件
├── docker-compose.yml   # Docker Compose 配置文件（可选）
└── .dockerignore       # Docker 忽略文件
```

## 快速开始

### 1. 本地开发环境

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 运行应用
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 查看应用

### 2. Docker 部署

#### 构建 Docker 镜像
```bash
docker build -t fastapi-demo .
```

#### 运行 Docker 容器
```bash
docker run -d -p 8000:8000 --name fastapi-container fastapi-demo
```

#### 使用 Docker Compose（推荐）
```bash
docker-compose up -d
```

## API 接口

### 基础接口

- `GET /` - 根路径，返回欢迎信息
- `GET /health` - 健康检查接口
- `GET /docs` - Swagger API 文档
- `GET /redoc` - ReDoc API 文档

### 示例接口

- `GET /items/{item_id}` - 获取指定 ID 的项目
- `POST /items/` - 创建新项目
- `PUT /items/{item_id}` - 更新指定项目
- `DELETE /items/{item_id}` - 删除指定项目

## Docker 配置说明

### Dockerfile 特性

- 使用多阶段构建优化镜像大小
- 基于 Python 3.9-slim 镜像
- 非 root 用户运行提高安全性
- 优化的层缓存策略
- 健康检查配置

### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `PORT` | 8000 | 应用监听端口 |
| `HOST` | 0.0.0.0 | 应用监听地址 |
| `WORKERS` | 1 | Uvicorn 工作进程数 |

## 部署选项

### 1. 单容器部署
适合简单的开发和测试环境

### 2. Docker Compose 部署
适合需要数据库、Redis 等依赖服务的场景

### 3. Kubernetes 部署
适合生产环境的大规模部署

## 性能优化

- 使用 Uvicorn 的多工作进程模式
- 启用 Gzip 压缩
- 配置适当的超时设置
- 使用健康检查确保服务可用性

## 监控和日志

- 结构化日志输出
- 健康检查端点
- 性能指标收集
- 错误追踪和报告

## 开发建议

1. **代码结构**: 使用合适的项目结构分离关注点
2. **依赖管理**: 固定依赖版本确保可重现构建
3. **安全性**: 不在镜像中包含敏感信息
4. **测试**: 编写单元测试和集成测试
5. **文档**: 保持 API 文档更新

## 故障排除

### 常见问题

1. **端口冲突**: 确保端口 8000 未被占用
2. **权限问题**: 检查 Docker 守护进程权限
3. **内存不足**: 调整容器内存限制
4. **网络问题**: 检查防火墙和网络配置

### 调试命令

```bash
# 查看容器日志
docker logs fastapi-container

# 进入容器调试
docker exec -it fastapi-container /bin/bash

# 检查容器状态
docker ps -a

# 查看镜像信息
docker images
```

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 项目讨论区

---

**注意**: 本项目仅用于学习和演示目的，生产环境使用请根据实际需求进行相应的安全配置和性能优化。