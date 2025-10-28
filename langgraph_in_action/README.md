# LangGraph 实战

本仓库包含交互式 Jupyter Notebook，用于演示 LangGraph 概念和实现。

## 关于本书

《LangGraph 实战》是一本深入探索 AI 智能体（Agent）技术的实用指南。在人工智能快速发展的今天，我们正站在 Agent 元年的前夜，而 LangGraph 作为构建复杂、自主决策 Agent 应用的强大框架，正是释放 LLM 潜力的关键。

本书系统梳理了 LangGraph 的最新技术，涵盖从基础概念到高级实践的完整学习路径。无论您是 AI 开发新手，还是希望提升技能的资深开发者，都能通过本书掌握 Agent 应用开发的核心技能。

### 主要特色

- **完整的技术体系**: 从 Agent 基础概念到 LangGraph 框架的深入讲解
- **丰富的实战案例**: 通过具体代码示例演示各种 Agent 应用场景
- **渐进式学习路径**: 由浅入深的内容组织，适合不同基础的读者
- **国内环境适配**: 使用硅基流动一站式大模型云服务平台 SiliconCloud 平台和国产大模型，依托开箱即用的大模型 API 服务，便于国内开发者学习实践

### 📱 加入飞书群

欢迎您扫描下方二维码加入我们的技术讨论群：

![](https://raw.githubusercontent.com/webup/langgraph-up-react/main/static/feishu.jpg)

## 环境配置

### 1. 使用 UV 创建 Python 虚拟环境

首先安装 [uv](https://docs.astral.sh/uv/)（如果尚未安装）:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建新的虚拟环境：

```bash
uv venv
```

### 2. 安装依赖

安装所有项目依赖（包含开发环境和 Jupyter 内核）：

```bash
uv sync --dev
```

### 3. 配置环境变量

复制环境变量示例文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件并添加您的 API 密钥：

- **SiliconCloud API Key**: 从 [SiliconCloud](https://cloud.siliconflow.cn/me/models) 获取 API 密钥并设置 `OPENAI_API_KEY`
- **LangSmith API Key**: 从 [LangSmith](https://smith.langchain.com/) 获取 API 密钥并设置 `LANGCHAIN_API_KEY`

### 4. IDE 配置

#### VS Code
1. 安装 **Jupyter** 扩展
2. 打开 tutorials 目录中的任意 `.ipynb` 文件
3. 当提示时，选择来自 UV 虚拟环境的 Python 解释器
4. 内核会自动检测 UV 创建的虚拟环境

#### 其他 IDE
确保您的 IDE 的 Jupyter 扩展能够访问 UV 虚拟环境中的 Python 解释器。

## 运行教程

1. 激活虚拟环境（如果未自动激活）：
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # 或
   .venv\Scripts\activate     # Windows
   ```

2. 在 IDE 中打开任意教程 notebook（`chapter2.ipynb`、`chapter3.ipynb`、`chapter4.ipynb` 等）

3. 选择对应虚拟环境的 Python 内核

4. 运行单元格开始探索 LangGraph 概念！

## 可用教程

- `chapter2.ipynb` - LangGraph 框架概览
- `chapter3.ipynb` - LangGraph 核心元素和子图
- `chapter4.ipynb` - AI 智能体的交互体验（流式处理、持久化、人机在环）
- `chapter5.ipynb` - AI 智能体的记忆系统
- `chapter6.ipynb` - 构建 AI 智能体的其它 API 选项
- `chapter7.ipynb` - AI 智能体系统的架构与范式

## 学习建议

### 快速入门路径
1. 首先通读前两章内容，建立对 Agent 技术和 LangGraph 框架的整体认知
2. 重点学习实战案例章节（特别是第 3-7 章），跟随案例代码进行实践操作
3. 根据需求深入研究高级主题章节，如架构设计、部署和未来趋势

### 不同背景读者指南
- **初学者**: 建议从 chapter2.ipynb 开始，快速上手构建简单的 Agent 应用
- **有经验的开发者**: 按顺序学习所有章节，深入理解 LangGraph 的高级功能
- **有 LangChain 基础**: 可以快速浏览基础概念，重点关注 LangGraph 的独特优势

## 故障排除

- **找不到内核**: 确保已激活 UV 虚拟环境并使用 `uv sync` 安装了依赖
- **API 错误**: 检查 `.env` 文件中的 API 密钥设置是否正确
- **导入错误**: 再次运行 `uv sync` 确保所有依赖都已安装

## 参考图书

![](https://raw.githubusercontent.com/webup/langgraph-up-react/main/static/langgraph-poster.jpg)
