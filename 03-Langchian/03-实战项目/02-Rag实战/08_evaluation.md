# RAG系统评估与优化

评估和优化是构建高质量RAG系统的关键步骤。本章将介绍评估RAG系统性能的方法，以及提高系统效果的优化技术。

## 评估指标

### 检索评估指标

1. **准确率(Precision)**：检索结果中相关文档的比例
2. **召回率(Recall)**：成功检索到的相关文档占所有相关文档的比例
3. **F1分数**：准确率和召回率的调和平均
4. **平均精度(Mean Average Precision, MAP)**：各召回水平下准确率的平均值
5. **归一化折现累积增益(NDCG)**：考虑排序位置的评估指标

### 生成评估指标

1. **事实准确性(Factual Accuracy)**：生成内容与事实的一致程度
2. **上下文相关性(Context Relevance)**：回答与提供上下文的相关程度
3. **忠实度(Faithfulness)**：回答对检索内容的忠实程度
4. **连贯性(Coherence)**：回答的逻辑连贯性和流畅性
5. **信息完整性(Completeness)**：回答是否包含所有必要信息

### 端到端评估指标

1. **问题回答准确率**：回答问题的准确程度
2. **延迟时间**：系统响应时间
3. **用户满意度**：基于用户反馈的指标
4. **幻觉率(Hallucination Rate)**：生成不存在于检索内容中的错误信息的比例

## 评估方法

### 使用RAGAS框架

[RAGAS](https://github.com/explodinggradients/ragas)是一个专门评估RAG系统的框架：

```python
# 使用RAGAS评估RAG系统

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate

def evaluate_with_ragas(questions, answers, contexts, ground_truths=None):
    """
    使用RAGAS评估RAG系统
    
    Args:
        questions: 问题列表
        answers: 生成的回答列表
        contexts: 用于生成回答的上下文列表
        ground_truths: 标准答案列表(可选)
    
    Returns:
        评估结果
    """
    # 准备评估指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    
    # 如果有标准答案，添加上下文召回率指标
    if ground_truths:
        metrics.append(context_recall)
    
    # 组织数据
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    if ground_truths:
        eval_data["ground_truths"] = ground_truths
    
    # 执行评估
    result = evaluate(eval_data, metrics)
    return result
```

### 自定义评估方法

自定义评估可以更针对性地评估特定应用场景：

```python
# 自定义评估实现

import numpy as np
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

class CustomRAGEvaluator:
    """自定义RAG评估器"""
    
    def __init__(self, llm: BaseLLM):
        """初始化评估器"""
        self.llm = llm
        
        # 事实准确性评估提示
        self.factual_accuracy_prompt = PromptTemplate.from_template(
            """评估以下回答的事实准确性。
            
问题: {question}
回答: {answer}
参考上下文: {context}

1. 判断回答中的每个事实陈述是否在参考上下文中有明确支持
2. 标记没有在参考上下文中支持的陈述
3. 评估整体事实准确性得分(1-10分)

请返回以下JSON格式:
{{
  "supported_facts": ["列出所有被支持的事实陈述"],
  "unsupported_facts": ["列出所有未被支持的事实陈述"],
  "factual_accuracy_score": 分数,
  "reasoning": "你的推理"
}}
"""
        )
        
        # 回答完整性评估提示
        self.completeness_prompt = PromptTemplate.from_template(
            """评估以下回答的完整性。
            
问题: {question}
回答: {answer}
参考上下文: {context}

1. 识别问题要求回答的所有关键点
2. 检查回答是否涵盖了这些关键点
3. 评估整体完整性得分(1-10分)

请返回以下JSON格式:
{{
  "key_points": ["应该包含的关键点列表"],
  "covered_points": ["回答中涵盖的关键点"],
  "missing_points": ["回答中缺失的关键点"],
  "completeness_score": 分数,
  "reasoning": "你的推理"
}}
"""
        )
    
    def evaluate_factual_accuracy(self, question, answer, context):
        """评估事实准确性"""
        prompt_input = {
            "question": question,
            "answer": answer,
            "context": context
        }
        
        evaluation = self.llm.invoke(self.factual_accuracy_prompt.format(**prompt_input))
        
        # 解析结果(实际应用中需要更健壮的解析)
        try:
            import json
            result = json.loads(evaluation)
            return result
        except:
            return {
                "factual_accuracy_score": 0,
                "error": "解析失败",
                "raw_evaluation": evaluation
            }
    
    def evaluate_completeness(self, question, answer, context):
        """评估回答完整性"""
        prompt_input = {
            "question": question,
            "answer": answer,
            "context": context
        }
        
        evaluation = self.llm.invoke(self.completeness_prompt.format(**prompt_input))
        
        # 解析结果
        try:
            import json
            result = json.loads(evaluation)
            return result
        except:
            return {
                "completeness_score": 0,
                "error": "解析失败",
                "raw_evaluation": evaluation
            }
    
    def comprehensive_evaluation(self, questions, answers, contexts):
        """综合评估多个指标"""
        results = []
        
        for q, a, c in zip(questions, answers, contexts):
            # 评估各项指标
            factual_result = self.evaluate_factual_accuracy(q, a, c)
            completeness_result = self.evaluate_completeness(q, a, c)
            
            # 汇总结果
            result = {
                "question": q,
                "factual_accuracy": factual_result.get("factual_accuracy_score", 0),
                "completeness": completeness_result.get("completeness_score", 0),
                "unsupported_facts": factual_result.get("unsupported_facts", []),
                "missing_points": completeness_result.get("missing_points", [])
            }
            
            # 计算综合分数
            result["overall_score"] = (
                result["factual_accuracy"] + result["completeness"]
            ) / 2.0
            
            results.append(result)
        
        # 计算平均分数
        avg_factual = np.mean([r["factual_accuracy"] for r in results])
        avg_completeness = np.mean([r["completeness"] for r in results])
        avg_overall = np.mean([r["overall_score"] for r in results])
        
        summary = {
            "individual_results": results,
            "average_scores": {
                "factual_accuracy": avg_factual,
                "completeness": avg_completeness,
                "overall": avg_overall
            }
        }
        
        return summary
```

## 人工评估

对于关键应用，人工评估仍然非常重要：

```python
# 人工评估表单示例

def create_human_evaluation_form(question, answer, context):
    """创建人工评估表单"""
    form = f"""
# RAG系统人工评估表

**问题:** {question}

**系统回答:** {answer}

**参考上下文:** {context}

## 评估项目

请为以下各项评分(1-5分)：

1. 事实准确性: [ ]
   - 回答中的信息是否与提供的上下文一致?
   - 回答是否避免了"幻觉"(虚构的信息)?

2. 回答相关性: [ ]
   - 回答是否直接针对问题?
   - 是否包含不必要的无关信息?

3. 完整性: [ ]
   - 回答是否涵盖了问题所需的全部信息?
   - 是否遗漏了上下文中的重要信息?

4. 清晰度和连贯性: [ ]
   - 回答是否结构清晰、逻辑连贯?
   - 是否易于理解?

5. 整体质量: [ ]
   - 总体而言，这个回答的质量如何?

## 额外反馈

优点:
[填写]

缺点:
[填写]

改进建议:
[填写]

评估人: ________________  日期: ________________
"""
    return form
```

## 性能优化

### 检索器优化

```python
# 检索器优化方法

def optimize_retriever(retriever, test_questions, ground_truth_docs, k_values=[1, 3, 5, 10]):
    """
    优化检索器参数
    
    Args:
        retriever: 要优化的检索器
        test_questions: 测试问题列表
        ground_truth_docs: 每个问题的标准相关文档列表
        k_values: 要测试的k值列表
    
    Returns:
        最佳k值和性能指标
    """
    results = {}
    
    for k in k_values:
        # 设置检索器参数
        retriever.search_kwargs["k"] = k
        
        # 评估指标
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # 对每个问题进行评估
        for q_idx, question in enumerate(test_questions):
            # 获取检索结果
            retrieved_docs = retriever.get_relevant_documents(question)
            retrieved_ids = [hash(doc.page_content) for doc in retrieved_docs]
            
            # 获取标准答案
            ground_truth_ids = [hash(doc.page_content) for doc in ground_truth_docs[q_idx]]
            
            # 计算指标
            tp = len(set(retrieved_ids) & set(ground_truth_ids))
            precision = tp / len(retrieved_ids) if retrieved_ids else 0
            recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # 计算平均指标
        results[k] = {
            "avg_precision": np.mean(precision_scores),
            "avg_recall": np.mean(recall_scores),
            "avg_f1": np.mean(f1_scores)
        }
    
    # 找出F1分数最高的k值
    best_k = max(results.keys(), key=lambda k: results[k]["avg_f1"])
    
    return {
        "best_k": best_k,
        "best_performance": results[best_k],
        "all_results": results
    }
```

### 提示模板优化

```python
# 提示模板优化

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

def optimize_prompt_template(llm: BaseLLM, retriever, test_questions, test_contexts, ground_truths):
    """
    优化提示模板
    
    Args:
        llm: 大语言模型
        retriever: 检索器
        test_questions: 测试问题列表
        test_contexts: 测试上下文列表
        ground_truths: 标准答案列表
    
    Returns:
        最佳提示模板和性能
    """
    # 定义要测试的提示模板
    templates = {
        "basic": """使用以下上下文来回答问题。如果你不知道答案，就说你不知道。

上下文:
{context}

问题: {question}

回答:""",
        
        "detailed": """你是一个人工智能助手，专门根据提供的上下文回答问题。使用以下参考上下文回答用户的问题。
如果上下文中没有足够的信息，请明确说明你不知道，而不要编造答案。
确保你的回答是直接基于提供的上下文的，并且准确无误。

上下文信息:
{context}

用户问题: {question}

助手回答:""",
        
        "step_by_step": """根据提供的上下文信息，一步一步地思考来回答问题。

上下文:
{context}

问题: {question}

请按以下步骤回答:
1. 确定问题要求什么信息
2. 从上下文中查找相关信息
3. 组织这些信息，形成连贯的回答
4. 如果上下文中没有答案，请明确说明

回答:"""
    }
    
    results = {}
    
    # 自定义评估器
    evaluator = CustomRAGEvaluator(llm)
    
    for name, template in templates.items():
        prompt = PromptTemplate.from_template(template)
        
        answers = []
        
        # 为每个测试问题生成回答
        for q, c in zip(test_questions, test_contexts):
            # 构建输入
            prompt_input = {
                "context": c,
                "question": q
            }
            
            # 生成回答
            answer = llm.invoke(prompt.format(**prompt_input))
            answers.append(answer)
        
        # 评估结果
        evaluation = evaluator.comprehensive_evaluation(
            test_questions, answers, test_contexts
        )
        
        results[name] = {
            "template": template,
            "evaluation": evaluation,
            "overall_score": evaluation["average_scores"]["overall"]
        }
    
    # 找出总体分数最高的模板
    best_template = max(results.keys(), key=lambda k: results[k]["overall_score"])
    
    return {
        "best_template": best_template,
        "best_template_content": results[best_template]["template"],
        "best_performance": results[best_template]["overall_score"],
        "all_results": results
    }
```

## 优化案例研究

以下是一个完整的RAG系统优化案例：

```python
# RAG系统优化案例

def optimize_rag_system():
    """RAG系统优化流程"""
    # 1. 准备评估数据
    print("准备评估数据...")
    # 这里应该加载实际的评估数据
    questions = ["问题1", "问题2", "问题3"]
    ground_truths = ["标准答案1", "标准答案2", "标准答案3"]
    
    # 2. 建立基线性能
    print("建立基线性能...")
    # 创建基础RAG系统
    embedding_model = EmbeddingFactory.create_embeddings(...)
    vector_store = VectorStoreFactory.load_vector_store(...)
    retriever = VectorStoreRetriever(...)
    llm = LLMFactory.create_llm(...)
    base_rag = BasicRAG(retriever, llm)
    
    # 评估基线性能
    base_answers = []
    for q in questions:
        answer = base_rag.query(q)
        base_answers.append(answer)
    
    base_evaluation = evaluate_with_ragas(
        questions, base_answers, 
        [base_rag.get_retrieved_documents(q) for q in questions],
        ground_truths
    )
    print(f"基线性能: {base_evaluation}")
    
    # 3. 优化检索器
    print("优化检索器...")
    retriever_optimization = optimize_retriever(
        retriever,
        questions,
        [ground_truth_docs for _ in questions],  # 这里应该是实际的标准文档
        k_values=[2, 3, 4, 5, 8]
    )
    
    best_k = retriever_optimization["best_k"]
    print(f"最佳k值: {best_k}")
    
    # 更新检索器
    retriever.search_kwargs["k"] = best_k
    
    # 4. 优化提示模板
    print("优化提示模板...")
    prompt_optimization = optimize_prompt_template(
        llm,
        retriever,
        questions,
        ["\n\n".join([d.page_content for d in retriever.get_relevant_documents(q)]) 
         for q in questions],
        ground_truths
    )
    
    best_template = prompt_optimization["best_template_content"]
    print(f"最佳提示模板: {best_template}")
    
    # 5. 集成优化结果
    print("集成优化结果...")
    optimized_rag = BasicRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=best_template
    )
    
    # 6. 评估优化后的性能
    print("评估优化后的性能...")
    optimized_answers = []
    for q in questions:
        answer = optimized_rag.query(q)
        optimized_answers.append(answer)
    
    optimized_evaluation = evaluate_with_ragas(
        questions, optimized_answers, 
        [optimized_rag.get_retrieved_documents(q) for q in questions],
        ground_truths
    )
    
    print(f"优化后性能: {optimized_evaluation}")
    
    # 7. 比较改进
    print("\n性能改进对比:")
    for metric in base_evaluation.keys():
        if metric in optimized_evaluation:
            improvement = optimized_evaluation[metric] - base_evaluation[metric]
            print(f"{metric}: {base_evaluation[metric]:.4f} -> {optimized_evaluation[metric]:.4f} (改进: {improvement:.4f})")
    
    return {
        "baseline": base_evaluation,
        "optimized": optimized_evaluation,
        "retriever_optimization": retriever_optimization,
        "prompt_optimization": prompt_optimization,
        "optimized_rag": optimized_rag
    }
```

## 常见性能问题及解决方案

### 1. 检索相关性低

**症状**：检索到的文档与查询关联度低

**解决方案**：
- 尝试不同的嵌入模型
- 优化块大小和重叠
- 实现查询重写或扩展
- 考虑混合检索策略

### 2. 上下文利用率低

**症状**：LLM没有充分利用提供的上下文信息

**解决方案**：
- 改进提示工程
- 实现上下文压缩
- 考虑将上下文分组或重新排序
- 使用多步骤生成过程

### 3. 生成幻觉

**症状**：生成的答案包含上下文中不存在的信息

**解决方案**：
- 调整LLM温度参数
- 使用更明确的提示约束
- 实现答案验证机制
- 考虑自问自答(Self-ask)策略

### 4. 长延迟时间

**症状**：系统响应时间过长

**解决方案**：
- 优化向量数据库索引
- 实现结果缓存
- 并行执行子查询
- 考虑使用更轻量级的模型

## 下一步

在本章中，我们学习了如何评估和优化RAG系统的性能。通过系统性地评估各个组件并应用有针对性的优化技术，可以显著提高RAG系统的效果。在下一章中，我们将探讨如何部署RAG系统到生产环境，包括构建Web API、开发用户界面以及处理实际部署中的挑战。 