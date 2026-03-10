# 从零构建多Agent协作系统：DAG编排实战

> 原创技术文章 | V1-MultiAgent团队出品
> 阅读时间：15分钟

## 前言

在AI应用开发的浪潮中，单一Agent的能力已经无法满足复杂业务场景的需求。当多个Agent需要协同工作时，**如何优雅地管理它们之间的依赖关系和执行顺序**成为关键挑战。

本文将深入解析多Agent协作的核心技术——**DAG（有向无环图）编排**，并结合V1-MultiAgent框架的实战代码，带你从零构建一个生产级的多Agent协作系统。

---

## 一、为什么需要DAG编排？

### 1.1 单Agent的局限性

想象一个智能客服场景：
- **意图识别Agent**：分析用户问题
- **知识检索Agent**：查找相关文档
- **回答生成Agent**：组织最终回复

如果只是简单串行执行，一旦知识检索失败，整个流程就中断了。我们需要更灵活的控制机制。

### 1.2 多Agent协作的三种模式

```python
# 模式1：简单串行（Sequential）
[Agent A] → [Agent B] → [Agent C]

# 模式2：并行执行（Parallel）
     ┌→ [Agent B] →┐
[Agent A]          [Agent D]
     └→ [Agent C] →┘

# 模式3：复杂DAG（混合模式）
[Agent A] → [Agent B] → [Agent D]
     └→ [Agent C] →┘
```

**DAG编排的优势**：
- ✅ 清晰定义依赖关系
- ✅ 支持并行执行提升效率
- ✅ 灵活的错误处理和重试
- ✅ 可视化执行流程

---

## 二、DAG编排的核心概念

### 2.1 什么是DAG？

**DAG（Directed Acyclic Graph）** 即**有向无环图**：
- **有向**：边有方向，表示依赖关系
- **无环**：不存在循环依赖（A依赖B，B又依赖A）

### 2.2 关键术语

| 术语 | 说明 | 示例 |
|-----|------|------|
| **节点（Node）** | 图中的任务单元 | 一个Agent的执行 |
| **边（Edge）** | 节点间的依赖关系 | A → B 表示B依赖A |
| **入度（In-degree）** | 指向某节点的边数 | 表示前置依赖数量 |
| **拓扑排序** | 满足依赖的执行顺序 | A → B → C 或 A → C → B |

### 2.3 拓扑排序算法

DAG编排的核心是**拓扑排序**——找出一个满足所有依赖关系的执行顺序：

```python
def topological_sort(graph):
    """
    Kahn算法实现拓扑排序
    """
    # 1. 计算所有节点的入度
    in_degree = {node: 0 for node in graph.nodes}
    for node in graph.nodes:
        for neighbor in graph.get_neighbors(node):
            in_degree[neighbor] += 1
    
    # 2. 入度为0的节点入队（没有依赖，可以立即执行）
    queue = [node for node in graph.nodes if in_degree[node] == 0]
    result = []
    
    # 3. 依次处理
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        # 减少后继节点的入度
        for neighbor in graph.get_neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result
```

---

## 三、V1-MultiAgent的DAG实现

### 3.1 核心架构

```python
# orchestrator.py 核心结构
class Orchestrator:
    """DAG编排器"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
    
    async def execute(
        self,
        workflow: Workflow,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        initial_input: Any = None,
        aggregator: Optional[Callable] = None
    ) -> ExecutionResult:
        """执行工作流"""
        pass
```

### 3.2 工作流定义

```python
from v1_multiagent.core import Orchestrator, WorkflowBuilder, ExecutionMode

async def create_research_workflow():
    """创建研究助手工作流"""
    orchestrator = Orchestrator()
    
    # 创建Agent
    planner = SimpleAgent(name="Planner")
    researcher = SimpleAgent(name="Researcher")
    writer = SimpleAgent(name="Writer")
    
    await planner.init(response_template="计划: {input}")
    await researcher.init(response_template="研究结果: {input}")
    await writer.init(response_template="文章: {input}")
    
    # 构建DAG
    builder = WorkflowBuilder(orchestrator, name="ResearchAssistant")
    
    # 1. 规划阶段（起点）
    builder.add_task(planner, name="plan")
    
    # 2. 研究阶段（依赖plan）
    builder.add_task(researcher, name="research", depends_on=["plan"])
    
    # 3. 写作阶段（依赖research）
    builder.add_task(writer, name="write", depends_on=["research"])
    
    workflow = builder.build()
    return orchestrator, workflow
```

### 3.3 串行执行模式

```python
# 最简单的DAG执行
result = await orchestrator.execute(
    workflow,
    mode=ExecutionMode.SEQUENTIAL,
    initial_input="写一篇关于AI Agent的技术文章"
)

print(f"执行成功: {result.success}")
print(f"最终结果: {result.results['write']}")
```

### 3.4 并行执行模式

```python
# 并行工作流示例
builder = WorkflowBuilder(orchestrator, name="ParallelResearch")

# 起点
builder.add_task(coordinator, name="start")

# 三个并行的研究任务
builder.add_parallel(
    [researcher1, researcher2, researcher3],
    names=["r1", "r2", "r3"],
    depends_on=["start"]
)

# 汇总结果
builder.add_task(synthesizer, name="merge", depends_on=["r1", "r2", "r3"])

workflow = builder.build()

# 并行执行
result = await orchestrator.execute(
    workflow,
    mode=ExecutionMode.PARALLEL,
    initial_input="研究多Agent协作技术",
    max_workers=3  # 最多3个并发
)
```

---

## 四、实战案例：智能报告生成

### 4.1 业务场景

构建一个自动化报告生成系统：
1. **数据收集Agent**：从多个来源获取数据
2. **分析Agent**：对数据进行统计分析
3. **可视化Agent**：生成图表
4. **报告撰写Agent**：整合内容生成报告

### 4.2 完整代码实现

```python
import asyncio
from v1_multiagent.core import (
    SimpleAgent, Orchestrator, 
    WorkflowBuilder, ExecutionMode
)

class DataCollectionAgent(SimpleAgent):
    """数据收集Agent"""
    async def _on_run(self, input_data, **kwargs):
        print(f"[{self.name}] 正在收集数据: {input_data}")
        # 模拟数据收集
        await asyncio.sleep(0.5)
        return {
            "raw_data": f"原始数据: {input_data}",
            "sources": ["source1", "source2", "source3"]
        }

class AnalysisAgent(SimpleAgent):
    """数据分析Agent"""
    async def _on_run(self, input_data, **kwargs):
        print(f"[{self.name}] 正在分析数据")
        raw_data = input_data.get("raw_data", "")
        # 模拟分析
        await asyncio.sleep(0.5)
        return {
            "insights": f"分析洞察: {raw_data}",
            "metrics": {"accuracy": 0.95, "coverage": 0.88}
        }

class VisualizationAgent(SimpleAgent):
    """可视化Agent"""
    async def _on_run(self, input_data, **kwargs):
        print(f"[{self.name}] 正在生成图表")
        # 模拟图表生成
        await asyncio.sleep(0.3)
        return {
            "charts": ["chart1.png", "chart2.png"],
            "format": "png"
        }

class ReportWritingAgent(SimpleAgent):
    """报告撰写Agent"""
    async def _on_run(self, input_data, **kwargs):
        print(f"[{self.name}] 正在撰写报告")
        await asyncio.sleep(0.5)
        return {
            "report": "完整的分析报告",
            "sections": ["摘要", "数据分析", "结论"]
        }

async def main():
    """主流程"""
    orchestrator = Orchestrator()
    
    # 创建Agent
    collector = DataCollectionAgent(name="DataCollector")
    analyzer = AnalysisAgent(name="Analyzer")
    visualizer = VisualizationAgent(name="Visualizer")
    writer = ReportWritingAgent(name="ReportWriter")
    
    # 初始化所有Agent
    await collector.init()
    await analyzer.init()
    await visualizer.init()
    await writer.init()
    
    # 构建DAG工作流
    # 结构：
    # [Collector] 
    #      ↓
    # [Analyzer] → [Visualizer]
    #      ↓            ↓
    #      └→ [Writer] ←┘
    
    builder = WorkflowBuilder(orchestrator, name="AutoReport")
    
    # 1. 数据收集（起点）
    builder.add_task(collector, name="collect")
    
    # 2. 并行执行分析和可视化（都依赖collect）
    builder.add_task(analyzer, name="analyze", depends_on=["collect"])
    builder.add_task(visualizer, name="visualize", depends_on=["collect"])
    
    # 3. 报告撰写（依赖analyze和visualize）
    builder.add_task(writer, name="write", depends_on=["analyze", "visualize"])
    
    workflow = builder.build()
    
    # 执行工作流
    print("=== 开始执行报告生成工作流 ===")
    result = await orchestrator.execute(
        workflow,
        mode=ExecutionMode.PARALLEL,
        initial_input="2026年AI Agent市场分析"
    )
    
    print(f"\n=== 执行结果 ===")
    print(f"成功: {result.success}")
    print(f"报告内容: {result.results['write']}")
    
    # 清理
    await collector.terminate()
    await analyzer.terminate()
    await visualizer.terminate()
    await writer.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 执行输出示例

```
=== 开始执行报告生成工作流 ===
[DataCollector] 正在收集数据: 2026年AI Agent市场分析
[Analyzer] 正在分析数据
[Visualizer] 正在生成图表
[ReportWriter] 正在撰写报告

=== 执行结果 ===
成功: True
报告内容: {'report': '完整的分析报告', 'sections': ['摘要', '数据分析', '结论']}
```

---

## 五、高级特性

### 5.1 自定义结果聚合

```python
# 合并多个Agent的输出
def custom_aggregator(results: Dict[str, Any]) -> Dict[str, Any]:
    """自定义聚合函数"""
    merged = {
        "timestamp": datetime.now().isoformat(),
        "contributors": list(results.keys()),
        "data": {}
    }
    
    for agent_name, output in results.items():
        merged["data"][agent_name] = output
    
    return merged

result = await orchestrator.execute(
    workflow,
    mode=ExecutionMode.PARALLEL,
    aggregator=custom_aggregator
)
```

### 5.2 错误处理与重试

```python
class RobustAgent(BaseAgent):
    """带重试机制的Agent"""
    
    async def _on_run(self, input_data, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self._do_work(input_data)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"重试 {attempt + 1}/{max_retries}...")
                await asyncio.sleep(1)
```

### 5.3 执行可视化

```python
# 打印工作流结构
workflow.visualize()

# 输出：
# Workflow: AutoReport
# 
# [collect]
#     ↓
# [analyze] ─→ [write]
#     ↑         ↑
# [visualize] ─┘
```

---

## 六、性能优化建议

### 6.1 并行度控制

```python
# 根据资源限制设置并发数
result = await orchestrator.execute(
    workflow,
    mode=ExecutionMode.PARALLEL,
    max_workers=5  # 最多5个Agent同时执行
)
```

### 6.2 资源池管理

```python
# 复用Agent实例
agent_pool = [SimpleAgent(name=f"Worker{i}") for i in range(5)]

# 动态分配任务
builder.add_task(agent_pool[0], name="task1")
builder.add_task(agent_pool[1], name="task2")
```

### 6.3 异步I/O优化

```python
# 所有Agent操作都是异步的
async def batch_execute(workflows):
    """批量执行多个工作流"""
    results = await asyncio.gather(*[
        orchestrator.execute(wf) for wf in workflows
    ])
    return results
```

---

## 七、总结

DAG编排是多Agent协作系统的核心技术，通过本文你应该掌握了：

1. ✅ **DAG的核心概念**：拓扑排序、依赖管理
2. ✅ **V1-MultiAgent的使用**：WorkflowBuilder、串并行模式
3. ✅ **实战案例**：自动化报告生成系统
4. ✅ **高级特性**：结果聚合、错误处理、可视化

### 下一步

- 探索**Agent记忆系统**的设计
- 学习**动态工作流**的构建
- 了解**生产环境部署**最佳实践

---

## 资源链接

- **V1-MultiAgent框架**: https://gitee.com/v1multiagent/v1-multiagent
- **示例代码**: 本文完整代码已包含在框架中
- **技术交流**: 欢迎Star和Fork，一起构建更好的多Agent生态

---

*本文首发于技术社区，转载请注明出处。*
