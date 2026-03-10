# Agent记忆系统深度解析：对话+向量双引擎

> 原创技术文章 | V1-MultiAgent团队出品
> 阅读时间：12分钟

## 前言

在构建AI Agent系统时，**记忆能力**是决定Agent智能程度的关键因素。一个没有记忆的Agent，每次对话都是"从零开始"，无法理解上下文，更无法积累知识。

本文将深入解析Agent记忆系统的设计原理，介绍**对话历史管理**和**向量语义检索**双引擎架构，并分享V1-MultiAgent框架的实战经验。

---

## 一、为什么Agent需要记忆？

### 1.1 无记忆Agent的困境

想象一个客服Agent的对话：

```
用户：我想退货
Agent：好的，请提供订单号
用户：订单号是12345
Agent：请问您想办理什么业务？  ← 完全忘记了上下文！
```

**问题根源**：
- ❌ 无法维护对话上下文
- ❌ 不能引用之前提到的信息
- ❌ 无法从长期交互中学习

### 1.2 人类记忆的启发

人类的记忆分为三类：

| 记忆类型 | 持续时间 | Agent对应实现 |
|---------|---------|--------------|
| **瞬时记忆** | 几秒 | 当前对话上下文 |
| **短期记忆** | 几分钟-几小时 | 对话历史窗口 |
| **长期记忆** | 几天-永久 | 向量数据库 |

**Agent记忆系统的目标**：模拟人类的三层记忆机制，实现：
- ✅ 上下文感知的对话
- ✅ 历史信息的快速检索
- ✅ 知识的持久化存储

---

## 二、对话历史管理

### 2.1 基础对话窗口

最简单的记忆实现——维护最近N轮对话：

```python
class ConversationMemory:
    """对话历史管理"""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append(Message(role=role, content=content))
        # 保持窗口大小
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]
    
    def get_context(self) -> List[Dict]:
        """获取对话上下文"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def clear(self):
        """清空历史"""
        self.messages = []
```

### 2.2 智能窗口管理

简单的固定窗口有问题：可能截断重要信息。改进方案：

```python
class SmartWindowMemory(ConversationMemory):
    """智能窗口管理"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages: List[Message] = []
        self.summaries: List[str] = []  # 历史摘要
    
    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        self._compress_if_needed()
    
    def _compress_if_needed(self):
        """当窗口过大时，压缩早期对话为摘要"""
        total_tokens = sum(len(msg.content) for msg in self.messages)
        
        if total_tokens > self.max_tokens:
            # 将前50%的对话压缩为摘要
            to_summarize = self.messages[:len(self.messages)//2]
            summary = self._summarize_messages(to_summarize)
            self.summaries.append(summary)
            
            # 保留近期对话
            self.messages = self.messages[len(self.messages)//2:]
    
    def _summarize_messages(self, messages: List[Message]) -> str:
        """使用LLM生成摘要"""
        prompt = f"请总结以下对话的关键信息：\n{self._format_messages(messages)}"
        return llm_generate(prompt)
    
    def get_context(self) -> List[Dict]:
        """组合摘要和近期对话"""
        context = []
        
        # 添加历史摘要
        if self.summaries:
            context.append({
                "role": "system",
                "content": f"历史对话摘要：{self.summaries[-1]}"
            })
        
        # 添加近期对话
        context.extend([
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ])
        
        return context
```

### 2.3 对话状态追踪

除了消息内容，还需要追踪对话状态：

```python
@dataclass
class ConversationState:
    """对话状态"""
    topic: str = ""                    # 当前话题
    user_intent: str = ""              # 用户意图
    pending_questions: List[str] = None  # 待回答的问题
    extracted_entities: Dict = None    # 提取的实体
    sentiment: str = "neutral"         # 情感倾向

class StatefulMemory(SmartWindowMemory):
    """带状态追踪的记忆"""
    
    def __init__(self, max_tokens: int = 4000):
        super().__init__(max_tokens)
        self.state = ConversationState()
    
    def update_state(self, **kwargs):
        """更新对话状态"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def get_context_with_state(self) -> List[Dict]:
        """获取包含状态的上下文"""
        context = self.get_context()
        
        # 在开头添加状态信息
        state_info = self._format_state()
        if state_info:
            context.insert(0, {
                "role": "system",
                "content": f"【对话状态】{state_info}"
            })
        
        return context
```

---

## 三、向量语义检索

### 3.1 为什么需要向量检索？

对话历史的问题：只能按时间顺序查找，无法基于语义检索。

**场景**：用户之前提到过"我在北京工作"，后来问"那边天气怎么样"。
- 对话历史：无法关联"那边"= "北京"
- 向量检索：可以语义匹配"那边"和相关地理位置信息

### 3.2 向量检索原理

```
文本 → Embedding模型 → 向量(768维或1536维)

查询："那边天气怎么样" → [0.1, -0.3, 0.8, ...]
记忆："我在北京工作"   → [0.2, -0.2, 0.7, ...]  ← 相似度高
记忆："我喜欢吃苹果"   → [-0.5, 0.1, 0.3, ...]  ← 相似度低
```

### 3.3 V1-MultiAgent向量记忆实现

```python
from typing import List, Dict, Optional
import numpy as np

class VectorMemory:
    """向量语义记忆"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        vector_store: Optional[VectorStore] = None
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store or InMemoryVectorStore()
    
    async def add_memory(
        self, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            metadata: 元数据（时间、类型、来源等）
        
        Returns:
            memory_id: 记忆ID
        """
        # 1. 生成embedding
        embedding = await self._generate_embedding(content)
        
        # 2. 存储到向量库
        memory_id = await self.vector_store.add(
            embedding=embedding,
            content=content,
            metadata=metadata or {}
        )
        
        return memory_id
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[MemoryResult]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            相关记忆列表
        """
        # 1. 查询向量化
        query_embedding = await self._generate_embedding(query)
        
        # 2. 向量检索
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter_dict
        )
        
        return results
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本embedding"""
        response = await openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
```

### 3.4 向量存储后端

支持多种向量存储方案：

```python
class InMemoryVectorStore:
    """内存向量存储（适合测试）"""
    
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.contents: List[str] = []
        self.metadatas: List[Dict] = []
    
    async def add(
        self, 
        embedding: List[float], 
        content: str, 
        metadata: Dict
    ) -> str:
        idx = len(self.vectors)
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadatas.append(metadata)
        return f"mem_{idx}"
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[MemoryResult]:
        if not self.vectors:
            return []
        
        query_vec = np.array(query_embedding)
        
        # 计算余弦相似度
        similarities = [
            self._cosine_similarity(query_vec, vec)
            for vec in self.vectors
        ]
        
        # 排序并返回top_k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            MemoryResult(
                content=self.contents[i],
                score=float(similarities[i]),
                metadata=self.metadatas[i]
            )
            for i in top_indices
        ]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 生产环境推荐：使用专业的向量数据库
class ChromaVectorStore:
    """基于Chroma的向量存储"""
    
    def __init__(self, collection_name: str = "agent_memory"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    async def add(
        self, 
        embedding: List[float], 
        content: str, 
        metadata: Dict
    ) -> str:
        import uuid
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[MemoryResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        memories = []
        for i in range(len(results['documents'][0])):
            memories.append(MemoryResult(
                content=results['documents'][0][i],
                score=results['distances'][0][i],
                metadata=results['metadatas'][0][i]
            ))
        return memories
```

---

## 四、组合记忆系统

### 4.1 三层记忆架构

```python
class CombinedMemory:
    """组合记忆系统：对话历史 + 向量检索"""
    
    def __init__(
        self,
        conversation_memory: Optional[ConversationMemory] = None,
        vector_memory: Optional[VectorMemory] = None,
        config: Optional[MemoryConfig] = None
    ):
        self.config = config or MemoryConfig()
        
        # 三层记忆
        self.immediate = ImmediateMemory()  # 瞬时记忆
        self.conversation = conversation_memory or SmartWindowMemory()
        self.vector = vector_memory or VectorMemory()
    
    async def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加记忆到所有层级"""
        # 1. 瞬时记忆
        self.immediate.add(role, content)
        
        # 2. 对话历史
        self.conversation.add_message(role, content)
        
        # 3. 向量记忆（异步，不阻塞主流程）
        if self._should_remember(content):
            asyncio.create_task(
                self.vector.add_memory(
                    content=content,
                    metadata={
                        "role": role,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
            )
    
    async def get_context(
        self,
        query: Optional[str] = None,
        recent_turns: int = 5
    ) -> List[Dict]:
        """
        获取组合上下文
        
        Args:
            query: 当前查询（用于向量检索）
            recent_turns: 最近对话轮数
        """
        context = []
        
        # 1. 添加系统提示
        context.append({
            "role": "system",
            "content": self._get_memory_prompt()
        })
        
        # 2. 添加相关历史记忆（向量检索）
        if query:
            relevant_memories = await self.vector.search(
                query=query,
                top_k=self.config.vector_top_k
            )
            if relevant_memories:
                memory_text = "\n".join([
                    f"- {m.content}"
                    for m in relevant_memories
                    if m.score > self.config.similarity_threshold
                ])
                if memory_text:
                    context.append({
                        "role": "system",
                        "content": f"【相关历史】\n{memory_text}"
                    })
        
        # 3. 添加近期对话历史
        recent_messages = self.conversation.get_context()[-recent_turns*2:]
        context.extend(recent_messages)
        
        return context
    
    def _should_remember(self, content: str) -> bool:
        """判断内容是否值得长期记忆"""
        # 过滤掉简单的问候、感谢等
        trivial_patterns = ["你好", "谢谢", "再见", "嗯", "哦"]
        return not any(p in content for p in trivial_patterns)
    
    def _get_memory_prompt(self) -> str:
        """记忆系统提示词"""
        return """你是一个具有长期记忆能力的AI助手。
- 你可以记住用户之前提到的重要信息
- 在回答时，优先参考相关历史记忆
- 如果记忆与当前问题相关，请主动引用"""
```

### 4.2 记忆优先级算法

```python
class MemoryRanker:
    """记忆优先级排序"""
    
    def rank_memories(
        self,
        query: str,
        memories: List[MemoryResult],
        current_context: List[Message]
    ) -> List[MemoryResult]:
        """
        综合多因素对记忆排序
        """
        scored_memories = []
        
        for mem in memories:
            score = 0.0
            
            # 1. 语义相似度 (40%)
            score += mem.score * 0.4
            
            # 2. 时间衰减 (30%)
            age_hours = self._get_memory_age(mem)
            time_score = np.exp(-age_hours / 24)  # 24小时衰减
            score += time_score * 0.3
            
            # 3. 访问频率 (20%)
            access_count = mem.metadata.get("access_count", 1)
            freq_score = min(access_count / 10, 1.0)
            score += freq_score * 0.2
            
            # 4. 上下文相关性 (10%)
            context_score = self._context_relevance(mem, current_context)
            score += context_score * 0.1
            
            scored_memories.append((mem, score))
        
        # 按得分排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored_memories]
```

---

## 五、实战案例：个人助理Agent

### 5.1 场景描述

构建一个记得用户偏好的个人助理：
- 记住用户的喜好、习惯
- 记住重要的日期和事件
- 能够关联上下文回答问题

### 5.2 完整实现

```python
from v1_multiagent.core import BaseAgent
from v1_multiagent.memory import CombinedMemory, MemoryConfig

class PersonalAssistant(BaseAgent):
    """个人助理Agent - 具备长期记忆"""
    
    async def _on_init(self):
        """初始化记忆系统"""
        config = MemoryConfig(
            max_tokens=4000,
            vector_top_k=3,
            similarity_threshold=0.7
        )
        self.memory = CombinedMemory(config=config)
        
        # 加载用户画像（如果有）
        self.user_profile = await self._load_user_profile()
    
    async def _on_run(self, input_data: str, **kwargs) -> str:
        """运行Agent"""
        # 1. 获取组合上下文
        context = await self.memory.get_context(
            query=input_data,
            recent_turns=3
        )
        
        # 2. 添加用户消息
        await self.memory.add("user", input_data)
        
        # 3. 调用LLM
        response = await self._call_llm(context, input_data)
        
        # 4. 保存助手回复
        await self.memory.add("assistant", response)
        
        # 5. 提取并存储关键信息
        await self._extract_and_remember(input_data, response)
        
        return response
    
    async def _call_llm(
        self, 
        context: List[Dict], 
        current_input: str
    ) -> str:
        """调用LLM生成回复"""
        messages = context + [{"role": "user", "content": current_input}]
        
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def _extract_and_remember(self, user_input: str, response: str):
        """提取并记忆关键信息"""
        # 使用LLM提取实体和事实
        extraction_prompt = f"""
        从以下对话中提取值得长期记忆的信息（用户偏好、重要事实等）：
        
        用户：{user_input}
        助手：{response}
        
        只输出关键信息列表，每行一条。如果没有则输出"无"。
        """
        
        extracted = await self._call_llm(
            [{"role": "user", "content": extraction_prompt}],
            ""
        )
        
        # 存储提取的信息
        for fact in extracted.split("\n"):
            fact = fact.strip()
            if fact and fact != "无":
                await self.memory.vector.add_memory(
                    content=fact,
                    metadata={
                        "type": "extracted_fact",
                        "timestamp": datetime.now().isoformat()
                    }
                )


# 使用示例
async def demo():
    assistant = PersonalAssistant(name="小助手")
    await assistant.init()
    
    # 第一轮对话
    response1 = await assistant.run("我喜欢喝美式咖啡，不加糖")
    print(f"用户：我喜欢喝美式咖啡，不加糖")
    print(f"助手：{response1}")
    
    # 第二轮对话（测试记忆）
    response2 = await assistant.run("明天早上我要一杯咖啡")
    print(f"\n用户：明天早上我要一杯咖啡")
    print(f"助手：{response2}")  # 应该记得用户喜欢美式不加糖
    
    # 第三轮对话（跨会话记忆）
    response3 = await assistant.run("我通常喜欢什么样的咖啡？")
    print(f"\n用户：我通常喜欢什么样的咖啡？")
    print(f"助手：{response3}")  # 应该能回忆起来
```

### 5.3 记忆效果展示

```
用户：我喜欢喝美式咖啡，不加糖
助手：好的，我记住了！您是美式咖啡爱好者，不加糖。☕

用户：明天早上我要一杯咖啡
助手：好的！给您准备美式咖啡，不加糖，对吧？需要我提醒您吗？

用户：我通常喜欢什么样的咖啡？
助手：您喜欢喝美式咖啡，不加糖。这是您之前告诉我的 preferences。☕
```

---

## 六、性能优化与最佳实践

### 6.1 Embedding缓存

```python
class EmbeddingCache:
    """Embedding缓存"""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存的embedding"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.cache.get(key)
    
    def set(self, text: str, embedding: List[float]):
        """缓存embedding"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = hashlib.md5(text.encode()).hexdigest()
        self.cache[key] = embedding
```

### 6.2 异步处理

```python
async def add_memory_async(self, content: str, metadata: Dict):
    """异步添加记忆，不阻塞主流程"""
    asyncio.create_task(
        self._add_memory_task(content, metadata)
    )

async def _add_memory_task(self, content: str, metadata: Dict):
    """后台任务：生成embedding并存储"""
    try:
        embedding = await self._generate_embedding(content)
        await self.vector_store.add(embedding, content, metadata)
    except Exception as e:
        logger.error(f"添加记忆失败: {e}")
```

### 6.3 记忆压缩策略

```python
class MemoryCompressor:
    """记忆压缩器"""
    
    async def compress_old_memories(self, days: int = 30):
        """压缩旧记忆"""
        old_memories = await self.get_memories_older_than(days)
        
        # 按主题聚类
        clusters = self._cluster_memories(old_memories)
        
        for cluster in clusters:
            # 生成摘要
            summary = await self._summarize_cluster(cluster)
            
            # 删除原始记忆，保存摘要
            await self.replace_with_summary(cluster, summary)
```

---

## 七、总结

通过本文，你学习了：

1. ✅ **三层记忆架构**：瞬时记忆 + 对话历史 + 向量检索
2. ✅ **对话窗口管理**：智能压缩、状态追踪
3. ✅ **向量语义检索**：Embedding、相似度计算
4. ✅ **组合记忆系统**：多维度记忆融合
5. ✅ **实战案例**：个人助理Agent的完整实现

### 下一步

- 探索**多模态记忆**（图片、音频）
- 实现**记忆遗忘机制**
- 设计**记忆共享**（多Agent间共享知识）

---

## 资源链接

- **V1-MultiAgent框架**: https://gitee.com/v1multiagent/v1-multiagent
- **完整代码**: 见 `v1_multiagent/core/memory.py`

---

*本文首发于技术社区，转载请注明出处。*
