# 项目立项建议书：AI 日志 (Project Proposal: AI Journal)

**日期**：2026-02-01  
**版本**：v2.2 (增加核心业务场景案例)

---

## 1. 项目背景与愿景 (Executive Summary)

当前市场上的 AI 产品普遍面临“高留存”与“高价值闭环”难以兼得的困境。本项目旨在构建名为 **AI 日志 (AI Journal)** 的智能系统，专注于**成人教育与职业成长**领域。

通过引入先进的**长期记忆（Long-term Memory）**架构，我们致力于打造一款深度理解用户职业轨迹的“成长伴侣”：
- **前端体验**：以“半结构化日志”为入口，自然沉淀用户的学习、工作与思考。
- **后端能力**：利用 AI 记忆辅助课程学习（Internal Courses），并提供面试准备、模拟与复盘等高价值职业服务。

## 2. 市场痛点与竞争分析 (Market Analysis)

目前的 AI 助手市场主要分为三类，通过分析其技术特征，本项目定于高价值的融合领域。

### 2.1 纯陪伴型 (Companionship AI)
*   **特征**：高情感价值，高用户留存，但在解决实际问题上能力薄弱。
*   **局限**：缺乏对外部世界的任务闭环能力，容易陷入无效闲聊。

### 2.2 通用工具型 (General/Productivity AI)
*   **特征**：也就是 ChatGPT/Copilot 模式，擅长回答通用问题、编写代码。
*   **局限**：缺乏个性化记忆（"不懂用户"），每次会话都是全新的，难以建立长期信任与默契，且不掌握用户的私有上下文。

### 2.3 本项目定位：基于记忆的执行型助手 (Memory-Augmented Personal Agent)
本项目不仅是两者的简单叠加，而是通过**架构分层**实现：前台维持关系感（基于记忆），后台按场景切换执行模型（通过 API/Agent 协作）。

---

## 3. 技术架构方案 (Technical Architecture)

基于开源项目 **Second-Me (AI-native Memory 2.0)** 的核心架构，本项目采用分层记忆网络（LPM）与双 Agent 协作模式。

### 3.1 核心底座：AI-native Memory 架构
我们采用分层治理策略，参考架构：[Mindverse/Second-Me](https://github.com/mindverse/Second-Me)。

*   **L0 (Raw Data 层)**：原始事实层。处理非结构化输入（文档/日志/片段）。
    *   *核心代码*：`lpm_kernel/file_data/document_service.py`
    *   *功能*：支持多格式文件上传、切块与向量化。
*   **L1 (Semantic Memory 层)**：语义记忆层。生成用户的 Bio（生平）、Clusters（观点簇）与 Shades（全局画像）。
    *   *核心代码*：`lpm_kernel/kernel/l1/l1_manager.py`
    *   *功能*：基于 Embedding 的聚类与主题生成，形成“可读的长期记忆”。
*   **L2 (Persona/Orchestrator 层)**：角色与编排层。负责融合记忆、上下文，并决定是对外输出情感还是调用执行模型。
    *   *核心代码*：`lpm_kernel/api/domains/kernel2/services/role_service.py`
    *   *功能*：System Prompt 动态绑定与记忆检索开关控制。

### 3.2 关键业务流程：双 Agent 协作模式 (The Keeper & The Solver)

1.  **Phase 1: 替身 Agent (The Keeper)**
    *   **职能**：记忆守护者。通过解析用户的半结构化日志，构建精准的用户画像。
2.  **Phase 2: 专家 Agent (The Solver)**
    *   **职能**：任务执行者。当识别到具体任务（如求职、编程）时，驱动 Solver 执行。
    *   **案例**：在**模拟面试**场景中，Keeper 提供用户简历与项目经历，Solver 扮演面试官进行提问与点评，无需用户重复输入背景信息。

### 3.3 现有技术资产盘点
基于已有代码库的分析，核心模块就绪度如下：
*   **统一对话接口**：已实现兼容 OpenAI 协议的本地/远程统一调用接口 (`routes_l2.py`)。
*   **检索增强 (RAG)**：L0 向量检索与 L1 全局检索链路已打通 (`knowledge_service.py`)。
*   **高级推理流**：已实现“需求增强 → 专家求解 → 结果校验”的多阶段对话逻辑 (`advanced_chat_service.py`)。

### 3.4 模型分工与原理 (Base/Teach/Thinking)

为复现 Second-Me 的记忆与推理链路，需要至少三类模型协作，原因是“记忆写入、可检索表达、深度推理”三条能力链的约束不同：

1.  **基础模型（Base Model）**
    *   **职责**：承担日常对话、日志摘要、任务指令的基础理解。
    *   **作用原理**：保证交互可用性与成本可控，是系统的默认执行模型。
2.  **Teach 模型（Teaching/Teacher Model）**
    *   **职责**：用于“生成可训练/可写入的记忆内容”，以及对齐结构化输出。
    *   **作用原理**：提供更稳定的“记忆条目生成与清洗”能力。
    *   **说明**：若不使用 OpenAI 一体化方案，通常需要两类模型配合：
        *   **文本模型**：产出可读记忆、摘要与结构化条目。
        *   **Embedding 模型**：生成向量表示，用于 L0/L1 检索与聚类。
3.  **Thinking 模型（Reasoning/Deliberation Model）**
    *   **职责**：处理复杂任务推理、角色扮演（如模拟面试）、多阶段校验。
    *   **作用原理**：提升“深度推理质量”与“复杂任务闭环”的准确性。

该模型分工可映射为：Base 负责“稳定交互”、Teach 负责“高质量记忆写入”、Thinking 负责“关键任务求解”。这也是本项目在工程侧需要多模型协作的根本原因。

### 3.5 外接任务型 Agent 的模型要求 (Reading/Interview Agents)

3.4 主要覆盖“复现 Second-Me 的记忆/训练链路”。当引入**阅读助学**或**面试教练**等外接 Agent 时，通常还需要以下额外模型能力：

1.  **任务型执行模型（Task Model）**
    *   **用途**：面向具体业务场景的高质量输出（课程讲解、面试追问、评测反馈）。
    *   **差异**：相较 Base/Teach，更偏“高专业度内容生成”与“强一致对话”。
2.  **检索与工具调用模型（Tool/Retrieval Model，可复用 Base 或 Thinking）**
    *   **用途**：调用外部知识库、课程大纲、题库或企业内部资料。
    *   **差异**：要求更稳定的结构化输出与工具选择能力。
3.  **评估/校验模型（Critic/Validator，可复用 Thinking）**
    *   **用途**：对面试回答或学习成果做评分与纠错，形成可解释反馈。

结论：外接 Agent 不一定“必须新增一套模型”，但至少需要**在 Task/Tool/Critic 三类能力上有可靠承担者**。模型复用可以按“能力映射 + 场景约束”来落地，建议如下：

1.  **复用 Thinking 作为 Task+Critic（资源受限方案）**
    *   **适用**：早期 MVP、成本敏感阶段。
    *   **做法**：
        *   Task：用同一 Thinking 模型承载“面试追问/课程讲解”等高质量生成。
        *   Critic：通过双轮调用（先回答，再自评）完成评分与纠错。
    *   **风险**：吞吐下降、延迟上升，需要对高价值场景限流。
2.  **Base 负责 Tool/检索调用，Thinking 负责 Task（分工复用）**
    *   **适用**：检索调用频繁、但推理复杂度中等的场景（阅读助学）。
    *   **做法**：
        *   Base 模型输出结构化工具调用（查询课程库/题库）。
        *   Thinking 模型生成解释、总结与追问。
    *   **收益**：降低思维模型的负载与成本。
3.  **Teach 文本模型复用为“知识讲解” Task（内容一致性优先）**
    *   **适用**：课程讲解、知识点拆解等“稳定输出”场景。
    *   **做法**：复用 Teach 文本模型的“结构化、可读”输出能力，保证讲解一致性；Embedding 仍用于检索。
4.  **独立任务模型（效果优先方案）**
    *   **适用**：面试教练等高价值、强效果需求场景。
    *   **做法**：为面试/阅读分别配置任务模型，Thinking 仅承担“复杂推理/校验”。
    *   **收益**：质量最佳，但成本与运维复杂度最高。

经验上可采用“**分阶段复用**”：MVP 用方案 1/2，稳定后在高价值场景引入独立任务模型。这样既能控成本，又能逐步提升体验。

---

## 4. 核心应用场景与案例 (Core Scenarios)

### 4.1 宏观场景：AI助教 (AI Journal for Career Growth)
*   **背景**：公司希望通过 AI 日志帮助学员在职业上快速成长。
*   **流程**：
    1.  学员在学习过程中，通过 AI Journal 记录每日思考、课程笔记与实战心得。
    2.  AI 自动分析日志，生成“成长轨迹报告”，识别学员的知识盲区与优势。
    3.  结合公司课程体系，AI 主动推送个性化的补充材料或下一步行动建议。
*   **价值**：将被动的课程学习转化为主动的“反思与复盘”循环，提升完课率与实战转化率。

### 4.2 微观场景：沉浸式助学 (Study Companion Module)
*   **背景**：在成人教育大框架下，依托具体的课程或书籍（如《非暴力沟通》、《金字塔原理》），问问大象直播课文档开发的轻量级模块。
*   **流程**：
    1.  **导入**：系统预置书籍/课程的核心知识库（L0）。
    2.  **共读**：用户每读完一章，在日志中记录感悟。
    3.  **反馈**：AI 扮演“书友”或“助教”，基于书本内容与用户日志进行深度探讨，甚至发起由书本内容衍生的思考题。
*   **定位**：作为 MVP（最小可行性产品）的首选落地场景，技术复杂度低，用户感知强。

### 4.3 执行场景：全链路面试教练 (End-to-End Interview Coach)
*   **背景**：在具体的职业任务执行层面，提供面试前、中、后的全流程服务。
*   **流程**：
    1.  **面试前（准备）**：AI 读取用户的历史项目日志与简历（L1 Memory），自动生成“针对该岗位的自我介绍”与“预期问题清单”。
    2.  **面试中（模拟）**：启动“模拟面试”模式，Expert Agent 扮演严厉的面试官，进行语音/文字模拟攻防。
    3.  **面试后（复盘）**：用户记录真实面试的录音或回忆，AI 进行复盘分析，指出回答亮点与改进空间，并更新到用户的“面试经验库”中。

---

## 5. 关键技术挑战与应对策略 (Challenges & Solutions)

从工程与量化指标视角，本项目需重点解决以下问题：

### 5.1 输出稳定性 (Output Reliability)
*   **挑战**：执行型任务要求严格的结构化输出（JSON/Action）。
*   **对策**：对关键任务强制使用 Schema 约束。

### 5.2 记忆准确性 (Hallucination Control)
*   **挑战**：模型可能混淆记忆或产生幻觉。
*   **对策**：建立记忆评测集。LPM 架构本身通过分层检索能有效缓解此问题。

### 5.3 安全与权限 (Security & Permissions)
*   **挑战**：执行层涉及外部工具调用，存在越权风险。
*   **对策**：实施**只读/可写分离**，引入“人机回环”机制。

---

## 6. 关键量化指标体系 (KPIs)

### 6.1 稳定性指标
*   **Parse Success Rate**：结构化输出解析成功率。
*   **Tool Call Precision**：工具调用准确率。

### 6.2 记忆质量指标
*   **Memory Precision/Recall**：记忆准召率。
*   **Memory Conflict Rate**：记忆冲突率。

---

## 7. 实施路线图 (Implementation Roadmap)

### 阶段一：MVP 验证 - [场景：书本助学] (预计 2-3 个月)
**目标**：跑通数据导入、记忆写入与基于特定书籍（或问问大象直播课文档）的对话。
1.  **模型准备**：完成 Base/Teach/Embedding 的选型与联调，确保记忆写入与检索可用。
2.  **数据接入**：支持上传课程 PDF 或书籍 epub。
3.  **助学对话**：验证基于 L0 检索的“读书伴侣”体验（对应场景 4.2）。
4.  **日志记录**：基础的日志录入与每日回顾功能。

### 阶段二：高级能力构建 - [场景：面试教练] (预计 3-4 个月)
**目标**：实现“双 Agent”协作、Thinking 模型接入与复杂任务闭环。
1.  **用户画像生成**：基于日志生成 L1 Bio，用于面试自我介绍生成。
2.  **专家模式**：开发“面试官”Persona 与模拟面试逻辑（对应场景 4.3）。
3.  **Thinking 接入**：引入推理模型，提升模拟面试的深度追问与一致性。
4.  **上下文打包**：实现从“我的经历”到“面试官问题”的信息流转。

### 阶段三：产品化与集成 (Productization) (预计 4个月+)
**目标**：企业级集成、场景运营与稳定性建设。
1.  **API 对接**：与企业内部学习平台（LMS）打通（对应场景 4.1）。
2.  **前端重构**：开发支持多场景切换的 App/Web 端。
3.  **指标闭环**：完善 KPI 埋点与质量回归机制。

---

## 8. 结论 (Conclusion)

本项目通过 **AI 日志** 这一形态，将抽象的“长期记忆技术”具象化为用户可感知的“职业成长工具”。从轻量级的**书籍助学**切入，逐步通过 **AI 面试** 等高价值场景建立壁垒，最终形成企业级的**人才成长辅助系统**。技术路径清晰，商业场景明确，建议按路线图推进。

---

## 9. 附录：参考资料 (Appendix)

*   **AI-native Memory (LPM 基础理论)**: [https://arxiv.org/abs/2406.18312](https://arxiv.org/abs/2406.18312)
*   **AI-native Memory 2.0 (系统架构)**: [https://arxiv.org/abs/2503.08102](https://arxiv.org/abs/2503.08102)
*   **ChatGPT (RLHF/Memory)**: [OpenAI Method](https://openai.com/index/chatgpt/)
*   **Second Me (本项目核心参考)**: [https://github.com/mindverse/Second-Me](https://github.com/mindverse/Second-Me)
*   **llama.cpp**: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
*   **ChromaDB**: [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)
*   **LangChain**: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
*   **LlamaIndex**: [https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)
*   **OpenAI Cookbook**: [https://github.com/openai/openai-cookbook](https://github.com/openai/openai-cookbook)
