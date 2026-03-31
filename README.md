# 工艺决策辅助系统

该项目面向 PCB / FPCB 工艺场景进行决策辅助。有三个核心功能：
    1、异常反馈入口：数据来源“异常库”，为产线班组长分析现场异常，快速得到可执行排查动作。（项目内已有模板样本）
    2、工艺优化入口：基于 EXCEL 数据，对工时与工序路径分析瓶颈，输出优化建议。
    3、RAG 工艺问答：有两条可运行的工艺问答链路。一条是自己实现的 RAG，另一条是并发的 LangChain 对照版。可自选路径。

项目出发点：很简单，就是方便工艺工程师的系统。帮助产线判决：“我现在这个异常先看什么”“为什么系统这么判断”“这个建议能不能追溯”。所以这个仓库里保留了两条思路并行存在：一条是结构化决策链，强调规则、案例和可解释；另一条是文档问答链，强调知识检索、来源引用和演示效果。
这样一来，既能直接问知识库，在历史中查询经验，或直接查询工艺工程师们设定的标准排查路径先行。

## 界面入口


![登录页](docs/screenshots/01-login.png)

系统从登录页进入，默认账号是 `user`，密码是 `123`。后续可自行配置云部署或接其他 API。

![角色入口页](docs/screenshots/02-role.png)

登录后会先来到角色页。这里把系统能力拆成了三个入口：班组长入口负责异常诊断，工程师入口负责路径优化，右边是工艺问答入口。正式版 RAG 和 LangChain 实验版 两个 RAG 入口。

## 异常诊断

异常诊断这部分没有直接让 RAG 接管，而是继续走结构化决策链。这样做的原因是，在现场排查这类问题里，可控和可解释通常比“像聊天”更重要。系统内部有一套样本知识库，里面放了 PCB / FPCB 常见工序下的异常、原因、规则和案例；前端表单先按工序切换，再显示这个工序真正需要填写的参数。

![异常诊断页](docs/screenshots/03-diagnose-form-a.png)

详细诊断页示例

![异常诊断页补充视图](docs/screenshots/04-diagnose-form-b.png)

根据权重，规则匹配、案例相似、原因画像，以及历史反馈对有效率的修正。它的好处是你可以比较自然地讲清楚系统为什么给出某条建议，而不是只能说“模型觉得像”。

## 路径优化

给工艺工程师用的小工具。它会读取 `reference` 目录下的 Excel 数据，结合工序节拍、负载比、良率和目标产能去找瓶颈，快速把“快速识别问题工段并形成初步建议”这一步做出来。

![路径优化输入页](docs/screenshots/05-optimize-form.png)

输入方式保持得比较简单，因为这个模块想体现的是分析逻辑，而不是复杂的数据接入界面。你输入文件名、关注工段、目标日产能之后，系统会给出一份结果报告。

![路径优化结果页](docs/screenshots/06-optimize-result.png)

结果展示页

## 工艺问答两条 RAG 链 

正式版 RAG 是项目自己搭的，链路里包含了文档接入、切分、向量化、检索、查询归一化、轻量重排、证据门控、回答生成和来源引用。结构清晰后续可以容易迭代。

LangChain 版则是专门做出来的对照版。不替换主干，只是单独保留一条 LangChain 标准组件的逻辑 `Splitter -> Embeddings -> FAISS -> Retriever -> Prompt -> LLM` 

![LangChain 实验版入口页](docs/screenshots/08-langchain-form.png)

LangChain

![LangChain 实验版结果页](docs/screenshots/07-langchain-result.png)

正式版 RAG 和 LangChain 版都能做工艺问答，前者更适合做正式演示和后续扩展，后者展示框架。

## 历史记录

历史页也做过一轮专门优化。最早的版本更像开发时的报告列表，适合查 JSON 来调试，不太适合普通用户。现在它先展示业务卡片，把时间、类型、主题、关键结论和摘要先放在第一层，真正的查阅历史。

![历史记录页](docs/screenshots/09-history.png)



## 本地运行

先安装依赖，然后准备好 `.env` 文件，再启动 Web 服务就可以了。RAG 相关功能由于作者自己用的是阿里千问，调的 OpenAI 的接口读取配置。如果你想用问答能力，需要在 `.env` 里配好 `PROCESS_ASSISTANT_API_KEY`、`PROCESS_ASSISTANT_BASE_URL`、`PROCESS_ASSISTANT_CHAT_MODEL` 和 `PROCESS_ASSISTANT_EMBED_MODEL`。如果你暂时只看结构化诊断和优化模块，不配这部分也没关系。

```bash
pip install -r requirements.txt
python web_ui.py
```

启动后直接访问 `http://127.0.0.1:5050` 就可以了。

如果你更喜欢先从命令行看，也可以直接用 CLI。这个项目保留了几条比较直观的命令，比如异常诊断、反馈写入、RAG 建索引、自研版问答和 LangChain 问答。命令参数已经尽量做得平实，配合 `examples` 目录就能跑通。

```bash
python -m process_assistant.cli diagnose --input examples/diagnosis_request.json
python -m process_assistant.cli feedback --input examples/feedback_event.json
python -m process_assistant.cli rag-build --docs-dir data/rag_docs --index-dir data/rag_index
python -m process_assistant.cli rag-ask --index-dir data/rag_index --question "焊线后虚焊先看什么？"
python -m process_assistant.cli rag-build-lc --docs-dir data/rag_docs --index-dir data/rag_index_langchain
python -m process_assistant.cli rag-ask-lc --index-dir data/rag_index_langchain --question "压合歪斜一般先看哪些因素？"
```

省事可以直接点根目录下 ' start_web_ui.bat '启动。



## 最后补一句项目定位

这是我从一个工艺工程师的视角把我留意到的几条常见工艺辅助能力认真做成样子的项目：结构化诊断负责把现场判断说清楚，路径优化负责把瓶颈看出来，RAG 负责把文档知识接进来，而 LangChain 对照版则负责把技术路径讲明白。所有事情也并非只能靠大模型来解决。

我在 code 里写了些注释说明这些函数之间的边界，他们之间是很清楚的，很多地方也刻意保留了可解释性。后面无论你想继续按你所在的企业环境补知识库、继续打磨前端，还是把某条链路真正往生产方向推进，估计应该都会比较平滑了。
