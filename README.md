# 工艺知识驱动的决策辅助系统（原型）

通过`决策引擎 + UI壳` 的后端原型，以PCB/柔性板的产品的工艺串行路线为例，造的一个决策辅助系统。当前 web 页面正在完善中。有两个核心能力：

- 生产问题异常诊断系统：为产线班组长服务：当产线遇到异常，可以通过结构化系统输入，实现即使在输入信息不完整的情况下，根据已有数据，反馈分析给出多条处理建议和预估用时，并存档。
- 工艺路径优化系统：为工艺工程师服务：基于投喂的历史工序工时或不良统计数据，分析优化路径以及具体工序优化方案，辅助改善。

网页有登录管理，可以进行搜索查询“历史数据”。

## 1. 模型的核心数据

- `Process`：工序
- `Symptom`：异常表现
- `Cause`：原因
- `Solution`：解决方案（动作清单）
- `Case`：历史案例
- `ProcessTime`：工时记录（从 Excel 抽取）
- `FeedbackEvent`：反馈闭环事件（成功/失败）

对应代码：`process_assistant/models.py`

## 2. 工艺路线（PCB/FPCB）

预置路线：

1. PI膜下料
2. 导电浆料印刷
3. 烘烤固化
4. 曝光对位
5. 显影/蚀刻
6. 清洗干燥
7. 双面胶/绝缘层贴合
8. 冲孔冲型
9. 预焊接
10. 焊线
11. 模压定型
12. 电阻/耐压/外观检测

知识库文件：`data/knowledge_base.json`

## 3. 异常诊断引擎（非LLM核心）

实现文件：`process_assistant/diagnosis_engine.py`

机制：

- 规则召回（Rule）
- 案例相似召回（Case Similarity）
- 原因画像匹配（Cause Profile + 条件判断）
- 历史反馈有效率修正（Feedback Effectiveness）

输出严格为动作方案，且每条动作带追溯信息：

- 规则ID
- 案例ID
- 反馈有效率

## 4. 工艺路径优化引擎（第一阶段无AI）

实现文件：`process_assistant/optimization_engine.py`

功能：

- 工序统计：`avg/max/std`（CT、ECT）
- 瓶颈识别：基于 `ECT/节拍`、日产量偏低、不良率阈值
- 优化建议：`并行 parallel / 合并 merge / 替代 substitute`

Excel解析：`process_assistant/excel_loader.py`

## 5. 反馈

- 写入反馈：`feedback_log.jsonl`
- 评分时自动计算原因有效率（拉普拉斯平滑
- 下一次诊断自动使用更新后的有效率

## 6. 运行方式

1、在项目根目录执行：

```bash
python -m process_assistant.cli diagnose --input examples/diagnosis_request.json --output reports/diagnosis_report.json
python -m process_assistant.cli optimize --excel "xxx.xlsx" --output reports/optimization_report.json
python -m process_assistant.cli feedback --input examples/feedback_event.json
```
2、转至 ## 9. 本地 Web UI（现场入口）


## 7. 隐含假设与补全说明

- 假设现场可提供工序ID和症状ID，不依赖自然语言描述。
- 你提供的 Excel 主要是单日/单版本记录，因此 `std` 在部分工序可能为 0；系统已支持多文件输入用于跨日统计。
- 工艺知识库先以规则+案例方式实现，便于可追溯和工程维护；后续可把 LLM 作为“术语归一化/解释增强层”，但不替代排序逻辑。
- `process_id` 与 Excel 工序名的映射当前为规则映射，可在后续接入主数据表提升一致性。

## 8. 推荐下一步

- 接入 MES/质量系统，自动生成结构化 `DiagnosisRequest`。
- 建立“工艺参数快照”表（温度、浓度、湿度、设备状态），增强条件匹配精度。
- 增加 A/B 反馈策略，在线更新规则权重。

## 9. 本地 Web UI（现场入口）

本地 Web 入口文件：`web_ui.py`

1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

2. 启动服务

```bash
python web_ui.py
```

或直接双击执行一键脚本：

```bash
start_web_ui.bat
```

3. 浏览器打开

```text
http://127.0.0.1:5050
```

页面有两个不同角色入口：

- 异常诊断（班组长）
- 路径优化（工程师）

新增现场化入口能力：

- 登录页：`/login`
- 账号角色页（班组长/工程师分入口）：`/role`
- 历史记录查询页：`/history`

默认账号：

- 用户名：`user`
- 密码：`123`

可选安全配置（推荐对外发布时使用环境变量覆盖）：

```bash
PROCESS_ASSISTANT_LOGIN_USERNAME=user
PROCESS_ASSISTANT_LOGIN_PASSWORD=123
PROCESS_ASSISTANT_SECRET_KEY=change-this-secret-before-production
```

二次发布到 GitHub 前建议检查 `reports/`、`reference/`、`data/feedback_log.jsonl` 是否包含本地业务数据。
