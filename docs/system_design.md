# 系统设计说明（数据模型优先）

## 1. 目标边界

系统核心是 `决策引擎`，不是单纯的对话式 AI：

- 输入：强结构化字段（工序ID、症状ID、观测参数）
- 输出：可执行动作（动作、责任角色、时长、检查点）
- 可追溯：每条建议都绑定规则ID/案例ID/反馈权重

LLM 仅作为可插拔增强层（术语归一化、解释文案），不参与核心排序。

## 2. 实体模型

### 2.1 主实体

- Process(id, route_id, sequence, name, category, stage)
- Symptom(id, code, name, severity)
- Cause(id, name, category, process_ids, symptom_ids, base_weight, conditions)
- Solution(id, cause_id, actions[], checkpoints[], expected_minutes, owner_role)
- Case(id, process_id, symptom_ids[], cause_id, context, success_score)
- ProcessTime(source_file, section, process_id, process_name, sequence, ct_sec, defect_rate, extra_factor, ect_sec, hourly_capacity, takt_sec, manpower, daily_output, ...)
- FeedbackEvent(request_id, cause_id, solution_id, result)

### 2.2 关系

- Process 1..n Symptom（通过规则/案例体现）
- Symptom n..n Cause
- Cause 1..n Solution
- Case -> (Process, Symptom[], Cause)
- ProcessTime -> Process（通过 process_id 映射，当前支持规则映射）

## 3. 异常诊断决策路径

### 3.1 召回层

1. 规则召回（Rule Match）
2. 案例召回（Case Similarity）
3. 原因画像召回（Cause Profile）

### 3.2 排序层

综合分数：

`FinalScore(cause) = (RuleScore + CaseScore + ProfileScore) * CauseBaseWeight * FeedbackEffectiveness`

其中：

- `FeedbackEffectiveness = (success+1)/(success+fail+2)`（拉普拉斯平滑）
- 条件匹配（如 defect_rate、humidity）通过倍率修正

### 3.3 输出层

按 cause 排序后，输出对应 solution 的动作清单：

- actions[]：执行步骤
- checkpoints[]：判定标准
- owner_role / expected_minutes
- traceability：规则/案例/反馈来源

## 4. 工艺路径优化决策路径（第一阶段无AI）

### 4.1 数据提取

从 Excel 解析 `ProcessTime`，支持多 section、多文件。

### 4.2 统计分析

按工序聚合输出：

- ct_avg / ct_max / ct_std
- ect_avg / ect_max / ect_std

### 4.3 瓶颈识别规则

满足任一条件即判定候选瓶颈：

- `ECT / takt > 1.05`
- `daily_output < 0.9 * section_median_output`
- `defect_rate >= 0.08`

### 4.4 优化策略规则

- parallel：ECT 超节拍 -> 建议并行工位人力下限
- merge：extra_factor 偏高 -> 前置准备动作
- substitute：高不良率 -> 防错治具/自动检测替代
- merge opportunities：相邻短节拍工序可尝试合并

## 5. 反馈闭环机制

- 反馈事件写入 `feedback_log.jsonl`
- 下次诊断自动读取并修正原因有效率
- 可进一步扩展为：分机台、分班次、分产品族权重

## 6. 关键建模选择与补全假设

1. 工序主键优先使用 `process_id`，Excel 工序名通过规则映射到 ID。
2. 对输入不完整场景，未提供条件项按“中性分”处理，不直接丢弃候选。
3. 当前优化引擎使用规则和统计，不依赖任何生成模型。
4. 对“工艺经验”采用 `规则 + 案例` 双通道建模，便于追溯和持续维护。
5. 对 Excel 的辅助岗位/返工岗位做过滤，避免污染瓶颈统计。

## 7. 后续扩展建议（不影响当前可用性）

- 加入设备状态流（温度、压力、浓度、寿命计数）作为条件特征。
- 将 Case 检索升级为向量检索（仅做召回增强）。
- 对建议动作建立标准作业模板（SOP step id）。
- 增加“执行前后指标对比”用于自动评估建议有效性。
