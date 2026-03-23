# 改动计划（先计划，后改代码）

从现在开始，所有代码改动都遵循：
1. 先在本文件补充本次改动计划；
2. 你确认后再实施代码修改；
3. 修改后补充“实际执行结果与验证记录”。

---

## 当前待执行改动

### 目标
将因子筛选从“人工固定阈值”改为“按本批因子分布的前 10% 分位数筛选（P90）”。

### 影响范围
- 文件：`main_gp_new.py`
- 函数：`read_and_pick()`
- 影响环节：`elite_factors_further_process()` 中 elite pool 的构建逻辑

### 计划步骤
1. 读取因子总表（现有逻辑保持不变）。
2. 选择筛选指标列（当前先使用）：
   - `fitness_sharp_train`
   - `fitness_sharp_test`
   - `fitness_avg_pic_train`
   - `fitness_avg_pic_test`
3. 对每个指标计算该批次的 `P90` 阈值。
4. 仅保留同时满足“各指标 >= 各自 P90”的因子（交集筛选）。
5. 对缺失列、空样本、NaN 做容错处理并打印提示。
6. 输出最终 `elite_pool` 数量与分位数阈值。

### 验证计划
1. 运行 `read_and_pick()`，确认不报错。
2. 检查日志中是否打印各指标 P90 阈值。
3. 对比改动前后 `elite_pool` 数量变化是否符合预期（通常更自适应、不过度依赖固定阈值）。
4. 若样本规模很小，确认不会因 NaN 或缺列直接崩溃。

### 备注
- 后续可将“筛选指标列表 + 分位数阈值（如 0.9）”配置化到 `parameters.yaml`，避免硬编码。

---

## 新增需求计划（待你确认后执行）

### A. 去掉相关性筛选

#### 目标
关闭当前 GP 流程中的“高相关因子互斥”逻辑，避免过早删掉可能互补的因子。

#### 背景
当前相关性筛选主要在 `genetic.py` 的 `diff_filter(...)` 两处调用中生效（并行进化阶段、最终 all_programs 汇总阶段）。

#### 可选方案
1. **软关闭（推荐先试）**：将 `corrcoef_threshold` 设为接近 1（如 `0.9999`），保留代码逻辑但几乎不触发删除。
2. **硬关闭**：在两处调用点直接跳过 `diff_filter`，保留全部候选因子到后续 `hall_of_fame/n_components`。
3. **阶段性关闭**：仅关闭进化过程中的相关性筛选，保留最终一次去相关（折中）。

#### 验证
1. 对比每代/最终保留因子数量。
2. 对比 run 时长与内存变化。
3. 对比最终 `best_programs_df` 的多样性与指标分布。

---

### B. 引入互信息/熵来衡量非线性关系

#### 目标
在现有相关性/收益类指标外，增加可刻画非线性依赖的 fitness 指标。

#### 设计思路
1. 新增 `mi_score`（互信息）：
   - 输入：`y` 与 `y_pred`；
   - 方法：离散化后用 mutual information，或直接用 `mutual_info_regression`；
   - 输出：标量，越大越好。
2. 新增 `entropy_gain_score`（信息增益/条件熵下降）：
   - 评价因子值对收益状态（涨/跌/大波动）不确定性的降低程度；
   - 输出：标量，越大越好。

#### 验证
1. 与 `avg_sic/avg_pic` 在同一批因子上的排序一致性（Spearman）。
2. 检查新指标 Top 因子在 test 上的稳定性。

---

### C. 参考 loss 设计 fitness：惩罚“大行情方向做错”

#### 目标
对“极端行情下方向错误”施加强惩罚，避免因子在关键时段失效。

#### 可参考的现有 loss/目标
1. **Focal Loss 思想（分类）**：对难样本加权，可迁移到方向预测（上涨/下跌）任务。
2. **Asymmetric Loss / Tilted Loss**：错方向时给予更大惩罚，正确方向惩罚较轻。
3. **Weighted BCE / Cost-sensitive loss**：对大收益绝对值样本提高权重（`|y|` 越大权重越高）。
4. **Huber / Tukey biweight（回归鲁棒）**：降低噪声段影响，同时可叠加方向惩罚项。
5. **Sortino / downside-risk 类目标**：重点惩罚负收益尾部。

#### 拟实施的 fitness 版本（建议）
定义：
- 方向标签：`sign(y)`；
- 样本权重：`w_t = 1 + alpha * I(|y_t| > q95) + beta * |y_t|`；
- 损失项：`L_dir = w_t * I(sign(y_pred_t) != sign(y_t))`；
- 收益项：`R = mean(y * position(y_pred))`；
- 最终 fitness：`score = R - lambda * mean(L_dir)`（越大越好）。

#### 验证
1. 统计“极端行情（|y|>q95）方向准确率”；
2. 对比改动前后 `max_dd/calmar/sharp`；
3. 防止过拟合：必须看 train/test 同时改善。

---

### 执行顺序建议
1. 先执行 A（关闭相关性筛选）做基线。
2. 再执行 C（大行情错向惩罚）快速看交易指标改善。
3. 最后执行 B（MI/熵）作为补充排序信号或并行 metric。

---

## Triple Barrier Label 改造计划（已确认，待执行）

### 约束
- 暂不做“相关性筛选逻辑”的代码改动（按最新要求冻结）。

### 目标
1. 将当前主要 label 从固定 horizon 的 `return_f / ret_rolling_zscore` 扩展为 **Triple Barrier 路径收益标签**。
2. 同时保留离散标签能力（`-1/0/1` 或 meta-label `0/1`）用于方向稳定性评估。
3. 在 fitness 中加入“大行情错方向惩罚”。

### 实施范围（计划）
- `dataload.py`
  - 新增 label 生成分支：`predict_label = triple_barrier_ret`（连续）/ `triple_barrier_cls`（离散）。
  - 调用 `triple_barrier.get_barrier_fast(...)` 生成事件退出点与路径收益 `ret`。
  - 对齐到现有样本索引，产出：
    - 连续：`tb_ret`（可选 rolling zscore -> `tb_ret_norm`）
    - 离散：`tb_label`（-1/0/1）
- `main_gp_new.py`
  - 在 `initialize_his_data()` 后支持读取新 label 字段到 `y_train/y_test`。
  - 保持现有 `norm_y_list/raw_y_list` 兼容，并新增 triple-barrier 场景下注释与分支说明。
- `fitness.py`
  - 新增一类方向惩罚 fitness（草案名：`extreme_dir_penalty_sharp`）：
    - 奖励：收益质量（如 `sharp` 或 mean return）
    - 惩罚：`|y|` 大于分位阈值（如 q95）时的错向率
  - 新 metric 纳入 `_fitness_map`。
- `genetic.py`
  - `metric` 白名单增加新 metric 名称，保证可被 `SymbolicTransformer` 接受。

### 关键设计参数（待落地前最终确认）
1. Triple Barrier 参数：
   - `pt_sl`：如 `[1.5, 1.0]` 或按波动率动态缩放
   - `max_holding`：如 `[0, 2]`（2小时）
2. 离散标签规则：
   - `tb_ret > eps -> 1`
   - `tb_ret < -eps -> -1`
   - 其余 `0`
3. 极端行情阈值：
   - `|y| > q95`（默认）
4. fitness 融合权重：
   - `score = sharpe_like - lambda * extreme_wrong_direction_rate`

### 验证计划
1. 标签质量验证：
   - `tb_ret` 分布、偏度峰度、有效样本率；
   - `tb_label` 各类占比（防止类别塌缩）。
2. 训练/测试一致性：
   - 比较 old label vs triple-barrier label 的 Top 因子重叠率与稳定性。
3. 交易相关验证：
   - 对比 `sharp/calmar/max_dd`；
   - 单独统计“极端行情错向率”是否下降。
4. 回退机制：
   - 保留原有 label 分支，支持配置开关一键回退。
