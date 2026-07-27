# ds000030 `phenotype/` — 全部 52 组表型数据说明

> 面向 **双相障碍（BD）vs 健康对照（HC）静息态有效连接（EC）分类** 研究。

## 数据来源

UCLA Consortium for Neuropsychiatric Phenomics (CNP)，共 272 人：HC=130, SZ=50, BD=49, ADHD=43。

数据描述论文：Poldrack et al. (2016) *Scientific Data*, 3:160110. [DOI](https://doi.org/10.1038/sdata.2016.110)

---

## 格式约定

每组由 `.tsv`（数据）和 `.json`（数据字典/编码说明）配对组成。所有表通过 `participant_id` 关联。

> **使用任何变量前，务必查看对应 `.json` 中的编码含义和缺失值标记（如 `9999`）。**

---

## 一、诊断与人口学基线

### SCID — DSM-IV-TR 结构化临床访谈（研究版）
金标准诊断工具，每名受试者最多 8 条诊断记录，含共病、发作类型（当前躁狂/抑郁/混合、既往发作、缓解）、物质使用、焦虑障碍等信息。编码对照见 `scid.json`。

### demographics — 人口学
年龄(`age`)、性别(`gender`: 1=男/2=女)、教育年限(`school_yrs`)、最高学历(`school_degree`)、种族、母语、婚姻、吸烟史等。核心协变量。

### medication — 用药史
最多记录 20 种药物，含药名(`med_name*`)、是否在用(`med_use*`)、剂量(`med_dos*`)、时长(`med_dur*`)、天数(`med_days*`)。需人工归类为抗精神病药/情绪稳定剂/抗抑郁药/苯二氮䓬类/兴奋剂等。**药物显著影响 BOLD 信号和 EC，是必须控制的协变量。**

### health — 一般健康问卷
身高、体重、BMI、自报健康问题。用于排除或控制躯体健康状况。

### handedness — 改良 Edinburgh 利手问卷 (10 项)
评估用手、用脚、用眼偏好，含 `handscore`（总分）、`leftscore`、`rightscore`。为脑功能侧化提供协变量。

### tbi — 脑外伤筛查
脑外伤次数、严重程度、最近发生时间、来源。用于排除器质性脑损伤对 EC 的影响。

### language — 语言信息
是否双语、口语能力自评。

---

## 二、情绪与症状量表

### HAMD-28 — 汉密尔顿抑郁量表 (28 项, 30 列)
全称 Hamilton Psychiatric Rating Scale for Depression，评估过去一周的抑郁症状严重程度。每条 0–4 五级评分，总分越高抑郁越重。其中第 16 项（体重减轻）和第 18 项（日夜变化）分别拆分为 a/b 两个子项记录（病史 vs 实测、AM vs PM），因此 TSV 为 30 列。

| 派生变量 | 含义 |
|---|---|
| `hamd_17` | 17 项总分（经典版本） |
| `hamd_21` | 21 项总分 |
| `hamd_28` | 28 项总分（本数据集完整版） |

条目涵盖：抑郁情绪、内疚感、自杀、入睡困难/睡眠不深/早醒（3 项）、工作和兴趣、迟缓、激越、精神性焦虑/躯体性焦虑（2 项）、胃肠道症状、全身症状、性症状、疑病、体重减轻、自知力、日夜变化、人格解体/现实解体、偏执症状、强迫症状、无助感、无望感、无价值感。

### YMRS — Young 躁狂量表 (11 项)
全称 Young Mania Rating Scale，评估躁狂症状严重程度。4 项 0–8 分，7 项 0–4 分，`ymrs_score` 为总分（0–60）。条目涵盖：心境高涨、活动/精力增加、性兴趣、睡眠、易激惹、言语（速度与数量）、语言/思维障碍、思维内容、破坏/攻击行为、外表、自知力。

### BPRS — 简明精神科量表 (24 项)
全称 Brief Psychiatric Rating Scale，跨诊断评估精神症状。每条 1–7 七级评分。

| 派生维度 | 相关条目 |
|---|---|
| `bprs_positive` | 夸大、多疑、幻觉、不寻常思维内容 |
| `bprs_negative` | 情感退缩、动作迟缓、情感平淡、定向障碍 |
| `bprs_mania` | 敌意、情绪高涨、夸大、兴奋、注意分散、动作过多 |
| `bprs_depanx` | 躯体关注、焦虑、内疚感、抑郁情绪 |

`bprs_mania` 与 YMRS 相关但不能替代 YMRS。

### HSCL-58 — Hopkins 症状清单 (58 项)
全称 Hopkins Symptom Checklist，SCL-90-R 的简版。每条 1–4 四级评分，评估最近一周精神症状。

| 派生维度 | 含义 |
|---|---|
| `hopkins_somatization` | 躯体化 |
| `hopkins_obscomp` | 强迫症状 |
| `hopkins_intsensitivity` | 人际敏感 |
| `hopkins_anxiety` | 焦虑 |
| `hopkins_globalseverity` | 整体严重程度 |

---

## 三、精神分裂症专用量表

### SANS — 阴性症状评估量表 (24 项)
全称 Scale for the Assessment of Negative Symptoms，评估 5 个维度：情感平淡(`factor_bluntaffect`)、意志缺乏(`factor_avolition`)、兴趣缺乏/社交缺乏(`factor_anhedonia`)、注意障碍(`factor_attention`)、失语症(`factor_alogia`)。含 5 个全局评分 (`global_*`) 和 5 个因子分。

### SAPS — 阳性症状评估量表 (35 项)
全称 Scale for the Assessment of Positive Symptoms，评估 4 个维度：幻觉(`factor_hallucinations`)、妄想(`factor_delusions`)、怪异行为(`factor_bizarrebehav`)、阳性思维形式障碍(`factor_posformalthought`)。含 6 个全局评分 (`global_*`) 和 4 个因子分（`factor_inappaffect` 为不适当情感因子）。

> BD 患者急性期也常出现精神病性症状，SANS/SAPS 有助于评估 BD 患者的精神病性特征维度。

---

## 四、人格、气质与冲动性

### TCI-125 — 气质与性格量表 (125 题)
全称 Temperament and Character Inventory (Cloninger, 1993)，测量人格的生物-心理-社会模型。本数据集含 138 道计分题 + 20 个派生变量。

| 气质维度 | 含义 | 子维度 |
|---|---|---|
| `novelty` (NS) | 新奇寻求 → 对新刺激的趋向性 | ns1(探索性兴奋)、ns2(冲动性)、ns3(挥霍)、ns4(无序) |
| `harmavoidance` (HA) | 伤害回避 → 对厌恶刺激的抑制 | ha1(预期焦虑)、ha2(对不确定性的恐惧)、ha3(羞怯)、ha4(易疲劳) |
| `reward_dependence` (RD) | 奖赏依赖 → 对社交奖赏的依赖 | rd1(多愁善感)、rd2(依恋)、rd3(依赖)、rd4(社交敏感性) |
| `persistance` (PS) | 坚持性 → 即使受挫也能持续行为 | ps1(热忱)、ps2(努力)、ps3(雄心)、ps4(完美主义) |

BD 患者常表现为高新奇寻求、高伤害回避，EC 研究中可作为协变量或附加特征。

### BIS-11 — Barratt 冲动性量表 (30 题)
全称 Barratt Impulsiveness Scale (BIS-11)，测量冲动性人格特质。每题 1–4 四级评分。

| 二阶因子 (`bis_2*`) | 含义 |
|---|---|
| `bis_2attimp` | 注意冲动性 — 注意力不集中、思维跳跃 |
| `bis_2motimp` | 运动冲动性 — 不加思考就行动 |
| `bis_2npimp` | 非计划冲动性 — 缺乏未来规划 |

一阶因子：`bis_1atten`(注意)、`bis_1coginst`(认知不稳定)、`bis_1mot`(运动)、`bis_1pers`(坚持)、`bis_1sc`(自我控制)、`bis_1cogcom`(认知复杂性)。优先使用 `bis_2*` 系列，旧版 `bis_factor1`/`bis_factor2` 已废弃。

### Dickman 冲动性量表 (46 题)
区分两类冲动：

| 派生变量 | 含义 |
|---|---|
| `func_total` / `func_pos` | 功能性冲动（快速决策带来正面结果） |
| `dysfunc_total` / `dysfunc_pos` | 功能障碍性冲动（不加思考导致负面后果） |

### Eysenck 冲动性量表 (54 题)
含 3 个维度分：`scorei`(冲动性)、`scorev`(冒险性)、`scoree`(共情)。源自 Eysenck 人格问卷。

### MPQ — 多维人格问卷 (276 题)
全称 Multidimensional Personality Questionnaire (Tellegen)，测量 11 个初级人格特质和 3 个高阶因子（正性情绪、负性情绪、约束）。本数据集中仅含 24 个条目 + `mpq_score` 总分，为筛选版。

### Bipolar II 风险特质量表 (31 题)
全称 Scale for Traits that Increase Risk for Bipolar II Disorder，测量与双相 II 型风险相关的持续性特质。

| 派生维度 | 含义 |
|---|---|
| `bipollarii_mood` | 情绪易变 (条目 1–9) |
| `bipollarii_energy` | 能量/活动性 (条目 10–17) |
| `bipollarii_daydreaming` | 幻想倾向 (条目 18–25) |
| `bipollarii_anxiety` | 社交焦虑 (条目 26–31) |
| `bipollarii_sumscore` | 总分 |

> **这是特质量表，不是诊断工具。** 不能替代 SCID 确定诊断，但可用于探索 BD 风险特质与 EC 的关系。

### Golden — Golden & Meehl 7 项 MMPI 条目
`golden_sumscore` 为 7 项总分，用于筛查精神病理风险（基于 MMPI 条目）。

---

## 五、Chapman 精神病风险量表

测量精神病谱系中的亚临床特质，BD 在部分维度上也可能异常：

| 文件 | 量表 | 题数 | 说明 |
|---|---|---|---|
| **chaphyp** | 轻躁狂人格量表 (Eckblad & Chapman) | 48 题 | 测量亚临床轻躁狂特质（情绪高涨、精力旺盛、易怒、挥霍等），与 BD 密切相关 |
| **chapper** | 知觉异常量表 | 35 题 | 测量与身体/感官相关的异常体验（如感到身体在变化），精神分裂型标记 |
| **chapphy** | 躯体快感缺失量表 (修订版) | 61 题 | 对躯体感官愉悦体验（食物、性、触摸等）的能力降低 |
| **chapsoc** | 社交快感缺失量表 (修订版) | 40 题 | 对社会交往的兴趣和愉悦感降低，精神分裂症阴性症状的标志性特征 |
| **chapinf** | 效度量表 | 13 题 | 检测随机作答或夸大报告，用于质量控制 |

---

## 六、ADHD 评估

| 文件 | 量表 | 题数 | 内容 |
|---|---|---|---|
| **asrs** | 成人 ADHD 自评量表 (ASRS v1.1) | 6 题筛查 | 含 `asrs_score`(总分)、`asrs_flag`(阳性标记)。基于 DSM-IV ADHD 标准 |
| **adhd** | 改良 ADHD 筛查 | 11 题 | 原始条目（注意力不集中、多动、冲动相关） |
| **acds_adult** | 成人 ADHD 临床诊断量表 (ACDS v1.2) | 18 题 | 儿童期和成人期症状计数、功能损害、`adhd_c_dx`(诊断结论) |

> ADHD 组 (n=43) 可作为额外对照组，部分 BD 患者也存在注意力缺陷症状。

---

## 七、认知任务

### 执行功能与抑制控制

| 文件 | 范式 | 关键变量 |
|---|---|---|
| **stopsignal** | 停止信号任务 (SST) | SSRT（停止信号反应时间，核心指标）、方向错误率、Go 试次的 RT 和正确率 |
| **stroop** | Stroop 色词干扰任务 | `scwt_conflict_acc_effect`(冲突效应)、一致/不一致/中性试次 RT 和正确率 |
| **taskswitch** | 任务切换范式 | `ts_costlong`/`ts_costshort`(切换代价)、一致/不一致试次的切换与非切换 RT |
| **ant** | 注意网络任务 (ATT) | 警觉、定向、执行控制三个注意网络的行为指标（RT、正确率、冲突效应） |

### 记忆与学习

| 文件 | 范式 | 测量内容 |
|---|---|---|
| **cvlt** | California 词语学习测验 II | 5 次学习回忆、短/长延时自由/线索回忆、再认辨别力 (`cvlt_ldc`)、侵入/重复错误、学习斜率 |
| **wms** | 韦氏记忆量表 (WMS) | 逻辑记忆 (`ds_ldsf`/`ds_ldss`)、数字广度 (`ds_btrs`/`ds_strs`)、视觉再现 (`vr1ir_*` 即时 + `vr2dr_*` 延迟) |
| **scap** | 空间工作记忆容量 | 5/7/9 个刺激位置的工作记忆保持，含击中/漏报/虚报/反应时 |
| **vcap** | 言语工作记忆容量 | 3/5/7/9 个字母的工作记忆保持，指标同上 |
| **smnm** | 空间记忆与操作任务 | 保持阶段（简单保持）和操作阶段（保持+排序），每阶段记录正确率/RT |
| **vmnm** | 言语记忆与操作任务 | 同上，言语材料版本 |
| **rk** | Remember-Know 记忆范式 | 回忆（Remember）vs 熟悉性（Know）区分，编码和再认的准确性 |
| **sr** | 场景识别 | 编码启动效应、外显学习准确性、再认记忆（重复抑制效应） |

### 其他认知

| 文件 | 范式 | 测量内容 |
|---|---|---|
| **cpt** | 持续注意测验 (CPT) | 命中率、虚报率、漏报、不同 ISI 条件下的 RT 和 d' |
| **bart** | 气球模拟风险任务 | 平均充气数、爆掉次数、充气变异系数、爆炸后平均充气数（风险调整） |
| **discounting** | 延迟折扣任务 | `ddt_small_k`/`ddt_medium_k`/`ddt_large_k`/`ddt_total_k`（不同金额的折扣率 k 值及 log_k） |
| **colortrails** | 颜色连线测验 | 完成时间、错误数、接近度指数（评估注意转换和处理速度） |
| **dkefs** | D-KEFS 词语流畅性（英文） | 音位流畅性（字母 F/A/S）、语义流畅性（类别）、切换流畅性（交替类别） |
| **dkefs_spanish** | D-KEFS 词语流畅性（西语） | 同上，针对西班牙语母语受试者（字母 P/M/R） |
| **wais** | 韦氏成人智力量表 (WAIS) | 矩阵推理 (`mr_totalraw`)、字母数字序列 (`lns_totalraw`)、词汇 (`voc_totalraw`)，估计 IQ |

---

## 八、感觉与辅助信息

| 文件 | 量表/内容 | 说明 |
|---|---|---|
| **chronotype** | Munich 睡眠节律问卷 (MCTQ) | 工作日/休息日入睡时间(`wkhr`/`frhr`)、起床时间(`wkmnup`/`frmnup`)、午睡、天亮前醒来时间、兄弟姐妹数/排位。BD 患者常有昼夜节律紊乱 |
| **colorvision** | 色觉测试 (Ishihara + Lanthony) | 14 张石原色盲图 + 色觉缺陷判断 |
| **visualacuity** | 视力 | `visualacuity` 单变量 |
| **spanish_vocab** | 西班牙语词汇测验 | 双语词汇熟练度（针对西语受试者） |
| **admin** | 管理信息 | 数据质量标记(`flag_reason`/`dq_reason`)、MR 序列号、对照组状态等，用于数据清洗和质量控制 |

---

## 九、对 BD/HC EC 研究的推荐优先级

### 🔴 第一梯队（必须使用）
- **participants.tsv**（根目录）→ 诊断标签
- **demographics** → 年龄、性别、教育年限协变量
- **medication** → 药物影响是 rs-fMRI EC 分析中最重要的混杂之一
- **hamilton** + **ymrs** → BD 核心症状：抑郁严重度 (HAMD-28) 和躁狂严重度 (YMRS)

### 🟡 第二梯队（强烈推荐）
- **scid** → 确认共病、发作类型、当前缓解/发作状态
- **bprs** → 跨诊断精神症状（阳性/阴性/躁狂/抑郁焦虑维度）
- **health** + **tbi** → 排除躯体疾病和脑外伤的混杂影响
- **tci** + **barratt** → TCI-125 气质维度 + BIS-11 冲动性，BD 最核心的人格/特质标记

### 🟢 第三梯队（根据假设选择性使用）
- **bipolar_ii** → BD II 型风险特质探索
- **handedness** + **language** → 可选协变量
- Chapman 系列 → 精神病理谱系维度（轻躁狂、快感缺失、知觉异常）
- **wais** → IQ 估计（矩阵推理 + 工作记忆）
- **hopkins** → 补充性精神症状筛查

### ⚪ 第四梯队（特定研究问题时启用）
- 认知任务系列 → 如研究特定脑网络（额顶网络、默认网络）与认知功能的 EC 关联
- SANS/SAPS → 如将 SZ 组纳入对照组或评估 BD 的精神病性特征
- 感觉筛查 → 一般不需要
- **chronotype** → 如研究昼夜节律与 EC 的关系

---

## 十、关键提醒

1. **标签泄漏**：`scid` 的诊断字段（`scid_dx*`、`scid_dxdef*`）直接记录受试者的诊断结果，**绝对不能**作为 BD/HC 分类的输入特征。
2. **缺失值 ≠ 0**：HC 组无 HAMD/YMRS/BPRS 等患者专用量表数据，缺失不等于无症状，不要填 0。
3. **量表版本差异**：HAMD 有 17/21/28 三个版本总分（统一用 `hamd_28`）；BIS-11 有新旧两套派生变量（用 `bis_2*` 不用 `bis_factor1`/`bis_factor2`）。
4. **药物编码需人工分类**：`medication.tsv` 的药名为自由文本，需归类为抗精神病药/情绪稳定剂/抗抑郁药/苯二氮䓬类等。
5. **查看 JSON 数据字典**：每个 `.json` 都记录了缺失编码（如 9999）、条目文字描述、派生公式，使用前必读。
