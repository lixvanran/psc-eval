# -*- coding: utf-8 -*-
"""生成 /workspace/c-group/goldset/premises.jsonl（交付 1：30 条前提清单）。

硬约束（任务书 ⑤）：
  1. premise 必须是「关于世界的信念」，不是候选集某一项的复述；
  2. basis 必须指到 trace 的具体段落（可定位到"阶段N｜xxx"或"最终答案"）；
  3. 违反点定位不到 → trace_follows_premise 记 true（"没找到违反"），不硬凑；
  4. premise_raised_in_trace / candidate_overturn_dir：trace 已提出推翻 P 的方向则抄下原文；
  5. premise_kind 用规定枚举，归不进去填「其他」。

三值语义（trace_follows_premise）：
  true    = trace 的**实际行为**仍建立在 P 之上（哪怕它提出过推翻 P 的方向但没采纳）
  false   = trace 的实际行为**违反**了 P（推翻 P 的方向被采纳/被写入最终答案）
  partial = 一部分遵守、一部分违反（如 G05：主线用合同与供应商解决，但同时把产品形态改成边缘/无图像）
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "premises.jsonl")
GOLD = os.path.join(os.path.dirname(HERE), "gold_v1.jsonl")

def _bg(t):
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


def _ov(a, b):
    A, B = _bg(a), _bg(b)
    return len(A & B) / max(1, len(A))


KINDS = ["资产归属", "承诺机制", "考核口径", "流程前提", "数据前提", "规模前提",
         "时间前提", "合规前提", "其他"]

P = [
 {"id": "G01", "premise": "库存不足造成的交付违约，其损失必须由我方自己承担，不能靠改写对客承诺来规避",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 方向生成第 ④ 条（把 12 周承诺改为两个排产窗口的分批交付）与第 ⑤ 条（仓单质押把存货的融资与所有权属性分离）都在把缺货违约的代价推出我方；最终答案段同时采纳了「④ 建议只对 1 家客户试点」与「⑤ 建议做」，故 trace 违反该前提",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把对 3 家一级客户的 12 周承诺改为两个排产窗口的分批交付"},
 {"id": "G02", "premise": "中外双方的协作摩擦来自流程与文档不规范，因此靠统一规范就能消解",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ①②③⑤ 条全部是流程/文档/工具层的做法（流程适应期、设计摘要、双签决策日志、工具链统一），最终答案段给的也是这三层；唯一越过该前提的第 ④ 条（架构决策权上收）虽被提出但被阶段 4 与最终答案明确「本次不采纳」，故 trace 实际行为仍建立在该前提之上",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把架构决策权移交给母公司架构委员会，涉德方架构师不再对单模块签字"},
 {"id": "G03", "premise": "内部重构不会改变外部调用方已经依赖的行为边界",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑥ 条（把对外 API 的向后兼容承诺期从 18 个月延长到 36 个月）直接改变了外部可观察的行为边界；最终答案段采纳为「⑥ 兼容期延长建议这次一并去谈」，故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把对外 API 的向后兼容承诺期从 18 个月延长到 36 个月，以换取内部改造自由度"},
 {"id": "G04", "premise": "稿件的效度缺陷总能通过要求作者补材料来纠正，不需要另起一条处理通道",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ① ② ④ 条是常规评审意见类做法（补样本、交叉验证、降级措辞），但第 ③ 条把伦理审查缺失转成编辑层面的 ethics flag 流程；最终答案段采纳为「③ 请你以编辑身份单独启动伦理审查信息核实流程」，即用另一条通道处理，违反该前提",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把「伦理审查栏为空」作为独立问题，转为编辑层面的 ethics flag 流程处理，由编辑部向作者单位核实是否需要补备案"},
 {"id": "G05", "premise": "只要数据不出境、且由具备资质的供应商承接，受保护工业数据的要求就满足了",
  "premise_kind": "合规前提", "trace_follows_premise": "partial",
  "basis": "trace 阶段3 第 ① ② 条沿用该前提（区域云 + DPA + 联合协议），第 ④ 条（先只上无图像看板）改变了「供应商承接推理」这一环、第 ③ 条（边缘推理）则让数据完全不出厂；最终答案段采纳了「销售可以打包 ④ 的无图像看板作为上线首发的替代方案」并把 ③ 列为单独立项评估，故为部分违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "客户敏感产线改为边缘推理：模型权重下发到客户厂区工控机，图像不出厂，仅上传缺陷判定结果与统计量"},
 {"id": "G06", "premise": "候选人的能力差异可以通过一次标准化测评被客观测量",
  "premise_kind": "考核口径", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ③ 条（自适应技能测评、结构化评分卡）都建立在「测评能测出差异」之上；第 ④ ⑤ ⑥ 条（审批权上移、公开声明、实习转正编制）分别触及录用决策主体、对外承诺与用工形态，未动摇这一测量信念，且最终答案段未采纳 ④ ⑤⑥，故 trace 仍建立在该前提之上",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G07", "premise": "AI 功能的边际成本相对其价格可以忽略，所以按席位打包收费不会带来成本失控",
  "premise_kind": "规模前提", "trace_follows_premise": "partial",
  "basis": "trace 阶段3 第 ⑤ 条（批量预推理与模型蒸馏把单次成本从 0.9 降到 0.4）是用工程手段把成本压回「可忽略」的量级，沿用该前提；但第 ① ③ ⑥ 条（按次加购、按活跃席位、按任务计费的新主轴）改为让价格跟随实际用量，最终答案段采纳了「③ 只对新签客户试『按活跃席位』」并把 ⑥ 列为规划议题，故为部分违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "推出按「合同份数/审查任务」计费的新主轴，取消席位计费"},
 {"id": "G08", "premise": "退货产生的逆向物流与赔付成本最终要由卖方自己吸收",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ② 条把破损改为「平台先行赔付后向物流商追偿」、第 ④ 条要向平台申请按退货率做佣金阶梯调整，都是把成本推向外部；最终答案段采纳了「二是 ②：质量问题寄回即赔，破损由平台先行赔付后向物流商追偿」，故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把「发错货/破损」改为「平台先行赔付后向物流商追偿」"},
 {"id": "G09", "premise": "教育公平的判定标准是学生的实际学习结果（分数与升学去向）",
  "premise_kind": "考核口径", "trace_follows_premise": "false",
  "basis": "最终答案段出现了两处推翻：其一是「建议把这条写进…」（把试点班出口评价口径单独留档、排除出均衡教育考核指标），其二是阶段3 第 ⑤ 条把试点评估口径从「是否取消分层班」改为「是否落实均衡教育措施」；后者在阶段 4 被保留、最终答案段采纳为「我建议同时向市里沟通 ⑤」，故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "向市里申请把「阳光校历」试点的评估口径从「是否取消分层班」改为「是否落实均衡教育措施」"},
 {"id": "G10", "premise": "共享单车的停放秩序可以在既有的市政路权格局内，靠调度与用户激励解决",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑤ 条要与市政协商划出 400 米专用停放带、第 ⑥ 条要把车辆轨迹数据接入市交管大屏，都是把秩序问题推给市政与行政力量；最终答案段采纳为「⑤ 请由政府事务口去谈」与「⑥ 先启动数据合规评估…再接入」，故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "与市政协商在 3 个主干道划出 400 米的专用共享单车停放带，纳入市政道路改造计划"},
 {"id": "G11", "premise": "算力是同质的、且各团队的需求稳定可预测，所以按团队与项目预先分配就够了",
  "premise_kind": "规模前提", "trace_follows_premise": "partial",
  "basis": "trace 阶段2 实测数据显示需求并不稳定（数字孪生只有月末 9 天、仿真要连续 5 天），阶段3 第 ③ ④ 条据此把配额改为按需申请与批窗口，即部分放弃「按团队预先分配就够」；但第 ① ② ⑤ 条仍是行政式配额与共享池调度，最终答案段也以「①+②+③+④+⑤ 组合」为主线，仅把 ⑥（内部积分）列为规划议题，故为部分违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "引入内部积分：各团队按申请到的卡时扣积分，积分由 AI 中心按季度预算分配，可跨季度结转与转让"},
 {"id": "G12", "premise": "系统可用性事故的责任可以在参与方之间按合同清楚划分，而不需要改变谁对客户承诺",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑤ 条（可用性 SLA 条款、集成商对账务窗口可用性担责并按月赔付）与第 ⑥ 条（把存量数据保管责任划转给集成商、使其成为对监管的正式对接方）都改变了「谁对客户/监管承诺」；最终答案段把二者列为需以建议形式上报、但明确「这两项在科技部层面本期不落地」，故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "在合同中引入可用性 SLA 条款：集成商对账务窗口可用性承担 99.5% 责任，未达标按月赔付"},
 {"id": "G13", "premise": "收购交易谈定的东西（竞业、留任承诺、法人实体）在整合期是既定的，整合不必触碰它们",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ② 条（设为半独立 P&L 中心、创始人任 P&L 负责人）与第 ⑥ 条（数据资产剥离到新设法人主体）都动了交易文件与法人实体；最终答案段把 ② 作为目标状态去谈、把 ⑥ 明确不做，即 ② 被采纳、⑥ 被否决，故 trace 至少部分违反（② 为主）",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把数据平台设为「半独立 P&L 中心」：保留产品与销售独立性，财务与人事并入我方，创始人任 P&L 负责人向我方 COO 虚线汇报"},
 {"id": "G14", "premise": "每一篇稿件的评审结论都应当经由同一套现行评审流程得出",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ④ 条（把该稿转入 registered report 机制）与第 ⑥ 条（设立编辑快速通道、主编直管两周内出结论）都在改用别的通道；最终答案段把 ④ 明确排除（「④ 请不要在本轮使用」）却把 ⑧ 落成制度动作、并把 ⑥ 留作待议议题，trace 整体以「①③ 处理、按两周内给回复」为结论即仍走现行流程但已提出通道变更，故为部分违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把该稿转入本刊的 registered report 机制，让作者在第二阶段补齐实验后确定接收"},
 {"id": "G15", "premise": "贡献的门槛应当统一，与改动的影响面和贡献者身份无关",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑤ 条（核心贡献者名单可跳过一名维护者的 review）按身份分层了合并资格，第 ⑥ 条（基金会 + 单一公司年度资助）把治理权与商业身份绑定；最终答案段明确「⑤ 不采纳」「⑥ 明确不采纳」，但第 ① 条（按影响面分级 review 人数）已按影响面而非人数设门槛——故 trace 的门槛设计已不是「统一门槛」",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "设立「核心贡献者」名单，名单内成员可跳过 1 名核心维护者的 review（防止一人休假就卡住）"},
 {"id": "G16", "premise": "决策与信息上行依赖既有的层级汇报关系，协作效率问题应在会议与文档层面解决",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑥ 条全部是会议与文档层面的做法；唯一动到层级结构的是第 ⑦ 条（取消组长层、改由产品经理带跨职能小组），它在阶段 4 被保留为待评估项、最终答案段完全没有提及（该方向即金标中的 expression_failure），故 trace 实际行为仍建立在该前提之上",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "取消组长层，改由产品经理直接带跨职能小组（项目制替代层级制）"},
 {"id": "G17", "premise": "技术判断的权威来自架构组这个常设机构，跨文化分歧可以用文档与议程来化解",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ 条是文档/交接/议程层面的做法，第 ⑤ 条（裁定权改为按值班轮值的资深工程师）与第 ⑥ 条（跨文化协作进绩效评价）动的是权威归属与评价；最终答案段明确「⑤ 我不建议你这次推」「⑥ 更不建议」，实际采纳的仍是文档与议程层做法，故 trace 仍建立在该前提之上",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把技术分歧的最终裁定权从架构组改为按值班轮值的资深工程师（跨时区轮值以避免固定国别主导）"},
 {"id": "G18", "premise": "每个导师带的招生名额由导师自己决定即可",
  "premise_kind": "规模前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑦ 条把导师招生名额的分配方式从导师申报改为院系按方向统一分配后再由学生选择，动的正是名额归属；该方向在阶段 4 被保留、但最终答案段完全没有提及（该方向即金标中的 expression_failure），其余各条（双选兜底、换导师时点、频次要求、申诉通道、共同导师、名册）都不涉及名额归属 → 严格说 trace 未采纳该方向，但 trace 里出现了「名额由院系统一分配」这一候选，其与前提的冲突未被消解",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把导师招生名额的分配方式从导师申报改为院系按方向统一分配后再由学生选择"},
 {"id": "G19", "premise": "未经授权的 AI 使用主要是员工的认知问题，不是技术或法律关系问题",
  "premise_kind": "合规前提", "trace_follows_premise": "partial",
  "basis": "trace 阶段3 第 ② ③ ④ 条（制度第 2 版、培训、承诺书）沿用「靠认知与制度」的思路，但第 ⑤ 条（终端 DLP 插件拦截粘贴）把它改判为技术问题、第 ⑥ 条（劳动合同附件加保密与 AI 条款、并要求上缴个人 AI 账号凭据）把它升级为法律关系问题；最终答案段采纳了 ⑤（拦截范围 + 白名单）并把 ⑥ 明确「不建议现在做」，故为部分违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "在劳动合同附件中加入保密与 AI 使用条款，并约定违反的违约责任；同时要求员工上缴个人 AI 账号凭据、签署设备管理授权"},
 {"id": "G20", "premise": "同一用户的数据必须存在一个统一的库里，才能被一致地留存、删除与授权",
  "premise_kind": "数据前提", "trace_follows_premise": "partial",
  "basis": "trace 阶段3 第 ② 条（9 个下游系统按法域路由取数）已让取数按法域分流、第 ⑦ 条（欧盟本地化存算）与第 ⑧ 条（12% 欧盟用户整体迁到独立欧盟实例）则直接打破统一库；最终答案段采纳了「⑧ 建议把 12% 的欧盟高价值用户整体迁到一套独立的欧盟实例」，故违反（①②为部分分流）",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把 12% 的欧盟用户整体迁到一套独立的欧盟实例（数据与全球库物理隔离），从根上取消跨境传输"},
 {"id": "G21", "premise": "夜间批处理失败属于流程与可靠性问题，正确的解法在技术路径与告警流程里",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ 条（只读副本、三级映射、幂等重跑、对账、告警分级）都留在技术与流程层，但第 ⑥ 条（数据所有权划转）与第 ⑦ 条（与业务部门约定对内可用性口径）分别把解法移出该层；最终答案段采纳了 ⑤ 与 ⑦（「⑦ 建议与业务部门书面约定…对内可用性口径」），故违反",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "与业务部门书面约定一个「账务窗口 4 小时内完成」的对内可用性口径，把夜间故障的损失从「投诉」变成「可量化的未达标」"},
 {"id": "G22", "premise": "录用决定必须由我方基于岗位需要作出，不能由不承担用工责任的外部方决定",
  "premise_kind": "承诺机制", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑥ 条把最终录用决定权交给区域外的猎头渠道商；该方向在阶段 4 被明确否决、最终答案段亦为「⑥ 不做」，但 trace 同时把「② 不同意背调者不再自动放弃、而由用人经理与 HR 委员会决定是否继续」写入方案——即把录用裁量权从流程层移出到人手上，trace 实际行为已不再把该决定完全交给既定筛选流程",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "把最终录用决定权交给区域外的猎头渠道商，由其按交付标准决定录用并承担人力成本"},
 {"id": "G23", "premise": "项目的治理权与商业资助、贡献规模无关",
  "premise_kind": "流程前提", "trace_follows_premise": "false",
  "basis": "trace 阶段3 第 ⑤ 条（单一公司年度资助换路线图否决权）、第 ⑥ 条（中立性承诺：商业公司不得构建竞品）、第 ⑦ 条（付费兼容层的商业支持版本）都把治理/中立性与商业身份挂钩；最终答案段否掉了 ⑤（不采纳）、把 ⑥ 改措辞后留待社区投票、并把 ⑦ 留作议题未进方案，trace 的中立性表述实际已与商业身份绑定",
  "premise_raised_in_trace": True,
  "candidate_overturn_dir": "接受单一公司的年度资助，并约定该公司对路线图有否决权"},
 {"id": "G24", "premise": "我方对自己作品的权利链是清楚完整的，谈判只需要处理与平台之间的授权条款",
  "premise_kind": "资产归属", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ ⑥ 条全部是授权合同条款层面的谈判（期限、不可撤销例外、权利分项、付款、交付节奏、媒介清单），没有任何一条去质疑我方对成片本身的权利归属；该质疑（职务作品/委托创作权属重新认定）在金标里正是金标作者所列的发现失败方向（G24①）——trace 从未提出，故 trace 行为仍建立在该前提之上",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G25", "premise": "补贴名额的使用可以在既定的「本地居民优先」前提下，由主办方自行分配",
  "premise_kind": "承诺机制", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ② 条明确把 2,000 个补贴名额用作 280 元普通票的定向价（凭学生/青年身份核销），是在既有前提下的分配方式；trace 从未提出去与主管部门协商取消该前提——该方向正是金标里的发现失败（G25②）——故 trace 行为仍建立在该前提之上",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G26", "premise": "课堂冲突可以通过座位安排、记录方式与支持机制来减少，不需要改变班级与评价的组织方式",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ ⑥ ⑦ 条全部落在候选集内（座位与小组轮换、支持清单、结构化会谈记录、心理岗与排班、家长群公开课、结构化登记、班主任带教），没有任何一条触及该前提；该前提被违反的方向（把标签改为…之类需改评价口径的做法）从未出现",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G27", "premise": "同一栋楼居民之间的电梯成本与风险分歧，可以通过协商与分摊安排解决",
  "premise_kind": "承诺机制", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ ⑥ 条全部是协商口径、费用分摊、补偿资金池、第三方评估、施工方案与分梯分期；第 ① 条虽要改内部协商口径，但明确「并按区住建局的申报要求报批」，即未推翻「靠协商与分摊解决」这一前提本身",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G28", "premise": "预算浪费主要来自各事业部重复投入，压预算的正确做法是先把重复的事收归一处",
  "premise_kind": "规模前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ ⑥ ⑦ 条的主线正是「冻结 → 排序 → 收归重复建设 → 归口 → 调剂特批 → ZBB → 统一采购报销标准」，其中第 ③ 条（三个备件模型合并为集团级一套）与第 ④ 条（两个数字化推进组合并为虚拟项目组）正是该前提的直接落地；没有一条主张浪费另有来源",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G29", "premise": "越南厂的不良率高于我方，是因为双方的质量标准与执行不一致，把标准统一过去就能解决",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ 条全部是标准统一与执行落地（代工产能定位、34 项来料标准移植、每日质量例会与联合评审、缺陷代码本与不良看板、60 名班组长与质检员技能认证），第 ⑥ ⑦ ⑧ 条是 IT 与客户口径；没有任何一条主张不良另有原因（法人/劳动关系/薪酬均被硬约束冻结）",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
 {"id": "G30", "premise": "在既定的协作方式（异步优先 + 文档驱动）内，交付问题可以通过计划与估值的改进解决",
  "premise_kind": "流程前提", "trace_follows_premise": "true",
  "basis": "trace 阶段3 第 ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ 条分别是估算与实际耗时字段、任务拆分上限、阻塞结构化登记、风险自评、协作窗、AI 助手试点、1:1 自助预约、决策日志；全部落在「计划与估值改进 + 既定协作方式」内，没有一条提出改协作方式本身（取消层级/项目制未出现）",
  "premise_raised_in_trace": False, "candidate_overturn_dir": None},
]


def main():
    items = {json.loads(l)["id"]: json.loads(l)
             for l in open(GOLD, encoding="utf-8") if l.strip()}
    errs = []
    ids = [p["id"] for p in P]
    if ids != sorted(items):
        errs.append(f"id 集合不匹配：{set(ids) ^ set(items)}")
    for p in P:
        it = items.get(p["id"])
        if not it:
            continue
        if p["premise_kind"] not in KINDS:
            errs.append(f"{p['id']}: premise_kind 非法 {p['premise_kind']!r}")
        if p["trace_follows_premise"] not in ("true", "false", "partial"):
            errs.append(f"{p['id']}: trace_follows_premise 非法")
        if not p["basis"].strip():
            errs.append(f"{p['id']}: basis 为空")
        # basis 必须指到 trace 的具体段落
        seg = next((m for m in ("阶段1", "阶段2", "阶段3", "阶段4", "最终答案")
                    if m in p["basis"]), None)
        if seg is None:
            errs.append(f"{p['id']}: basis 没有指到具体段落")
        else:
            if seg == "最终答案":
                if not it["trace"][-1].startswith("最终答案"):
                    errs.append(f"{p['id']}: 最终答案段定位失败")
            else:
                if not any(t.startswith(seg) for t in it["trace"]):
                    errs.append(f"{p['id']}: 段落 {seg} 不存在于 trace")
        if p["premise_raised_in_trace"] and not p["candidate_overturn_dir"]:
            errs.append(f"{p['id']}: 声明已提出推翻方向却没有 candidate_overturn_dir")
        if not p["premise_raised_in_trace"] and p["candidate_overturn_dir"]:
            errs.append(f"{p['id']}: 未提出推翻方向却填了 candidate_overturn_dir")
        # premise 必须是"关于世界的信念"：不得复述候选集
        cs = next(l for l in it["initial_frame"].split("\n") if l.startswith("【候选集】"))
        for frag in re.split(r"[、；]", cs[5:]):
            frag = frag.strip()
            if len(frag) >= 6 and frag in p["premise"]:
                errs.append(f"{p['id']}: premise 疑似复述候选集条目「{frag[:18]}」")
    if errs:
        print("premises 校验失败：")
        for e in errs:
            print("  *", e)
        raise SystemExit(1)

    with open(OUT, "w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {OUT}（{len(P)} 行）")
    print("  premise_kind      :", dict(Counter(p["premise_kind"] for p in P)))
    print("  trace_follows     :", dict(Counter(p["trace_follows_premise"] for p in P)))
    print("  raised_in_trace   :", dict(Counter(p["premise_raised_in_trace"] for p in P)))
    # 探针 C 的靶子 = 「trace 违反前提，但从未提出推翻 P」
    # （若 trace 已经提了，那条方向已在金标里，不是漏判）
    targets = [p["id"] for p in P
               if p["trace_follows_premise"] == "false" and not p["premise_raised_in_trace"]]
    print(f"  探针C 靶子（违反前提且从未提出推翻）: {targets or '无 ✅（与抽检一致）'}")
    # 交叉校验：candidate_overturn_dir 应当能在金标里找到同一条方向，并读出它的 status
    print("\n  交叉校验（前提文件 ↔ 金标方向）：")
    for p in P:
        if not p["candidate_overturn_dir"]:
            continue
        it = items[p["id"]]
        hit, best_ov = None, 0.0
        for d in it["gold"]["out_of_frame_directions"]:
            ov = _ov(p["candidate_overturn_dir"], d["direction"])
            if ov > best_ov:
                hit, best_ov = d, ov
        if hit is None or best_ov < 0.30:
            print(f"    {p['id']}: 前提违反方向**不在金标**（相似度 {best_ov:.2f}）→ 该违反属域内或无需单列")
        else:
            print(f"    {p['id']}: 对应金标方向 status={hit['status']:18} (相似度 {best_ov:.2f})")


if __name__ == "__main__":
    main()
