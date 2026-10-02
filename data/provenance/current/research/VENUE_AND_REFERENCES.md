# Applied Soft Computing 期刊与新证书路线引用核查

核查日期：2026-10-02。任务阶段：重设计后的方法、近邻和期刊准备。本文没有新增训练、实证结果或录用概率判断；只整理新英文稿可用的期刊依据、来源状态和准确引用。研究总结保存于 D 盘；本轮未新增原始数据或论文下载缓存。

## 调研结论

**期刊方向匹配，核心贡献须集中在可验证的新能力。** Applied Soft Computing 的官方主页及作者指南已索引内容支持“解决实际问题的软计算”“Machine and Deep Learning”“Signal or Image Processing”；最合适的类型是技术研究论文。2025 年该刊已经发表将早退集成、不确定性估计、概率校准和手机真实延迟结合的 HAR 论文，新稿不能以这些组成部分本身主张创新。[官方主页](https://www.sciencedirect.com/journal/applied-soft-computing)、[官方作者指南](https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors)、[Liu 等，2025](https://doi.org/10.1016/j.asoc.2025.113861)。

**目前可保留的贡献定位是：** 在部分信号特征尚未计算时，利用冻结树集成完整输出的可靠包络，返回对每个案例、每个可能标签均满足相对完整预测器半 Brier 损失容忍的概率；然后检验其在包含特征处理、包络更新、证书计算和路由开销的完整时间上是否有效。这是待验证的组合能力，不能改写为首次风险控制、首次树集成安全早停、首次共享计算、首次最小最大概率预测或首次 HAR 早退。[Fast yet Safe](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)、[Jang 等](https://doi.org/10.3390/math14132343)、[Zhao 等](https://cs.emory.edu/~lzhao41/materials/papers/R1450p-zhaoA.pdf)、[Vovk 等扩展工作稿](https://alrw.cs.rhul.ac.uk/articles/13.pdf)。

**近邻保证对象必须分开。** Fast yet Safe 给出校准抽样假设下的分布风险控制；Jang 的剩余容量证书保留完整树集成的最终类别；Emde 等认证已知标签样本上的对抗校准指标；本文拟认证任意标签的相对完整模型概率损失。新保证并不证明完整预测器准确、概率校准、类别完全一致或硬实时满足。[Fast yet Safe，§2–3](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)、[Jang 等，§4.4.3](https://www.mdpi.com/2227-7390/14/13/2343)、[Emde 等，§3.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/c50f8180ef34060ec59b75d6e1220f7a-Paper-Conference.pdf)。

本轮调研限定四个问题：官方范围与文章类型是否适合；期刊内最近相关稿件已经做到什么；最近安全早退、共享特征处理与最小最大概率预测的原文保证是什么；哪些引用可以直接支持新稿。使用内置网页检索、Exa 官方页面索引与 Jina/原文阅读交叉核查。不是系统综述，也不是首创的穷尽性检索证明。

## 来源状态与官方期刊信息

来源状态记号如下：F 表示取得全文并读到表中指定关键段，不等于逐字读完全部附录；I 表示取得出版商或作者的已索引摘要、方法片段或仓库说明，未取得完整论文；M 表示只核验正式题录、摘要或元数据。前轮已经阅读的材料单独标明，不把本轮仅打开摘要写成全文复核。

| 信息 | 已核实内容 | 来源与访问限制 |
|---|---|---|
| 官方范围 | 官方主页索引说明，该刊通过软计算解决实际问题，并强调软计算方法的综合视角；作者指南索引明确出现机器学习、深度学习、信号或图像处理。足以支持本文算法与传感信号计算应用的方向匹配。 | [官方主页](https://www.sciencedirect.com/journal/applied-soft-computing)；[官方范围页](https://www.sciencedirect.com/journal/applied-soft-computing/about/aims-and-scope)；[官方指南](https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors)。直接网页访问返回 403；Jina 返回机器人验证。依据为官方页面索引，不声称本轮已取得完整当前指南。 |
| 文章类型 | 已索引官方指南列出 technical papers、state-of-the-art survey papers、commentaries 和 letters。本任务应走 technical research/full-length research article；近期目标刊正式文章的出版商元数据也标为 Research article。 | [官方指南](https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors)；[近期研究文章](https://www.sciencedirect.com/science/article/pii/S1568494625011743)。主页还介绍承接 letters 的姊妹刊 Systems and Soft Computing，因此不据旧索引单独推断当前 letters 投稿资格。 |
| 本稿的期刊意义 | 特征处理、选择器和认证本身可能消耗节省的推理时间；需要在具有主体泛化问题的 HAR 与工业信号上证明一个实际可用的算法能力。该刊近期 HAR 稿件已关心真实设备延迟，因此仅报告特征数量或理想 FLOPs 不足。 | [Liu 等，2025](https://doi.org/10.1016/j.asoc.2025.113861)；[Yao 等，2024](https://doi.org/10.1016/j.asoc.2024.112444)。这是基于近期内容的期刊匹配推断，不是编辑接收承诺。 |

未使用第三方期刊转载页代替完整当前官方指南；未查询或引用影响因子、分区、接收率或精确录用概率。正式投稿时仍须重新打开可访问的官方作者指南确认实际提交入口与声明要求；该访问限制不妨碍继续完成正文和实验。

## 近五年期刊内最相关稿件

近五年的检索时间窗按 2021-10-02 至 2026-10-02 处理。以下是与新路线有直接支持关系的选择性样本，不为了数量列举不相关架构论文。作者、年份、卷和文章号均与 DOI/Crossref 或出版商题录交叉核对。未取得全文的稿件只用于其已索引摘要明确支持的主张。

| 编号与正式题录 | 已取得内容与实际支持 | 新稿的处理 |
|---|---|---|
| V1. Xin Liu, Lei Zhang, Wenbo Huang, Dongzhou Cheng, Hao Wu, Aiguo Song. Fixing deep early exit ensembles for sensor-based human activity recognition through uncertainty quantification. Applied Soft Computing 184 (2025), 113861. DOI: 10.1016/j.asoc.2025.113861. | I：出版商索引摘要、highlights 和引言片段；M：出版商/Crossref 题录。描述权重共享早退集成、最后一层 Laplace 近似和温度校准；使用 UCI HAR、UniMiB SHAR、WISDM；报告理论计算量和手机真实延迟。未取得完整正文。 | 必须进入相关工作。其已核验摘要支持“早退 HAR 已结合不确定性、校准和实际延迟”。不能凭未取得全文断言它绝无任何理论结果；具体对比应写保证对象与本文可靠包络所推出的逐标签损失边界。[出版商](https://www.sciencedirect.com/science/article/pii/S1568494625011743)、[DOI](https://doi.org/10.1016/j.asoc.2025.113861)。 |
| V2. Minghui Yao, Dongzhou Cheng, Lei Zhang, LiangDong Liu, Shuangteng Song, Hao Wu, Aiguo Song. A sparse diverse-branch large kernel convolutional neural network for human activity recognition using wearables. Applied Soft Computing 167 (2024), 112444. DOI: 10.1016/j.asoc.2024.112444. | I：出版商摘要和 highlights；M：Crossref 题录。SLKNet 结合大卷积核、多分支结构和动态稀疏性，在 UCI HAR、WISDM、UniMiB、USC-HAD 等验证，并测量移动设备延迟。未取得完整正文。 | 可支持可穿戴 HAR 的计算效率意义，不是新证书的直接技术先例。若篇幅有限，优先 V1。[出版商](https://www.sciencedirect.com/science/article/pii/S1568494624012183)、[DOI](https://doi.org/10.1016/j.asoc.2024.112444)。 |
| V3. Chaojie An, Qifeng Zhou, Shen Yang. A reinforcement learning guided adaptive cost-sensitive feature acquisition method. Applied Soft Computing 117 (2022), 108437. DOI: 10.1016/j.asoc.2022.108437. | I：出版商摘要、方法章节片段和问题表述；M：Crossref。AdaptRADIN 结合 RNN 和强化学习，针对个体逐步决定成本敏感的可变长度特征获取；不是固定全样本同一子集。未取得本轮完整正文。 | 保留原稿此引用，支持目标刊对自适应获取与成本建模的兴趣。将外部特征查询价格和本文实测处理时间区分开；不能据摘要说其含有本文式每案例概率损失证书。[出版商](https://www.sciencedirect.com/science/article/pii/S1568494622000163)、[DOI](https://doi.org/10.1016/j.asoc.2022.108437)；复现 capsule 为 [10.24433/CO.5735560.v1](https://doi.org/10.24433/CO.5735560.v1)。 |
| V4. Haoran Su, Jinkun Chen, Yaojin Lin. A four-stage branch local search algorithm for minimal test cost attribute reduction based on the set covering. Applied Soft Computing 153 (2024), 111303. DOI: 10.1016/j.asoc.2024.111303. | I：出版商摘要与部分章节；M：Crossref。将最小测试成本属性约简转化为集合覆盖并作分支局部搜索；已索引实验说明使用模拟成本。未取得完整正文。 | 可用于范围准备或静态成本约简背景；不必强塞新稿。其优化单位与逐案例真实处理不同。[出版商](https://www.sciencedirect.com/science/article/abs/pii/S1568494624000772)、[DOI](https://doi.org/10.1016/j.asoc.2024.111303)；capsule [10.24433/CO.6025040.v1](https://doi.org/10.24433/CO.6025040.v1)。 |

出版商/Crossref 将 V1 标为 2025 年、卷 184；个别聚合索引显示 2026 年日期，正式引用采用出版商/Crossref 年份，不采用聚合站日期。该刊 2021 年 8 月的 Class-specific early exit design methodology for convolutional neural networks（DOI 10.1016/j.asoc.2021.107316）早于本次严格五年窗口，可在需要时作为早期相关稿，但不计入上表近期论文。[V1 DOI](https://doi.org/10.1016/j.asoc.2025.113861)、[2021 论文 DOI](https://doi.org/10.1016/j.asoc.2021.107316)。

## 证书路线最近邻：实际保证与主张边界

| 编号 | 原文定位、取得状态与实际保证 | 可支持的对比；禁止借此宣称的创新 |
|---|---|---|
| C1. Jazbec 等，Fast yet Safe | F：正式 NeurIPS 2024 论文及关键附录。§2 的校准集 Dcal 是从 P 独立同分布抽样；式 (4)–(5) 讨论期望或高概率总体风险控制。§3.2 式 (6) 明确使用早退与完整模型的性能差，式 (7) 包含多类 Brier；§3.3 使用 conformal risk control 或上置信界选择阈值；附录 A.3 处理相对损失可为负的情形。 | 相对完整模型的 Brier 风险和安全早退均已有直接先例。本文的差异是从输入完成集推出每案例、每可能标签的确定性损失增量，而不是用校准样本估计人口风险。distribution-free 不等于不需要抽样假设或自动覆盖测试主体分布变化。[正式全文](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)。 |
| C2. Emde 等，ICLR 2025 | F：正式全文 §3。条件 C1 为允许扰动下预测类别保持不变，C2 为最高类别置信度具有上下界。§3.1、Theorem 3.3 给出已知标签样本上的最坏 top-label Brier score；相关章节还处理 ECE 的认证或近似认证。Theorem 3.3 的正确性向量 c 是已知样本标签构造的量。 | 不能声称首次认证 Brier 或首次概率质量证书。本文全多类概率、未知标签上的相对损失，与其扰动范围内、已知标签集合上的 top-label 校准评价不是同一目标。本文相对损失证书也不自动认证 ECE。[正式全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c50f8180ef34060ec59b75d6e1220f7a-Paper-Conference.pdf)。 |
| C3. Jang 等，Mathematics 2026 | I：已索引官方 §3、§4.4.3、Proposition 1、§7，另读作者仓库。直接全文页返回 429，PDF 链接未取得，不能称已通读全文。LazyGBM 的当前类别差 Δt 与剩余容量 Rt 满足 Δt > 2Rt 时，完整集成不会改变最终 argmax，这是 exact/lossless 模式。LazyRF 的 Dirichlet 投票及其他 heuristic 模式不能一并称为该确定性保证。 | 安全剩余容量早停已有直接近邻。本文拟对部分未计算特征的完整概率包络给出相对 proper loss 容忍，而不是仅保持 argmax。Jang 主要评估树工作量或提前停止收益，不能将工作量等同任意实现的墙钟收益。该文证书模式和启发式模式必须分列。[官方索引来源](https://www.mdpi.com/2227-7390/14/13/2343)、[作者仓库](https://github.com/pz1004/lazy_inference)。 |
| C4. Zhao 等，KDD 2018 | F：作者托管正式论文，§3–4。联合优化全局选定特征子集与分类器；用计算依赖图/超图表示可共享的中间处理组件，T(U;G) 对共同组件只计一次，并包括提取与预测相关代价。其均值、标准差、偏度、峰度例子直接涉及统计信号特征的共享计算。 | 共享处理 DAG、非加性特征成本均不可宣称新。差异须建立于案例自适应的部分求值、可靠概率包络和停止证书；不能仅把共享 FFT/统计量包装为贡献。[作者全文](https://cs.emory.edu/~lzhao41/materials/papers/R1450p-zhaoA.pdf)、[KDD 官方接受页](https://www.kdd.org/kdd2018/accepted-papers/view/prediction-time-efficient-classification-using-feature-computational-depend)。 |
| C5a/C5b. Vovk、Petej、Fedorova | F：正式 NIPS 2015 九页论文和扩展工作稿关键段。正式论文 §4 使用最小最大准则合并 Venn–Abers 概率，主要正文推导 log loss；Brier 推导指向扩展工作稿，后者 §4 给出二元 Brier 合并规则。正式与扩展稿标题不同，扩展稿封面注明初稿 2015、最后修订 2019。 | 最小最大 proper-loss 概率合并与二元 Brier 合并已有先例。本文可能的新部分是多类有界单纯形包络的具体解与部分特征计算应用，不能声称首次最小最大概率输出或继承 Venn–Abers 的校准有效性。[正式论文](https://papers.neurips.cc/paper_files/paper/2015/file/a9a1d5317a33ae8cef33961c34144f84-Paper.pdf)、[扩展工作稿](https://alrw.cs.rhul.ac.uk/articles/13.pdf)。 |
| C6. Lucchese 等，QuickScorer | F：作者机构托管正式十页论文，§3.1 候选退出叶集合、位向量与 Theorem 1；§3.2 按特征排序并交错处理阈值以改善缓存与控制流。其目标是精确完整树集成求值，文中也讨论与早停策略兼容。 | 可达叶剪枝、按特征批量处理树节点与位集本身已有基础。本文应对未计算信号块保留所有允许分支，认证概率损失增量，而非将精确树执行优化改名。QuickScorer 官方仓库说明原源码需与 Tiscali 签 NDA；自写实现必须称为 reimplementation/inspired control，不得称复现原作者公开实现。[机构全文](https://iris.cnr.it/bitstream/20.500.14243/303247/1/prod_342594-doc_107179.pdf)、[官方仓库](https://github.com/hpclab/quickscorer)。 |
| C7. Bolukbasi 等，AAAI 2017（可选） | F：前轮取得并读关键段。资源约束结构预测已考虑特征代价、中间推理、策略选择和 anytime 执行的开销。 | 完整时间闭合是必要公平评价，不是独立首创。若成本讨论已经由 C4 和实际实现解释充分，可以不再添加此引用。[官方全文](https://ojs.aaai.org/index.php/AAAI/article/download/10917/10776)。 |

证书新稿应明确半 Brier 定义为 L(p,y)=0.5||p−ey||²。与使用未除以二的多类 Brier 或二元不同缩放的旧稿比较时，应显式换算容忍 δ；不能直接复制阈值。严格适当评分规则支持概率预测评价，但“proper”不是某个训练模型已校准的证明。[C1](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)、[F2，Example 1](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。

## 原稿中值得保留的其他引用

| 引用 | 核验状态与相关工作支持关系 |
|---|---|
| A1. Gadgil、Covert、Lee，DIME | 本轮 M：ICLR 2024 正式题录和摘要；前轮近邻核查已读关键方法段。条件互信息动态获取、掩码预测器与预测损失差监督已存在。适合作为学习式 AFA 背景或适用强基线；不据其存在或不出现某术语证明本文首创。[正式来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html)。 |
| A2. Valancius、Lennon、Oliva，AACO | 本轮 M：PMLR 正式题录与 BibTeX；前轮读 §3.5 及附录 G 等。非贪心获取、终端子集损失、成本权衡、低开销学生/策略近似和运行时间讨论均与旧门控路线重叠。新稿如实测采用它，按相同数据和成本条件分列。[正式来源](https://proceedings.mlr.press/v235/valancius24a.html)。 |
| A3. Norcliffe 等，SEFA | 本轮 M：PMLR 正式题录；前轮读方法与不确定性编码部分。用于不可见特征的随机编码和非贪心 AFA；不将其潜变量采样误写成每个特征完成值的确定性包络。[正式来源](https://proceedings.mlr.press/v267/norcliffe25a.html)。 |
| A4. Huang、Dinh、Oliva，TAFA | 本轮 M：arXiv 版本页；前轮读信息模板、学生策略与成本段。当前明确为 arXiv 2508.18380v2，2026-02-04 修订，原始 2025-08-25；本轮未发现正式出版信息。若作为基线/前沿讨论，必须标预印本，不添加不存在的期刊卷页。[版本页](https://arxiv.org/abs/2508.18380)。 |
| F1. Brier，1950 | 本轮 M：题录与 DOI，未取得原始全文。用于指标历史出处；当前多类 proper-loss 理论更直接由 F2 原文支持。[DOI](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)。 |
| F2. Gneiting、Raftery，2007 | F：作者全文 §3、Example 1。多类 Brier 严格适当性及平方距离散度；正式期刊是 JASA，DOI 10.1198/016214506000001437，避免混用其他 DOI。[作者全文](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。 |
| F3. Guo 等，2017 | 本轮 M：PMLR 题录与摘要。若模型实际使用冻结温度缩放，可支持经验后处理校准；不得将温度缩放当作逐案例风险保证。[正式来源](https://proceedings.mlr.press/v70/guo17a.html)。 |

原稿的调度、有限计算理性、蒸馏、拒绝专家及其他数据集引文只在新正文真实讨论相应问题时保留。FTTT 或 conformal thinking 可解释旧研究路线，但不宜挤占树输出认证的直接近邻。参考文献按支持关系选取，无需把旧稿全部引用搬入或凑够某个数量。

## 数据来源与完整时间边界

**Hydraulic：** UCI 官方数据含 2,205 个 60 秒循环，三个采样率的过程传感矩阵，并列出物理传感器与 CE、CP、SE 等虚拟量。这里可从提供的循环矩阵重算统计/频谱块；虚拟量不能自动当作无需计算的原始传感器。作者与 DOI 使用官方引用建议；页面捐赠时间与原研究年份不能混同。[D1 官方完整说明](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)。

**HAR：** 官方 240 数据集由 30 名志愿者产生，官方 train/test 按主体分离；50 Hz、128 点、2.56 秒窗口、50% 重叠。提供的惯性窗口已经做过去噪和分窗，body acceleration 已通过滤波分离重力。因此本文可称“从提供的惯性窗口重算信号特征”，不能声称测量了原始连续信号的全部预处理时间。若没有重跑去噪和分窗，完整时间边界写为 provided inertial windows → recomputed feature blocks → returned probabilities；不包括传感采集和发布前预处理。官方 561 维表是已有特征，不能称原始惯性数据。[D2 官方完整说明](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)。

每案例证书可以在新主体下继续成立，前提是完整预测器语义及可靠包络仍正确；这不意味着新主体上的完整预测器准确率不下降。HAR 的主体隔离训练、调参和测试仍然必要。数据集官方提供的切分是主体泛化证据设计的一部分，不是分布稳定性的数学证明。[D2](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)、[C1 的抽样条件](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)。

## 可用于英文新稿的 related-work 段落

以下是基于已核验支持关系的正文素材，引用编号对应下方书目；没有加入待运行的性能数值。原稿若实际未复现某 AFA 方法，可讨论其方法定位，但不能在实验段把它写成已完成比较。

Cost-sensitive feature acquisition learns which information to request for an individual prediction. Reinforcement-learning acquisition has been studied in this journal [V3], while recent methods use conditional mutual information [A1], acquisition-conditioned oracles [A2], stochastic encodings [A3], or information templates [A4]. These approaches provide relevant acquisition baselines when their observation model and cost assumptions match the task. A separate line represents computational dependencies among features: Zhao et al. [C4] explicitly account for intermediate components that are shared by several features. We therefore treat shared signal processing as an established cost-modeling principle. Our question concerns when a frozen predictor can return a probability vector before all signal features have been computed, with a bound on its loss relative to that same predictor evaluated on all features.

Risk-controlled early exit is also established. Fast yet Safe [C1] includes relative performance gaps and multiclass Brier loss, selecting exit thresholds from calibration data under its stated sampling assumptions. Instance-adaptive lazy tree inference [C3] includes a deterministic residual-capacity condition that preserves the complete ensemble's predicted class. Emde et al. [C2] derive certificates for uncertainty calibration under adversarial perturbations, including worst-case top-label Brier scores on labeled samples. These guarantees address different objects. The proposed stopping condition bounds the increase in multiclass probability loss for each allowed input and every possible label, provided that the full predictor's probability vector belongs to the computed outer set. It does not certify absolute predictive accuracy or calibration, and it does not generally imply an identical predicted class.

The computational and decision-theoretic ingredients have important precedents. QuickScorer [C6] uses candidate-leaf elimination and feature-wise processing to accelerate exact tree-ensemble evaluation. Vovk et al. [C5a, C5b] use minimax proper-loss criteria to merge probability predictions, including a binary Brier construction in the extended working paper. We build on these principles rather than claiming their introduction. The specific methodological target is a tractable multiclass loss envelope under uncomputed feature blocks, coupled to a stopping and acquisition policy whose complete execution cost is measured. This emphasis also distinguishes the evaluation from a count of skipped features or trees alone. Recent Applied Soft Computing studies already measure mobile latency for wearable recognition [V1, V2], and V1 combines early exit with uncertainty estimation and calibration. The empirical question is whether the additional pointwise loss guarantee remains useful after feature extraction, enclosure updates, certification, routing and prediction are all timed within the stated input boundary.

英文使用边界：第三段的 tractable、useful 和 stopping/acquisition 描述仍需最终数学、数值守护与测时结果支撑；不得提前加 superior、state-of-the-art、first、calibration-free accuracy、hard real-time 或泛称 distribution-free superiority。

## 可直接录入英文稿的核验书目

### 目标刊近邻

[V1] X. Liu, L. Zhang, W. Huang, D. Cheng, H. Wu, A. Song, Fixing deep early exit ensembles for sensor-based human activity recognition through uncertainty quantification, Applied Soft Computing 184 (2025) 113861. https://doi.org/10.1016/j.asoc.2025.113861.

[V2] M. Yao, D. Cheng, L. Zhang, L. Liu, S. Song, H. Wu, A. Song, A sparse diverse-branch large kernel convolutional neural network for human activity recognition using wearables, Applied Soft Computing 167 (2024) 112444. https://doi.org/10.1016/j.asoc.2024.112444.

[V3] C. An, Q. Zhou, S. Yang, A reinforcement learning guided adaptive cost-sensitive feature acquisition method, Applied Soft Computing 117 (2022) 108437. https://doi.org/10.1016/j.asoc.2022.108437.

[V4] H. Su, J. Chen, Y. Lin, A four-stage branch local search algorithm for minimal test cost attribute reduction based on the set covering, Applied Soft Computing 153 (2024) 111303. https://doi.org/10.1016/j.asoc.2024.111303.

### 最近保证与计算基础

[C1] M. Jazbec, A. Timans, T. Hadži Veljković, K. Sakmann, D. Zhang, C.A. Naesseth, E. Nalisnick, Fast yet Safe: Early-Exiting with Risk Control, Advances in Neural Information Processing Systems 37 (2024) 129825–129854. https://doi.org/10.52202/079017-4124. Official article: https://papers.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html.

[C2] C. Emde, F. Pinto, T. Lukasiewicz, P. Torr, A. Bibi, Towards Certification of Uncertainty Calibration under Adversarial Attacks, International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/c50f8180ef34060ec59b75d6e1220f7a-Abstract-Conference.html. The official proceedings BibTeX also supplies pages 79428–79468; no DOI is added without a verified formal DOI.

[C3] S. Jang, S. Woo, A. Lee, Instance-Adaptive Lazy Inference for Tree Ensembles via Stability Surrogates and Residual-Capacity Certificates, Mathematics 14 (13) (2026) 2343. https://doi.org/10.3390/math14132343.

[C4] L. Zhao, A. Alipour-Fanid, M. Slawski, K. Zeng, Prediction-time Efficient Classification Using Feature Computational Dependencies, Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 2787–2796. https://doi.org/10.1145/3219819.3220117.

[C5a] V. Vovk, I. Petej, V. Fedorova, Large-scale probabilistic predictors with and without guarantees of validity, Advances in Neural Information Processing Systems 28 (2015) 892–900. https://papers.neurips.cc/paper_files/paper/2015/hash/a9a1d5317a33ae8cef33961c34144f84-Abstract.html.

[C5b] V. Vovk, I. Petej, V. Fedorova, Large-scale probabilistic prediction with and without validity guarantees, On-line Compression Modelling Project (New Series), Working Paper 13, 2015, revised 3 November 2019. https://alrw.cs.rhul.ac.uk/articles/13.pdf. Cite this working paper specifically when relying on its expanded binary Brier derivation; it is not a second formal NIPS article.

[C6] C. Lucchese, F.M. Nardini, S. Orlando, R. Perego, N. Tonellotto, R. Venturini, QuickScorer: A Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees, Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2015, pp. 73–82. https://doi.org/10.1145/2766462.2767733. Full title and authors verified from the formal paper; some metadata shortens the title to QuickScorer.

[C7] T. Bolukbasi, K.-W. Chang, J. Wang, V. Saligrama, Resource Constrained Structured Prediction, Proceedings of the AAAI Conference on Artificial Intelligence 31 (1) (2017) 1756–1762. https://doi.org/10.1609/aaai.v31i1.10917. Optional background citation.

### 强获取方法与损失依据

[A1] S. Gadgil, I. Covert, S.-I. Lee, Estimating Conditional Mutual Information for Dynamic Feature Selection, International Conference on Learning Representations, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html.

[A2] M. Valancius, M. Lennon, J. Oliva, Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition, Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research 235 (2024) 48957–48975. https://proceedings.mlr.press/v235/valancius24a.html.

[A3] A.L.I. Norcliffe, C. Lee, F. Imrie, M. van der Schaar, P. Liò, Stochastic Encodings for Active Feature Acquisition, Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research 267 (2025) 46784–46814. https://proceedings.mlr.press/v267/norcliffe25a.html.

[A4] H.-T. Huang, D. Dinh, J.B. Oliva, Information Templates: A New Paradigm for Intelligent Active Feature Acquisition, arXiv preprint arXiv:2508.18380v2, 2026. https://doi.org/10.48550/arXiv.2508.18380. Original version submitted in 2025; the version used in the prior capability audit is v2, revised 4 February 2026.

[F1] G.W. Brier, Verification of forecasts expressed in terms of probability, Monthly Weather Review 78 (1) (1950) 1–3. https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2.

[F2] T. Gneiting, A.E. Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation, Journal of the American Statistical Association 102 (477) (2007) 359–378. https://doi.org/10.1198/016214506000001437.

[F3] C. Guo, G. Pleiss, Y. Sun, K.Q. Weinberger, On Calibration of Modern Neural Networks, Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research 70 (2017) 1321–1330. https://proceedings.mlr.press/v70/guo17a.html.

### 数据

[D1] N. Helwig, E. Pignanelli, A. Schütze, Condition monitoring of hydraulic systems [Dataset], UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5CW21. The official page's citation year is 2015; the repository donation date is not a replacement publication year.

[D2] J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, X. Parra, Human Activity Recognition Using Smartphones [Dataset], UCI Machine Learning Repository, 2013. https://doi.org/10.24432/C54S4K.

## 阶段判断与正文使用建议

**期刊/文献准备阶段：CONDITIONAL。** 算法与信号处理方向适合 ASC；独立能力的准确表述已有可用近邻依据。尚需用实际结果确认可靠包络的紧度、数值保守性、认证总开销、完整预测器质量，以及相对简单固定顺序和适用强方法的实际增量。这个阶段不构成投稿 GO，也不因官方页面访问限制而停止已授权研究。[目标刊最近 HAR 早退稿](https://doi.org/10.1016/j.asoc.2025.113861)、[直接安全树推理近邻](https://doi.org/10.3390/math14132343)、[Fast yet Safe](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)。

新正文优先解决三个可能动摇主线的问题：完整模型本身是否在合法主体/任务切分下有竞争力；包络证书是否在足够多案例尚未计算全部块时达到容忍；累计证书与路由时间是否小于真实节省的特征处理时间。已失败的旧路线应保留为方法选择和代价诊断证据，不能删除失败来源来提高表面成功率。本文的文献证据支持这个研究问题与边界，不替代这些尚待实测的结论。
