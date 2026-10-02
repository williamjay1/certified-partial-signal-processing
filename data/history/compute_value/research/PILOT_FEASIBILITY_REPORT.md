# 500案例试点：STOP，不扩训练

2026年10月2日，在近邻能力核查和独立代码审查后执行唯一锁定试点，进程退出码0，实测总时间69.787秒。没有重新选阈值、换模型、换种子、删除失败case、下载HAR或扩展强基线训练。成功执行与扩展通过是两个不同结论。

已有1770个Hydraulic开发周期按146个完整条件段预分；旧435个test信号没有解析。predictor fit为907周期/76段，内部val223/25段；gate fit分配380/19后固定抽300，cal和audit各130/13后固定抽100。未抽中的140个gate分区周期没有回流训练。段是连续工况代理单元，不是可保证独立的物理重复实验。

新分类器使用同一seed17，70轮上限，31轮早停，内部validation CE=1.2805999517，temperature=1.25。scaler和donors仅来自predictor fit，student仅用该分区200个deep teacher动作拟合，其300个gatefit终端风险标签保持独立。全部方法共用该冻结分类器。

14个物理传感器每个分为statistics和rFFT两块，共28计算块、238特征。共同初始块是按训练平均即时风险改善/实测块时间选出的EPS1 statistics。固定特征工作额度是实测块成本和的30%，即0.800280毫秒估计额度。它约束估计块工作，不保证真实截止时间；软件计时从完整60秒raw window已经到达内存开始，涵盖实际选中块计算、缩放、路由、规划、续接和最终预测。采样、传输和磁盘读入不在该软件计时内。

cal static风险0.3369500622，按预先规则锁定ceiling=0.3389500622。十个compute tradeoff全部不满足校准风险容忍。最低风险lambda=0被保留为诊断，因此成本项实际不起作用，部署决策与quality-only完全一致；没有把不可行设置说成校准成功。

audit100案例/13段的once风险0.3624303114，完整均时0.786146ms，P95=3.175750ms，P99=4.521187ms；static为0.3565564838/0.850881ms，quality-only为相同风险0.3624303114/0.775114ms。全部冻结controls都超过原cal ceiling，所以最快合格control不存在，主终点速度增益未定义。没有用audit static重设ceiling。once规划率17%，廉价分支率83%，平均计算1.11块。

单独instrumented profile中gate均时0.056107ms；同case deep分支决策工作减once已选择分支决策工作的signed均值为2.199425ms，全case纳入且不截负值。这个描述性净工作账目涵盖浅规划、深规划、student和廉价路由，不能解释为卸除模块的因果效果。仅此开销门槛通过；其余扩展条件未通过。非法动作0，估计额度违规0，但独立profile真实feature计算超过估计额度的比例为2%。平均feature工作0.150513ms，不能冒充所有case的实时保证。

全局固定混合对cal五分支R/T枚举单分支和双分支顶点，冻结为skip 9.5434%、static 90.4566%。固定case-keyed随机路由直接量测；其rng开销和实际risk偏离保留，不把counterfactual线性期望当作部署时延。完整十controls风险、均时、尾延迟、违规和paired segment诊断区间均在pilot_summary.json及CSV中。

## STOP后已有case的描述性诊断

只读取同一checkpoint和D缓存，对已打开的cal100/audit100计算全信息与zero-mask锚点，不增加拟合或选择。全信息cal half-Brier=0.3522730172、accuracy=.39；audit=.3693979681、accuracy=.42。zero-mask cal=.3628794849、accuracy=.56；audit=.3644064367、accuracy=.53，zero-mask全部预测最多数类。全信息不是该partial-budget下的可部署基线；它在audit没有优于zero-mask，提示冻结分类器的可靠性是本试点的重要边界。不能从这个单seed诊断把失败全部归因于分类器，也不能推出所有门控算法无效。

cal类别73/80/90/100计数为18/12/14/56，audit为15/15/17/53。static风险audit−cal为0.0196064216，精确描述性分解为：类别构成项0.0049749359，加类别内部风险项0.0146314857。公式为Σ(pA−pC)RC + ΣpA(RA−RC)，重构误差约7×10^-18。这只是算术分解，不是工况变化或类别构成的因果归因；没有借此改变风险容忍或删除auditcase。

状态为**STOP_NO_EXPANSION**，仅针对本次锁定实现与投资规则。它未完成未见主体验证、跨任务泛化或强采集方法比较；cost-aware选择没有独立增量。完整诊断及逐类风险在pilot_posthoc_diagnostics.json，protocol和原七来源负面证据保留。

## 28块特征的准确计算定义

每传感器一块11统计量：mean；population std（ddof0）；min；max；q25/q50/q75（默认linear样本分位数）；first；last；slope=dot(x,t)/dot(t,t)，t为[-.5,.5]上的n点等距向量，单位是每个标准化60秒周期的变化；rms=sqrt(mean(x²))。

另一块先对x减均值并rFFT，P=|rFFT(x−mean(x))|²，q=P/max(sum(P),10^-30)，频率f=rfftfreq(n,d=60/n)（Hz）。六量为log1p(sum(P)/n²)；Σqf谱质心；sqrt(Σq(f−centroid)²)谱宽；−Σq log(max(q,10^-30))/log(max(bin数,2))标准化谱熵；first argmax(P)对应频率（含DC）；Σq[f≤最高FFT频率/3]。若实际计算总谱功率为0，q和各量为0；数学恒定信号的浮点去均值可能留下微小残余功率，原始实现没有额外阈值删去它。

每个channel的两块是独立可选计算，不声称对应两次传感器购买。确定性特征后仅用predictor-fit均值/标准差缩放；std<10^-8时取1。online只计算选中块，并使用冻结scale。详细源码定义、hash和软件环境均在计算结果记录中，正文不需放本地路径或后台hash。
