# 分类器修复可行性诊断：存在信号，但门控收益尚未验证

这是用户重新授权后的独立重设计诊断，不是把旧试点由STOP改成通过。旧试点结果、selection lock和MLP checkpoint的SHA-256在本次计算前后完全一致。没有读取或评价gate-fitting/calibration/audit分区，没有使用旧435个outer-test案例，没有训练HAR，没有改风险容忍或重跑门控策略。

输入仅为已有D盘特征缓存中的907个predictor-fitting案例（76个工况代理段）与223个内部validation案例（25段），段交集为0。拟合类别73/80/90/100计数为150/150/150/457；validation为40/40/40/103。沿用原907案例拟合的238维scaler。拟合前D盘可用150,436,294,656字节；全部新脚本、模型与结果写D。Torch/BLAS/OpenMP限制为单线程。

## 原MLP拟合与early stopping检查

新raw-pilot wrapper在构造MLP前设置seed17；原fit方法每个batch独立抽取训练案例、随机whole-block masks并给予约10%全观测，没有按epoch确保每行恰好出现一次。validation masks在训练前固定为.15/.35/.65/1密度，checkpoint是最低验证CE的state_dict深拷贝，恢复后再选temperature。

用223个validation案例、原seed17与原固定masks重建checkpoint CE，结果**1.2805999517440796**，与保存的best CE差为0。没有发现明显的best-checkpoint恢复或保存错误。31是终止epoch；stale达到15仍至少训练到31轮才停。best epoch和完整训练轨迹没有保存，不能把31轮称为最佳轮，也不能从这份checkpoint确定哪一个优化步骤导致较弱表现。

## 只拟合两个固定模型

两模型均为HistGradientBoostingClassifier，seed17，learning_rate=.1，31 leaves，min_samples_leaf20，L2=1，255 bins，early_stopping=False，无网格。

1. 全信息HGB：238输入、150轮，只在907个fit案例上训练；用来检查完整现有representation是否包含可泛化预测信号。
2. 联合masked HGB：266输入（238个按group mask遮盖的值+28个mask bit），100轮；每个fit案例固定10个副本：1个全信息、1个EPS1-statistics-only、8个每副本密度Uniform(0,1)的随机whole-block mask。9070行增强样本仍只有907个原案例，不扩样本量。精确构成与参数在执行前记录。

全信息HGB未训练缺失输入，所以没有把它对EPS1-only零填充的输出当作公平partial-input基线。冻结MLP和联合masked HGB均在相同full与initial EPS1模式评价。

| 模型与输入 | 内部validation half-Brier | accuracy |
| --- | ---: | ---: |
| 冻结masked MLP，全信息 | 0.339247 | 47.09% |
| 全信息HGB，全信息 | 0.106148 | 86.55% |
| 联合masked HGB，全信息 | 0.142932 | 80.72% |
| 冻结masked MLP，initial EPS1 stats | 0.356021 | 46.19% |
| 联合masked HGB，initial EPS1 stats | 0.329552 | 52.02% |

HGB全信息训练耗时5.761秒；联合masked模型15.811秒；本诊断总耗时22.262秒（不含模块导入）。均达到固定迭代上限，没有按validation挑轮数。风险差相对原MLP为：全信息HGB −0.233099，联合masked全信息 −0.196314，联合masked EPS1 −0.026469。这些是同一内部validation的描述性模型可行性比较，不是独立确认性胜利。

## 结果支持和不支持什么

现有完整raw statistics/frequency representation有明显预测信号。把旧试点失败解释成“数据本身没有信号”不成立；更可靠的共同预测器具有可行性。联合mask输入的树模型在相同full模式也明显优于被冻结的MLP，说明修复并不必然要求给每种mask训练一个完全独立分类器。

初始EPS1依然弱：联合masked模型在fit的EPS1风险0.007644、accuracy99.89%，而validation风险0.329552、accuracy52.02%。这个大差距显示该初始块上的拟合能力不能转成工况段外可靠预测，也限制“只见EPS1就估值”的信息基础。完整输入与初始输入的表现不能被混为一谈。

同类fit→validation的238维均值变化RMS为0.301–0.350个训练标准差；validation值超出同类训练范围的单元格比例分别为3.64%、7.93%、0.96%、3.20%。这些只是已知分区的描述性差异，不是分类器失效的因果解释。单seed、单来源和内部validation均不能证明树模型在新主体或新系统上泛化。

因此，可推进的修复方向是先确保共同partial-observation预测器有充分的内部验证质量，再重新设计并锁定计算选择试验；不能继续在这份弱MLP上把少规划的速度当作有效能力。新的HGB并未放入旧门控、不曾重算旧风险ceiling，也没有验证规划、gate、风险约束或完整时间前沿。两个HGB比较还同时改变了学习器与mask训练样本设计，不能作为隔离单一优化因素的消融。旧试点仍是STOP_NO_EXPANSION；下一项正式实验须有新的明确设计和独立评价边界。

执行来源：research/predictor_repair_diagnostic.py；完整参数与线程记录、实际结果、两份新模型均在results/redesign_diagnostic。所有数值可追溯，结果不覆盖旧证据。
