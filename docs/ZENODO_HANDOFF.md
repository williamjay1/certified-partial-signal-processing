# Zenodo 交接：由作者亲自生成 DOI

仓库：[williamjay1/certified-partial-signal-processing](https://github.com/williamjay1/certified-partial-signal-processing)。唯一作者与通讯作者为 **Junjie Zhang**；GitHub 账户名不是新增作者。

本仓库已经准备 `CITATION.cff`、`.zenodo.json`、文件级 `release_manifest.json`、许可说明、代码、模型、逐案例研究结果和验证记录。准备的软件版本为 **1.0.0**。本次不创建 GitHub release，不操作 Zenodo，也不预填或杜撰 DOI。

1. 用自己的账户登录 Zenodo，连接 GitHub，在 Zenodo 的 GitHub 仓库列表里启用本仓库。应先启用，再建立新的 release。[官方启用说明](https://help.zenodo.org/docs/github/enable-repository/)
2. 核对 GitHub `main` 的提交和文件完整性，在 GitHub 建立新的 `v1.0.0` release。需要归档的代码与结果已放进对应 Git 树，不依赖额外 release 附件或 LFS 指针。[官方归档说明](https://help.zenodo.org/docs/github/archive-software/github-upload/)
3. 打开 Zenodo 生成的记录，核对标题、版本、作者 `Zhang, Junjie`、ORCID `0009-0004-8821-4018`、两个机构，以及 Files 中实际归档的 ZIP。下载解压到 D 盘，以该副本运行 `python -m asc_certified validate`；可继续运行 `replay` 验证主结果。
4. 核对许可：主软件记录为 MIT；原创研究输出/图表为 CC BY 4.0，上游来源字段和第三方资源按 `DATA_LICENSES.md` 保留各自条款。不要用 Zenodo 的一个默认许可覆盖所有来源材料。[官方许可说明](https://help.zenodo.org/docs/deposit/describe-records/licenses/)
5. 使用实际已归档记录的 **version DOI** 引用论文所用确切版本。再将真实 DOI 写入论文的数据与代码可用性声明和软件参考文献，并按需要更新 `CITATION.cff` 与仓库说明。来源数据的两个 UCI DOI 仍然保留。[官方版本说明](https://help.zenodo.org/docs/deposit/manage-versions/)

`.zenodo.json` 与 `CITATION.cff` 同时存在时，Zenodo 以 `.zenodo.json` 为准，GitHub 的引用界面继续使用 CFF。元数据中已固定唯一作者，避免以 GitHub 贡献者统计替代科研作者信息。[官方 JSON 元数据说明](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)

首次 DOI 生成后，记录真实 DOI、版本及对应提交。若改动归档科学文件，建立有记录的新版本；不要把文件改动伪装成原来的冻结版本。更新 `CITATION.cff`、README 等被 manifest 覆盖的文件时，也应重新生成 manifest 并验证后提交。以 Zenodo 实际 Files 与验证结果为准，而不是只检查网页有 DOI 字样。
