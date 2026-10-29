# 提交清单：Zhang Yaxing / 3036707458

本地材料已备齐（代码、checkpoint、报告、官方评分 JSON、实验记录、哈希）。
GitHub 推送、公开链接与课程网站提交尚未完成，必须由本人操作产生。
填写正式 test 分数：**82.65524625267666**（386/467）；不要使用 validation 分数。

## 上传前
1. 阅读 REPORT.pdf 和 README.md，确认能够解释实现、证据、局限性和 AI 披露。
2. 在干净目录克隆仓库，按 README 复现：`python check_data.py` → 训练或直接
   `--checkpoint code/runs/model_bundle` 推理 → 官方评分，核对 val 92.3559 / test 82.6552。
3. 保持 `code/runs/model_bundle/model.npz` 字节不变（sha256 见 ASSET_INVENTORY.json）。

## GitHub
1. 登录账号 911zyx，新建**公开**仓库 `dase7506-mp2`（不勾选初始化 README/.gitignore）。
2. 推送本地仓库（含 tag）：
   ```
   git remote add origin https://github.com/911zyx/dase7506-mp2.git
   git push -u origin main --tags
   ```
   仓库含约 106 MB 课程数据（单文件最大 72 MB < 100 MB，无需 LFS）。
3. 推送后复制**完整 commit SHA**。提交用代码地址应为：
   `https://github.com/911zyx/dase7506-mp2/tree/<完整commitSHA>`
4. checkpoint 链接指向同一 commit 下的资产目录：
   `https://github.com/911zyx/dase7506-mp2/tree/<完整commitSHA>/code/runs/model_bundle`
   （26 KB，评审可直接下载复现，无需 Release；如想用 Release 附 zip 亦可，附 SHA256SUMS。）
5. 用退出登录的窗口测试两个链接可访问、SHA256 核对一致。
   提交后不要覆盖资产或移动标签。

## 课程提交
- 网站：https://xudongwu-0.github.io/courses/dase7506/mp2/
- 填写学号 3036707458 与 test accuracy 82.65524625267666，以及上面两个链接。
- **点 “Submit on GitHub” 创建 issue**——只准备链接不点提交等于没交。
- 2026-10-31 23:59 (UTC+8) 前最后一次提交必须包含代码与 checkpoint 两个链接。
- 始终使用同一 GitHub 账号更新；保存 issue 地址和提交回执。

## Peer review
- 用别人提交的代码版本+匹配 checkpoint，按其安装说明跑官方评分器。
- 交 Peer Review Report（含 submission number 与复现分数），或邮件
  wu.xudong@connect.hku.hk。

账号、发布链接和提交回执必须由实际操作产生；目前未代填或虚构。
