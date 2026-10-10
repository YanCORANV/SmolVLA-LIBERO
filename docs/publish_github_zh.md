> 历史说明（2026-10-09）：本文是首次发布 v0.1 时的流程记录。当前仓库已有 `origin`，最终 v4 实验也已完成；不要直接照抄文中的 `git add .`、`git remote add` 或旧项目描述。当前公开叙事见 [README](../README.md) 和 [最终结果](results.md)。

# 第一次发布到 GitHub

目标仓库：`YanCORANV/SmolVLA-LIBERO`。第一版定位为已有可运行代码和小规模
实验结果的 research prototype，后续继续迭代。发布不需要重新训练。

## 1. 在 GitHub 创建空仓库

打开 [创建仓库](https://github.com/new)，设置：

- Owner：`YanCORANV`
- Repository name：`SmolVLA-LIBERO`
- Description：`Controlled language-grounding evaluation and adaptation of SmolVLA on LIBERO.`
- Visibility：Public
- 不勾选 README，不添加 .gitignore，不选择许可证模板；本地已经有这些文件。

点击 Create repository。这种从本地导入的流程见
[GitHub 官方指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 2. 在项目目录打开 Windows PowerShell

本机 Windows Git 已配置 Git Credential Manager，推送时可以按提示通过浏览器
登录。发布用 Windows PowerShell；训练仍在原来的 WSL 环境中进行。
以下命令均从仓库根目录执行，本地 Git 已初始化为 `main`。

先检查：

```powershell
git status --short
git remote -v
```

如果出现 `detected dubious ownership`，仅为你自己的这个项目添加信任后重试：

```powershell
git config --global --add safe.directory (Get-Location).Path.Replace('\', '/')
```

## 3. 设置提交署名并检查待上传文件

如未设置过 Git 作者信息，用自己的署名和邮箱替换下面占位文字。
希望隐藏私人邮箱时，从 [GitHub 邮箱设置](https://github.com/settings/emails)
复制 GitHub 提供的完整 noreply 邮箱。

```powershell
git config user.name "YanCORANV"
git config user.email "zhanxi.24@intl.zju.edu.cn"
git add .
git diff --cached --stat
git status --short
```

上传内容应是代码、配置、文档、许可证和精选结果。
不应出现 `.venv/`、`outputs/`、`data/`、`checkpoints/`、密钥或大模型文件。
如果发现这些内容，先停止提交并检查 `.gitignore`。

## 4. 提交并推送

当前本地没有 remote；在第 2 步仍为空的情况下执行：

```powershell
git commit -m "Publish initial SmolVLA LIBERO research prototype"
git remote add origin https://github.com/YanCORANV/SmolVLA-LIBERO.git
git push -u origin main
```

如果提示登录，选择浏览器登录自己的 GitHub 账号。不要把密码或 token 发到聊天里。
如果出现 `origin already exists` 或推送冲突，保留报错并核对 remote；不要强制推送。

## 5. 检查发布后的页面

打开 [项目地址](https://github.com/YanCORANV/SmolVLA-LIBERO)，检查 README、图片、
结果表和 MIT 许可证均正常显示。这个地址在成功创建并推送后才能作为简历链接。
可在 About 中添加 `robotics`、`vision-language-action`、`smolvla`、`libero` 等 topics。

第一版简历描述可以写：

> Built a controlled language-grounding evaluation pipeline for SmolVLA on LIBERO
> and compared original-language and language-augmented fine-tuning on a single GPU.

暂不写“全面提升语言鲁棒性”：目前只有一个任务、三个开发状态，改写 B 仍失败。

后续每完成一个可验证改动就新增提交，逐步补充评估、图表和视频。
首次页面检查通过后，可按 `CHANGELOG.md` 创建 `v0.1.0` release，并将精选 MP4
作为 release 附件上传；模型权重和原始视频仍不进入 Git 历史。
