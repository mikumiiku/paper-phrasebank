# paper-phrasebank (`ppb`)

个人学术写作辅助 CLI。把论文喂进去 → LLM 抽取出可复用的模板句 → 人工审核 → 写入本地向量库；写论文时用自然语言检索可直接参考或改写的句子。

```bash
# 建库：读论文存好句子
ppb extract paper.pdf        # 解析 + LLM 抽取 → 待审核队列
ppb review                    # 逐条审核（Ctrl+C 断点恢复），通过的写入向量库

# 检索：写论文时直接搜
ppb search "研究空白的描述"   # 自然语言检索 Top-K 匹配句
```

## 特性

- **PDF/DOCX 解析** + OCR（MinerU / PaddleOCR 可插拔）兜底图片页
- **多级降级文本分块**：章节 → 段落 → 标点回退，**绝不硬截断**
- **LLM 抽取**：论文元数据解耦提取、按 chunk 抽取候选句（六类功能分类：研究背景/研究空白/方法/结果/局限性/贡献）、JSON 容错 + 失败分块重试
- **断点续审**：逐条审核实时落盘，随时 `Ctrl+C` 退出
- **本地向量库**：BGE-M3 编码（中英双语，无额外费用） + Chroma 单进程持久化
- **模型一致性校验**：collection 写入模型标识，启动时 first-fault 校验防漂移
- **纯交互式配置**：`ppb config` 全程引导，无需手动编辑任何文件

## 安装

```bash
# 用户级 CLI 安装（推荐）
uv tool install .

# 或开发态
uv sync
```

可执行命令注册为 `ppb`。

## 首次运行

首次执行任意 `ppb` 子命令，会自动进入引导式配置流程：

1. **选择 LLM 提供商**：DeepSeek（官方 API，自动绑定 base_url + model）或 OpenAI Compatible（通义千问 / Kimi / Ollama 等）
2. **配置 OCR 后端**（可选）：MinerU / PaddleOCR，未配置则图片页跳过
3. **确认并写入**：`~/.config/ppb/config.toml`（密钥脱敏展示）

## 用法

```bash
ppb config                         # 交互式查看/修改配置
ppb config --show                  # 只读展示当前完整配置（密钥脱敏）
ppb config --set <key> <value>    # 快速设置单项

ppb extract <file.pdf|docx>       # 抽取候选句，写入待审核队列
ppb extract <file> --force        # 已处理过也强制重新抽取

ppb review                         # 逐条审核（支持 Ctrl+C 断点恢复），通过的写入向量库

ppb search "<query>"               # 自然语言检索，返回 Top-K 匹配句（默认 10）
ppb search "<query>" -k 20        # 指定返回条数
```

## 测试

```bash
pytest tests/                      # 102 测试（含 E2E 闭环）
pytest tests/ --ignore=tests/e2e  # 仅单测（跳过真实 vector 路径）
```

## 技术栈

| 模块 | 库/服务 |
|------|---------|
| CLI 框架 | Typer |
| 终端交互 | questionary |
| 终端展示 | rich |
| PDF 解析 | PyMuPDF |
| Word 解析 | python-docx |
| OCR | MinerU API / PaddleOCR API |
| LLM | OpenAI Compatible 客户端 |
| Embedding | sentence-transformers + BAAI/bge-m3 |
| 向量库 | chromadb（PersistentClient） |
| 配置持久化 | platformdirs + TOML |

## 项目结构

```
src/phrasebank/
├── cli.py                 # 入口，4 子命令注册
├── config.py              # 配置读写（platformdirs + TOML）
├── pipeline.py            # 建库链路编排
├── search.py              # 检索链路编排
├── parsing/               # PDF/DOCX 解析 + 脏数据清洗
├── chunking.py            # 多级降级文本分块
├── llm/                   # LLM 抽取（client + metadata + extract）
├── ocr/                   # OCR 插件协议 + MinerU/PaddleOCR 实现
├── review/                # 待审核队列 + 交互式审核 + 富文本渲染
├── vector/                # 向量建库（embed + store + schema）
└── ui/                    # 配置引导菜单 + 候选句渲染
```

## 隐私说明

- 论文正文会发送给 LLM API 做抽取；未公开稿件需评估内容外传风险
- Embedding 本地完成，不经网络
- OCR 若走云端 API 同样涉及内容外传

## v1 范围外（后续规划）

- 检索结果的 LLM Rerank 二次排序
- `ppb list` / `ppb stats` 等辅助查看命令
- 跨论文相似句子自动去重提示
- 功能分类体系自定义扩展

详见 `paper-phrasebank-requirements-final.md`。

## License

MIT
