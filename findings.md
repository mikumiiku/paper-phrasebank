# Findings & Decisions (v1 MVP 实现完成)

## Requirements
- CLI 命令：`ppb extract <file>` / `ppb review` / `ppb search "<query>"` / `ppb config [show|set]`
- PDF(DOCX) → 元数据解耦抽取（仅首页一次）→ 按块抽取候选句（含六类功能分类）→ 待审核队列 → 人工审核（断点续审）→ BGE-M3 本地编码 → Chroma 入库
- 自然语言 → BGE-M3 编码 → Chroma 向量检索 → Top-K 纯向量召回（不做 Rerank）
- 去重用 `source_file_hash`；图片页 OCR（可插拔 MinerU/PaddleOCR）；未配仅警告跳过

## Technical Decisions
| 决策 | 理由 |
|------|------|
| B/C/F 三链路用并行 subagent 推进（ultracode 模式） | 三链路互不依赖、可独立开发与测试 |
| 按依赖顺序串行推进 A→D→E（主代理）→ G/H（主代理） | 配置/OCR/审核被建库链路依赖 |
| 以需求 §9 v1 范围为唯一权威，子代理测试若与需求冲突 → 修测试 | 四测试暴露 chunking lossless/clean 整体 drop 假设与需求矛盾 |

## Implementation Status (ultracode v1 MVP 完成)
- **测试: 102 passed / 0 failed, 21.78s**
  - E2E 闭环 5（extract→chunk→LLM→queue→review→vector 真实 BGE-M3→search 命中）
  - LLM 29、OCR 9、Review 6、Parsing+Chunking 24、Vector+Search 21、Config 8
- **关键集成修复（阶段 G/H）**：
  - `vector/__init__.py`: client singleton 感知 data_dir 变化（`reset_client`）→ E2E 隔离
  - `review/interactive.py::_flush_kept`: 用显式字段映射替代 `**e` 解包（entry 含 status/decision 等额外字段，to_metadata 是 kwarg-only）
  - `pipeline.py`: 真正的 thin orchestration；metadata 失败非 fatal-first-fault，sentence chunk 失败记录 failures

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| ocr/__init__ register 签名初版不是装饰器表单 | 修成 `def register(name) -> deco` |
| subagent C metadata.py 早期 `raise LLMErrorEmpty()` 未定义 | subagent 自修复 |
| embed 测试 mock assert `call_args[0]`（ST 实际 kwargs 传入） | subagent F 自修复 |
| vector.client 单例与 E2E 测试隔离冲突 | 主代理修：client 记录 `_ppb_path`，data_dir 变化时 rebuild；加 `reset_client()` |
| `_flush_kept` 传 `**entry` 给 kwarg-only `to_metadata` | 主代理修：显式字段映射 `_entry_to_vector_rec` |
| `_entry_to_vector_rec` 定义在调用之后 | 提到 `_flush_kept` 之前 |

## Online E2E Verification (real DeepSeek API + real PDF)
- 真实 PDF: Xu et al. 2025 "A-MEM Agentic Memory for LLM Agents" (1MB)
- DeepSeek: 抽取 53 块 → 191 条候选句入库
- 搜索 3 个自然语言查询（中英文混合）Top-1 均命中论文原句，分类正确

## Known Fix (post-E2E): tags char-level corruption
- Bug: DeepSeek serialises `tags` JSON 字段为单个逗号字符串，extract.py 位置把字符串当 iterable → char-level list
- 修复: `llm/extract.py::_build_sentence` 加 str 类型守卫 + 短串过滤
- 迁移: 用 review queue 的干净 tags 重建 vector store（`/tmp/migrate_tags.py`），191 条目完整迁移

## Resources
- 需求文档：`paper-phrasebank-requirements-final.md`
- 实现计划：`/home/mikumiiku/.claude/plans/clever-drifting-sutton.md`
- 技术栈（已落盘）: Typer / questionary / rich / PyMuPDF / python-docx / OpenAI SDK / sentence-transformers (BAAI/bge-m3) / chromadb / platformdirs / tomli+tomli_w
