# Task Plan: paper-phrasebank

## Goal
构建 `paper-phrasebank`（CLI 命令行 `ppb`）Python 工具：存论文模板句到本地向量库 + 自然语言检索复用。v1 MVP 完整实现并带测试。

## Current Phase
阶段 A：项目骨架 + 配置

## Phases
- 阶段 A：项目骨架 + 配置（pyproject / config / config_menu）
- 阶段 B：解析 + 分块（pdf/docx/clean + chunking）
- 阶段 C：LLM 抽取（client/prompts/metadata/extract）
- 阶段 D：OCR 插件（protocol + mineru + paddle）
- 阶段 E：审核队列 + 交互（queue + review_render）
- 阶段 F：向量建库 + 检索（embed/store/schema + search）
- 阶段 G：CLI 整合（cli.py + extract --force + 兜底行为）
- 阶段 H：全链路 E2E + 收尾（test_pipeline + README）

## Decisions Made
| 决策 | 理由 |
|------|------|
| import 包名 `phrasebank` | 用户确认 |
| 配置 `platformdirs` + TOML（`~/.config/ppb/config.toml`） | 用户确认；标准位置、可读写 |
| uv + pyproject.toml | 用户确认；CLAUDE.md 强制 uv |
| pytest + 模块单测 + E2E 闭环 | 用户确认；CLAUDE.md 要求测试完整严格 |
| src/ 布局 | 区分包与项目元信息，单包多子模块 |
| 建库/检索仅共享 vector/ + config/ | 低耦合，可独立开发与测试 |
| Embedding、LLM client 单实例复用 | 防重复加载与版本漂移 |
| OCR 协议抽象 + 工厂 | 可插拔，避免 if-else 散落 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (待填) | | |

## Notes
- 单文件 >500 行按职责拆分（CLAUDE.md）
- 测试 fixtures 按领域拆分，禁止集中堆放（CLAUDE.md）
- 错误处理走 first-fault，不吞错
- 不在 v1：Rerank、ppb list/stats、跨论文去重提示、功能分类自定义
