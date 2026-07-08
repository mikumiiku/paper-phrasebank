# Changelog

All notable changes to `paper-phrasebank` will be documented in this file.

## [1.0.0] - 2026-07-08

### Added
- **建库链路**：PDF/DOCX 解析 → 脏数据正则清洗 → 多级降级文本分块（章节/段落/标点回退，绝不硬截断）→ LLM 抽取（元数据解耦 + 按块抽取候选句）→ 待审核队列（按论文分文件、JSON 落盘）→ 交互式人工审核（保留/丢弃/编辑/批量、Ctrl+C 断点续审）→ BGE-M3 本地编码 → Chroma 入库
- **检索链路**：自然语言 → BGE-M3 编码 → Chroma 向量检索 → rich 格式化展示（原句/分类/标签/来源/score）
- **LLM 抽取**：OpenAI Compatible 客户端（DeepSeek / 通义千问 / Kimi / Ollama 等）；JSON 夹逼容错（首 `[` 末 `]`）；dict 单 unwrap 兜底；按 chunk 分调 + 失败分块单独记录可重试
- **OCR 插件**：MinerU API / PaddleOCR API 可插拔（`OcrBackend` 协议 + `register` 装饰器工厂）；未配置时图片页仅警告跳过
- **配置管理**：`ppb config` 全程交互式（方向键菜单 + 密码掩码 + 脱敏展示）；首次运行自动引导三步流程；`.env` 只是持久化载体，用户无需感知
- **CLI 4 子命令**：`ppb extract <file>` / `ppb review` / `ppb search "<query>"` / `ppb config [show|set]`
- **Chroma 一致性校验**：collection metadata 写入 `model_name` + `model_identifier`（含 sentence-transformers 库版本），启动时 first-fault 校验防向量空间漂移
- **去重**：`source_file_hash` 防重复处理；`--force` 强制重抽
- **测试**：102 测试全绿（模块单测 + E2E 闭环，含真实 BGE-M3 + on-disk Chroma）
- **在线 E2E 验证**：真实 PDF（Xu et al. 2025 A-MEM）+ DeepSeek API → 191 条候选句入库 → 自然语言搜索精确命中原句

### Known Limitations (v1 scope, 后续规划)
- 检索结果不做 LLM Rerank 二次排序
- 无 `ppb list` / `ppb stats` 等辅助查看命令
- 无跨论文相似句子自动去重提示
- 功能分类体系暂不支持自定义扩展
