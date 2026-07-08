# Progress Log

## Session: 2026-07-08

### Phase 0: Planning & Decision
- **Status:** complete
- Actions taken:
  - 分析 `paper-phrasebank-requirements-final.md`（终版需求）
  - 向用户确认 4 个关键决策（包名/platformdirs+TOML/uv/pyproject/pytest+E2E）
  - 编写实现计划文件 `/home/mikumiiku/.claude/plans/clever-drifting-sutton.md`
  - 用户批准计划
- Files created/modified:
  - `task_plan.md` (created)
  - `progress.md` (created)

### Phase A: 项目骨架 + 配置
- **Status:** complete
- Actions taken:
  - 写 `pyproject.toml`（Typer/questionary/rich/PyMuPDF/python-docx/openai/sentence-transformers/chromadb/platformdirs/tomli+tomli_w/pytest/respx）
  - 写 `config.py`（platformdirs + TOML、脱敏、单例缓存、provider 默认值）
  - 写 `ui/config_menu.py`（首次引导三步 + 交互式单项修改循环）
  - 写 `cli.py`（Typer app + 4 子命令骨架）
  - 写 `tests/conftest.py`（tmp_dir + isolated_config fixture）
  - `uv sync --extra dev` 通过；`ppb --help` 输出 4 子命令
- Files created/modified:
  - `pyproject.toml`, `.gitignore`, `README.md`
  - `src/phrasebank/__init__.py`, `config.py`, `cli.py`
  - `src/phrasebank/ui/__init__.py`, `ui/config_menu.py`
  - `src/phrasebank/pipeline.py`, `search.py`, `review/interactive.py`（占位 stub）
  - `tests/conftest.py`, `tests/__init__.py`

### Phase D: OCR 插件
- **Status:** complete
- Actions taken:
  - 写 `ocr/__init__.py`（OcrBackend 协议 + REGISTRY + register 装饰器 + get_backend 工厂）
  - 写 `ocr/mineru.py`（MinerU v4 API）
  - 写 `ocr/paddle.py`（通用 PaddleOCR HTTP）
  - 写 `tests/ocr/test_mineru.py`、`tests/ocr/test_paddle.py`（respx mock HTTP）
  - 修复 register 装饰器签名（支持 `@register("name")` 形式）
  - 9 测试全绿
- Files created/modified:
  - `src/phrasebank/ocr/__init__.py`, `ocr/mineru.py`, `ocr/paddle.py`
  - `tests/ocr/test_mineru.py`, `tests/ocr/test_paddle.py`

### Phase B/C/F: 并行推进（subagent）
- **Status:** in_progress（ultracode：主代理不接管，等子代理跑完）
- Actions taken:
  - 启动 3 个并行 subagent：阶段 B（解析+分块）、阶段 C（LLM 抽取）、阶段 F（向量建库检索）
  - 阶段 C 已交付：`llm/{client,metadata,prompts,extract}.py` + `tests/llm/{test_client,test_metadata,test_extract}.py`（**29 passed**）
  - 阶段 C 关键接口决策：
    - `LLMClient.call_object(system, user) -> dict`（元数据抽取，unwrap 单 list-of-one）
    - `LLMClient.call_json(system, user) -> list[dict]`（句子抽取，dict 单 unwrap 一次）
    - `CandidateSentence.tags` 保持 `list[str]`，逗号字符串化推迟到 review→vector 入库
    - `write_failures(file_hash, failed_indices)` 写到 `review_queue/{hash}_failures.json`
  - 阶段 F 已交付：`vector/{__init__,embed,schema,store}.py` + `search.py` + `tests/vector/{test_embed,test_store}.py` + `tests/test_search.py`（**21 passed**，含真实 BGE-M3 + on-disk Chroma 端到端验证）
  - 阶段 F 关键决策：
    - `ModelMismatchError` 在任一 `model_name` 或 `model_identifier` 不一致时 first-fault
    - `model_identifier = sentence-transformers@{pkg_version}:{model_name}`，用 `importlib.metadata.version` 取得（不依赖不存在的 `_ST.__version__`）
    - `encode()` 位置传参 ST，避免 ST 版本 kwarg 名 `texts`→`inputs` 漂移
  - 阶段 B 已交付：`parsing/{__init__,pdf,docx,clean}.py` + `chunking.py` + `tests/parsing/{conftest,test_pdf,test_docx,test_clean}.py` + `tests/test_chunking.py`（**24 passed**）
