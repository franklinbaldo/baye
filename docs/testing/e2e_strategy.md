# End-to-End (E2E) Testing Strategy

This document outlines the strategy for End-to-End testing of the Egregora pipeline. Unlike unit tests which verify isolated functions, these tests validate the integration of subsystems—from raw file ingestion to final site generation—while mocking expensive external dependencies (LLMs).

## Philosophy

1.  **Real I/O, Mocked Intelligence**: We use real files (ZIPs, Parquet, DuckDB) and real file system operations. We only mock the stochastic/expensive parts (Google Gemini API and PydanticAI Agent decisions).
2.  **Schema as Contract**: Every test stage must verify adherence to the `IR_MESSAGE_SCHEMA` (Interchange Representation).
3.  **State Verification**: We verify side effects (database rows, checkpoints, file creation) rather than just return values.

---

## 1. Input Adapter E2E
**Goal**: Verify that platform-specific exports (e.g., WhatsApp ZIPs) are correctly parsed into the standardized Interchange Representation (IR).

### Scope
*   **Input**: A synthesized ZIP file containing chat logs (`_chat.txt`) and dummy media files.
*   **System**: `WhatsAppAdapter` class and `parse_source` logic.
*   **Output**: Ibis Table conforming to IR v1.

### Test Plan
1.  **Fixture Setup**: Create a temporary `whatsapp.zip` containing:
    *   A chat log with edge cases (multiline messages, system messages, attachments).
    *   A dummy image file (e.g., `IMG-2025.jpg`) to test extraction.
2.  **Execution**:
    *   Initialize `WhatsAppAdapter`.
    *   Call `adapter.parse(zip_path)`.
    *   Call `adapter.deliver_media(...)`.
3.  **Assertions**:
    *   **Schema**: Verify columns `event_id`, `author_uuid`, `ts`, and `text` exist.
    *   **Privacy**: Ensure `author_raw` contains names but `author_uuid` contains deterministic UUID5s.
    *   **Normalization**: Verify standard WhatsApp attachment text (`(file attached)`) is rewritten to Markdown (`![Image](...)`).
    *   **Extraction**: Verify `deliver_media` successfully extracts bytes from the ZIP.

---

## 2. Core Pipeline Orchestration E2E
**Goal**: Verify the data flow from ingestion through windowing, tracking, and orchestration, ensuring the system correctly manages state and configuration.

### Scope
*   **Input**: Validated Ibis Table (from step 1).
*   **System**: `write_pipeline.run`, `DuckDBStorageManager`, `create_windows`.
*   **Output**: Run tracking database rows, checkpoint files, and agent tool calls.

### Test Plan
1.  **Mocking Strategy**:
    *   Patch `egregora.orchestration.write_pipeline.write_posts_with_pydantic_agent`.
    *   *Why*: This is the "AI Seam." By mocking this, we avoid API costs but test everything *around* it (windowing logic, context building, database logging).
2.  **Execution**:
    *   Initialize a fresh `PipelineContext` and `EgregoraConfig` (disable RAG for speed).
    *   Execute `run(source="whatsapp", ...)` with specific windowing params (e.g., `step_size=10`).
3.  **Assertions**:
    *   **Windowing**: Verify the mock agent was called $N$ times based on input size and step size.
    *   **Observability**: Query `.egregora/runs.duckdb` to ensure a run row exists with `status='completed'` and correct `rows_in`/`rows_out`.
    *   **Resumability**: Verify `.egregora/checkpoint.json` is created and contains the timestamp of the last processed message.

---

## 3. Output Adapter E2E
**Goal**: Verify that internal `Document` primitives are correctly serialized into the target static site format (MkDocs or Eleventy).

### Scope
*   **Input**: `Document` objects (Post, Profile, Enrichment).
*   **System**: `MkDocsAdapter` and `EleventyArrowAdapter`.
*   **Output**: Filesystem artifacts (`.md`, `.parquet`, `.yml`).

### Test Plan (MkDocs)
1.  **Setup**: Initialize `MkDocsAdapter` on a temp directory.
2.  **Execution**:
    *   Call `serve()` with a **Post** document (verifies slugification, frontmatter generation).
    *   Call `serve()` with a **Profile** document (verifies `.authors.yml` updates).
    *   Call `serve()` with a **Media Enrichment** document (verifies path resolution).
3.  **Assertions**:
    *   File existence: `posts/YYYY-MM-DD-slug.md` exists.
    *   Content integrity: Markdown content matches input; Frontmatter YAML is valid.
    *   Config side-effects: `.authors.yml` contains the author UUID.

### Test Plan (Eleventy/Arrow)
1.  **Setup**: Initialize `EleventyArrowAdapter`.
2.  **Execution**:
    *   Buffer multiple documents via `serve()`.
    *   Trigger `finalize_window("window-1")`.
3.  **Assertions**:
    *   Verify `data/window_0.parquet` exists.
    *   Read Parquet file using `pandas`/`pyarrow` and verify schema matches Document primitive columns (`kind`, `body_md`, `metadata`).

---

## 4. Implementation Reference
*See `tests/e2e/` for actual implementation.*

*   **Test Framework**: `pytest` with `tmp_path` fixture.
*   **Mocks**: `unittest.mock.patch`.
*   **Data Validation**: `ibis` for table inspection, `pyarrow` for Parquet validation.
