# IoT Knowledge Taxonomy Contract

`taxonomy.json` is a fixture-only taxonomy array. Each entry represents one IoT asset subcategory and must include `doc_id`, `coarse_category`, `sub_category`, `aliases`, `description`, `image_dir`, and `document_path`.

The production Visual RAG knowledge layout is expected to be:

```text
data/iot_knowledge/{coarse_category}/{sub_category}/document.md
data/iot_knowledge/{coarse_category}/{sub_category}/images/*
```

All paths stored in taxonomy entries must be relative paths. The current taxonomy contains only the three deterministic fixture categories used by tests; it is not a final production taxonomy.
