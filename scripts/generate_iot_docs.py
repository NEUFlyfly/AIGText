#!/usr/bin/env python3
"""
从 iot_taxonomy.json 生成 document.md 文件与 flat taxonomy.json

结构:
  data/iot_knowledge/
    iot_taxonomy.json          (源 taxonomy)
    taxonomy.json              (生成的 flat taxonomy)
    {coarse_id}/{sub_id}/
      document.md              (子类别的 intro 正文)
"""

import json
import os
import sys
from pathlib import Path


IOT_KNOWLEDGE_DIR = Path("data/iot_knowledge")
SOURCE_TAXONOMY = IOT_KNOWLEDGE_DIR / "iot_taxonomy.json"
OUTPUT_TAXONOMY = IOT_KNOWLEDGE_DIR / "taxonomy.json"


def generate_documents(taxonomy_path: Path, output_dir: Path) -> int:
    """Read iot_taxonomy.json and create document.md files for each subcategory.

    Returns the number of document.md files generated.
    """
    if not taxonomy_path.exists():
        print(f"[ERROR] Taxonomy not found: {taxonomy_path}", file=sys.stderr)
        sys.exit(1)

    with taxonomy_path.open("r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    generated_count = 0
    flat_entries: list[dict[str, str]] = []

    for coarse in taxonomy["coarse_categories"]:
        coarse_id = coarse["id"]
        coarse_name = coarse["name"]

        for sub in coarse["subclasses"]:
            sub_id = sub["id"]
            sub_name = sub["name"]
            intro = sub["intro"]

            # Build directory: {coarse_id}/{sub_id}/
            sub_dir = output_dir / coarse_id / sub_id
            sub_dir.mkdir(parents=True, exist_ok=True)

            # Build document.md with richer content
            aliases = sub.get("aliases", [])
            aliases_text = ", ".join(aliases) if aliases else "N/A"
            content = (
                f"# {sub_name}\n\n"
                f"**Category**: {coarse_name}\n\n"
                f"**Also known as**: {aliases_text}\n\n"
                f"{intro}\n"
            )
            document_path = sub_dir / "document.md"
            document_path.write_text(content, encoding="utf-8")

            # Record flat taxonomy entry
            rel_doc_path = str(document_path).replace("\\", "/")
            flat_entries.append({
                "doc_id": f"{coarse_id}/{sub_id}",
                "coarse_category": coarse_name,
                "sub_category": sub_name,
                "document_path": rel_doc_path,
            })
            generated_count += 1

    # Write flat taxonomy.json
    with OUTPUT_TAXONOMY.open("w", encoding="utf-8") as f:
        json.dump(flat_entries, f, indent=2, ensure_ascii=False)

    return generated_count


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 60)
    print("  IoT Knowledge Document Generator")
    print("=" * 60)

    if not SOURCE_TAXONOMY.exists():
        print(f"[ERROR] Source taxonomy missing: {SOURCE_TAXONOMY}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Source] {SOURCE_TAXONOMY}")
    print(f"[Output] {IOT_KNOWLEDGE_DIR}/")

    count = generate_documents(SOURCE_TAXONOMY, IOT_KNOWLEDGE_DIR)

    print(f"\n  Generated {count} document.md files")
    print(f"  Written flat taxonomy: {OUTPUT_TAXONOMY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
