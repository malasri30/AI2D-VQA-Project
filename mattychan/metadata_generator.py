"""
AI2D Dataset Metadata Extractor
================================
Produces:
  - metadata_images.csv       : one row per image (annotation + question counts, flags)
  - metadata_questions.csv    : one row per question
  - metadata_integrity.csv    : one row per data issue found
  - metadata_full.json        : full nested metadata per image

Usage:
  python ai2d_metadata.py --root /path/to/ai2d
  python ai2d_metadata.py --root /path/to/ai2d --output /path/to/output_dir
"""

import os
import json
import csv
import math
import argparse
from collections import defaultdict


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"File read error: {e}"


def polygon_point_count(poly):
    return len(poly) if poly else 0


def bounding_box_area(rect):
    """rect is [[x1,y1],[x2,y2]]"""
    try:
        w = abs(rect[1][0] - rect[0][0])
        h = abs(rect[1][1] - rect[0][1])
        return w * h
    except Exception:
        return None


def polygon_approx_area(polygon):
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def extract_element_ids(annotation):
    """All defined element IDs in an annotation file."""
    ids = set()
    for section in ["blobs", "arrows", "arrowHeads", "containers", "meaningfulSpaces", "text", "imageConsts"]:
        ids.update(annotation.get(section, {}).keys())
    return ids


# ─────────────────────────────────────────────
# ANNOTATION METADATA
# ─────────────────────────────────────────────

def parse_annotation(ann, image_name):
    """Returns (summary_dict, detail_dict, issues_list)"""
    issues = []
    defined_ids = extract_element_ids(ann)

    blobs      = ann.get("blobs", {})
    arrows     = ann.get("arrows", {})
    arrowheads = ann.get("arrowHeads", {})
    containers = ann.get("containers", {})
    spaces     = ann.get("meaningfulSpaces", {})
    texts      = ann.get("text", {})
    image_consts = ann.get("imageConsts", {})
    relationships = ann.get("relationships", {})

    # ── Relationship analysis ──
    rel_categories = defaultdict(int)
    rel_directionality_count = 0
    referenced_ids = set()
    orphaned_ids = set()

    for rel_id, rel in relationships.items():
        cat = rel.get("category", "unknown")
        rel_categories[cat] += 1

        if rel.get("hasDirectionality"):
            rel_directionality_count += 1

        origin = rel.get("origin")
        dest   = rel.get("destination")
        conn   = rel.get("connector")

        for eid in [origin, dest, conn]:
            if eid:
                referenced_ids.add(eid)
                if eid not in defined_ids:
                    issues.append({
                        "image": image_name,
                        "issue_type": "dangling_relationship_reference",
                        "severity": "ERROR",
                        "detail": f"Relationship '{rel_id}' references undefined element '{eid}'"
                    })

    # ── Orphaned elements (defined but never in any relationship) ──
    for eid in defined_ids:
        if eid not in referenced_ids:
            orphaned_ids.add(eid)
            # imageConsts being unreferenced is fairly normal — lower severity
            severity = "WARNING" if eid.startswith("I") else "INFO"
            issues.append({
                "image": image_name,
                "issue_type": "orphaned_element",
                "severity": severity,
                "detail": f"Element '{eid}' is defined but not referenced in any relationship"
            })

    # ── Text analysis ──
    text_details = []
    for tid, t in texts.items():
        val   = t.get("value", "")
        repl  = t.get("replacementText", "")
        rect  = t.get("rectangle")
        area  = bounding_box_area(rect) if rect else None

        if not val or val.strip() == "":
            issues.append({
                "image": image_name,
                "issue_type": "empty_text_value",
                "severity": "WARNING",
                "detail": f"Text element '{tid}' has empty 'value'"
            })
        if not repl or repl.strip() == "":
            issues.append({
                "image": image_name,
                "issue_type": "empty_replacement_text",
                "severity": "WARNING",
                "detail": f"Text element '{tid}' has empty 'replacementText'"
            })

        text_details.append({
            "id": tid,
            "value": val,
            "replacementText": repl,
            "bbox_area": area,
            "in_relationship": tid in referenced_ids
        })

    # ── Blob analysis ──
    blob_details = []
    for bid, b in blobs.items():
        poly = b.get("polygon", [])
        pts  = polygon_point_count(poly)
        area = polygon_approx_area(poly) if pts >= 3 else 0

        if pts < 3:
            issues.append({
                "image": image_name,
                "issue_type": "malformed_polygon",
                "severity": "ERROR",
                "detail": f"Blob '{bid}' has only {pts} polygon points (minimum 3 needed)"
            })

        blob_details.append({
            "id": bid,
            "polygon_points": pts,
            "approx_area": round(area, 2),
            "in_relationship": bid in referenced_ids
        })

    # ── Arrow analysis ──
    arrow_details = []
    for aid, a in arrows.items():
        poly = a.get("polygon", [])
        pts  = polygon_point_count(poly)

        if pts < 2:
            issues.append({
                "image": image_name,
                "issue_type": "malformed_polygon",
                "severity": "ERROR",
                "detail": f"Arrow '{aid}' has only {pts} polygon points"
            })

        arrow_details.append({
            "id": aid,
            "polygon_points": pts,
            "in_relationship": aid in referenced_ids
        })

    # ── Summary dict (for images CSV) ──
    summary = {
        "blob_count":           len(blobs),
        "arrow_count":          len(arrows),
        "arrowhead_count":      len(arrowheads),
        "container_count":      len(containers),
        "meaningful_space_count": len(spaces),
        "text_count":           len(texts),
        "image_const_count":    len(image_consts),
        "relationship_count":   len(relationships),
        "orphaned_element_count": len(orphaned_ids),
        "rel_with_directionality": rel_directionality_count,
        "rel_categories":       dict(rel_categories),   # kept as dict for JSON
        # flattened rel category counts for CSV
        **{f"rel_cat_{k}": v for k, v in rel_categories.items()},
    }

    # ── Full detail dict (for full JSON) ──
    detail = {
        "blobs":         blob_details,
        "arrows":        arrow_details,
        "arrowHeads":    list(arrowheads.keys()),
        "containers":    list(containers.keys()),
        "meaningfulSpaces": list(spaces.keys()),
        "texts":         text_details,
        "imageConsts":   list(image_consts.keys()),
        "relationships": {
            "count": len(relationships),
            "categories": dict(rel_categories),
            "directional_count": rel_directionality_count,
            "ids": list(relationships.keys())
        },
        "orphaned_ids":  list(orphaned_ids),
        "defined_ids":   list(defined_ids),
    }

    return summary, detail, issues


# ─────────────────────────────────────────────
# QUESTION METADATA
# ─────────────────────────────────────────────

def parse_questions(q_data, image_name):
    """Returns (summary_dict, question_rows_list, issues_list)"""
    issues = []
    q_rows = []

    questions = q_data.get("questions", {})
    seen_ids  = set()

    abc_label_count  = 0
    desc_label_count = 0
    answer_counts    = []
    correct_indices  = []

    for q_text, q in questions.items():
        qid         = q.get("questionId", "")
        abc_label   = q.get("abcLabel", None)
        answers     = q.get("answerTexts", [])
        correct_idx = q.get("correctAnswer", None)

        # ── duplicate question ID ──
        if qid in seen_ids:
            issues.append({
                "image": image_name,
                "issue_type": "duplicate_question_id",
                "severity": "ERROR",
                "detail": f"Question ID '{qid}' appears more than once"
            })
        seen_ids.add(qid)

        # ── correctAnswer out of bounds ──
        if correct_idx is not None and len(answers) > 0:
            if correct_idx < 0 or correct_idx >= len(answers):
                issues.append({
                    "image": image_name,
                    "issue_type": "correct_answer_out_of_bounds",
                    "severity": "ERROR",
                    "detail": f"Question '{qid}': correctAnswer={correct_idx} but only {len(answers)} answers"
                })

        # ── missing fields ──
        if not qid:
            issues.append({
                "image": image_name,
                "issue_type": "missing_question_id",
                "severity": "WARNING",
                "detail": f"A question has no questionId: '{q_text[:60]}...'"
            })
        if abc_label is None:
            issues.append({
                "image": image_name,
                "issue_type": "missing_abc_label",
                "severity": "WARNING",
                "detail": f"Question '{qid}' has no abcLabel field"
            })
        if not answers:
            issues.append({
                "image": image_name,
                "issue_type": "no_answer_choices",
                "severity": "ERROR",
                "detail": f"Question '{qid}' has no answerTexts"
            })
        if correct_idx is None:
            issues.append({
                "image": image_name,
                "issue_type": "missing_correct_answer",
                "severity": "ERROR",
                "detail": f"Question '{qid}' has no correctAnswer"
            })

        # ── empty question text ──
        if not q_text or q_text.strip() == "":
            issues.append({
                "image": image_name,
                "issue_type": "empty_question_text",
                "severity": "ERROR",
                "detail": f"Question '{qid}' has empty question text"
            })

        if abc_label:
            abc_label_count += 1
        else:
            desc_label_count += 1

        answer_counts.append(len(answers))
        if correct_idx is not None:
            correct_indices.append(correct_idx)

        q_rows.append({
            "image_name":       image_name,
            "question_id":      qid,
            "question_text":    q_text,
            "question_length":  len(q_text),
            "abc_label":        abc_label,
            "answer_count":     len(answers),
            "correct_answer_index": correct_idx,
            "correct_answer_text": answers[correct_idx] if (correct_idx is not None and 0 <= correct_idx < len(answers)) else "",
            "answer_texts":     " | ".join(answers),
        })

    summary = {
        "question_count":      len(questions),
        "abc_label_questions": abc_label_count,
        "desc_label_questions": desc_label_count,
        "unique_answer_counts": list(set(answer_counts)),
        "correct_index_distribution": dict(
            sorted(defaultdict(int, {str(i): correct_indices.count(i) for i in set(correct_indices)}).items())
        ),
    }

    return summary, q_rows, issues


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run(root_dir, output_dir):
    images_dir      = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, "annotations")
    questions_dir   = os.path.join(root_dir, "questions")
    categories_path = os.path.join(root_dir, "categories.json")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load categories ──
    categories, cat_err = load_json(categories_path)
    if cat_err:
        print(f"[ERROR] Could not load categories.json: {cat_err}")
        categories = {}

    # ── Collect all image names ──
    image_files = set()
    if os.path.isdir(images_dir):
        image_files = {f for f in os.listdir(images_dir) if f.endswith(".png")}
    else:
        print(f"[WARNING] Images directory not found: {images_dir}")

    annotation_files = set()
    if os.path.isdir(annotations_dir):
        annotation_files = {f.replace(".json", "") for f in os.listdir(annotations_dir) if f.endswith(".json")}

    question_files = set()
    if os.path.isdir(questions_dir):
        question_files = {f.replace(".json", "") for f in os.listdir(questions_dir) if f.endswith(".json")}

    # Union of all known image names across all sources
    all_images = (
        image_files
        | {f for f in annotation_files}
        | {f for f in question_files}
        | set(categories.keys())
    )

    print(f"  Total unique image names found across all sources: {len(all_images)}")
    print(f"  Images dir:      {len(image_files)}")
    print(f"  Annotations dir: {len(annotation_files)}")
    print(f"  Questions dir:   {len(question_files)}")
    print(f"  Categories:      {len(categories)}")

    # ── Accumulators ──
    all_integrity_issues = []
    all_question_rows    = []
    full_metadata        = {}
    image_rows           = []

    # ── Category distribution ──
    category_dist = defaultdict(int)
    for img, cat in categories.items():
        category_dist[cat] += 1

    for image_name in sorted(all_images, key=lambda x: int(x.replace(".png", "")) if x.replace(".png", "").isdigit() else float("inf")):

        row = {"image_name": image_name}
        full_entry = {"image_name": image_name}

        # ── File presence checks ──
        has_image      = image_name in image_files
        has_annotation = image_name in annotation_files
        has_question   = image_name in question_files
        has_category   = image_name in categories

        row["has_image_file"]      = has_image
        row["has_annotation_file"] = has_annotation
        row["has_question_file"]   = has_question
        row["has_category"]        = has_category
        row["category"]            = categories.get(image_name, "")

        if not has_image:
            all_integrity_issues.append({
                "image": image_name, "issue_type": "missing_image_file",
                "severity": "ERROR", "detail": f"No .png file found for {image_name}"
            })
        if not has_annotation:
            all_integrity_issues.append({
                "image": image_name, "issue_type": "missing_annotation_file",
                "severity": "ERROR", "detail": f"No annotation JSON found for {image_name}"
            })
        if not has_question:
            all_integrity_issues.append({
                "image": image_name, "issue_type": "missing_question_file",
                "severity": "INFO", "detail": f"No question JSON found for {image_name}"
            })
        if not has_category:
            all_integrity_issues.append({
                "image": image_name, "issue_type": "missing_category",
                "severity": "WARNING", "detail": f"No category entry in categories.json for {image_name}"
            })

        # ── Parse annotation ──
        ann_summary, ann_detail, ann_issues = {}, {}, []
        if has_annotation:
            ann_path = os.path.join(annotations_dir, image_name + ".json")
            ann_data, err = load_json(ann_path)
            if err:
                all_integrity_issues.append({
                    "image": image_name, "issue_type": "annotation_parse_error",
                    "severity": "ERROR", "detail": err
                })
            else:
                ann_summary, ann_detail, ann_issues = parse_annotation(ann_data, image_name)
                all_integrity_issues.extend(ann_issues)

        row.update(ann_summary)
        full_entry["annotation"] = ann_detail

        # ── Parse questions ──
        q_summary, q_rows_for_image, q_issues = {}, [], []
        if has_question:
            q_path = os.path.join(questions_dir, image_name + ".json")
            q_data, err = load_json(q_path)
            if err:
                all_integrity_issues.append({
                    "image": image_name, "issue_type": "question_parse_error",
                    "severity": "ERROR", "detail": err
                })
            else:
                # imageName consistency check
                q_image_name = q_data.get("imageName", "")
                if q_image_name != image_name:
                    all_integrity_issues.append({
                        "image": image_name, "issue_type": "imageName_mismatch",
                        "severity": "WARNING",
                        "detail": f"Question file imageName='{q_image_name}' but filename implies '{image_name}'"
                    })
                q_summary, q_rows_for_image, q_issues = parse_questions(q_data, image_name)
                all_integrity_issues.extend(q_issues)
                all_question_rows.extend(q_rows_for_image)
        else:
            q_summary = {"question_count": 0}

        row.update({f"q_{k}": v for k, v in q_summary.items() if not isinstance(v, (dict, list))})
        row["q_question_count"] = q_summary.get("question_count", 0)
        full_entry["questions"] = {
            "summary": q_summary,
            "items":   q_rows_for_image
        }

        # ── Issue count per image ──
        image_issue_count = sum(
            1 for iss in (ann_issues + q_issues)
        )
        row["issue_count"] = image_issue_count

        full_entry["integrity"] = {
            "has_image": has_image,
            "has_annotation": has_annotation,
            "has_question": has_question,
            "has_category": has_category,
            "issue_count": image_issue_count,
            "issues": ann_issues + q_issues
        }

        image_rows.append(row)
        full_metadata[image_name] = full_entry

    # ─────────────────────────────────────────
    # WRITE OUTPUTS
    # ─────────────────────────────────────────

    # Collect all column names dynamically for images CSV
    all_img_keys = []
    seen_keys = set()
    priority_keys = [
        "image_name", "category", "has_image_file", "has_annotation_file",
        "has_question_file", "has_category", "issue_count",
        "blob_count", "arrow_count", "arrowhead_count", "container_count",
        "meaningful_space_count", "text_count", "image_const_count",
        "relationship_count", "orphaned_element_count", "rel_with_directionality",
        "q_question_count", "q_abc_label_questions", "q_desc_label_questions",
    ]
    for k in priority_keys:
        seen_keys.add(k)
        all_img_keys.append(k)
    for row in image_rows:
        for k in row:
            if k not in seen_keys and not isinstance(row[k], (dict, list)):
                seen_keys.add(k)
                all_img_keys.append(k)

    # 1. metadata_images.csv
    images_csv_path = os.path.join(output_dir, "metadata_images.csv")
    with open(images_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_img_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)
    print(f"\n  ✓ metadata_images.csv       ({len(image_rows)} rows)")

    # 2. metadata_questions.csv
    q_keys = [
        "image_name", "question_id", "question_text", "question_length",
        "abc_label", "answer_count", "correct_answer_index",
        "correct_answer_text", "answer_texts"
    ]
    questions_csv_path = os.path.join(output_dir, "metadata_questions.csv")
    with open(questions_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=q_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_question_rows)
    print(f"  ✓ metadata_questions.csv    ({len(all_question_rows)} rows)")

    # 3. metadata_integrity.csv
    int_keys = ["image", "issue_type", "severity", "detail"]
    integrity_csv_path = os.path.join(output_dir, "metadata_integrity.csv")
    with open(integrity_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=int_keys)
        writer.writeheader()
        writer.writerows(all_integrity_issues)
    print(f"  ✓ metadata_integrity.csv    ({len(all_integrity_issues)} issues)")

    # 4. metadata_full.json
    full_json_path = os.path.join(output_dir, "metadata_full.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2)
    print(f"  ✓ metadata_full.json        ({len(full_metadata)} images)")

    # ── Console summary ──
    print("\n" + "─" * 50)
    print("DATASET SUMMARY")
    print("─" * 50)
    print(f"  Total images tracked:        {len(all_images)}")
    print(f"  With image file:             {sum(1 for r in image_rows if r.get('has_image_file'))}")
    print(f"  With annotation:             {sum(1 for r in image_rows if r.get('has_annotation_file'))}")
    print(f"  With questions:              {sum(1 for r in image_rows if r.get('has_question_file'))}")
    print(f"  With category:               {sum(1 for r in image_rows if r.get('has_category'))}")
    print(f"  Images with 0 questions:     {sum(1 for r in image_rows if r.get('q_question_count', 0) == 0)}")
    print(f"  Total questions:             {len(all_question_rows)}")
    print(f"  Total integrity issues:      {len(all_integrity_issues)}")

    error_count   = sum(1 for i in all_integrity_issues if i["severity"] == "ERROR")
    warning_count = sum(1 for i in all_integrity_issues if i["severity"] == "WARNING")
    info_count    = sum(1 for i in all_integrity_issues if i["severity"] == "INFO")
    print(f"    → ERRORs:   {error_count}")
    print(f"    → WARNINGs: {warning_count}")
    print(f"    → INFOs:    {info_count}")

    print("\n  Category distribution:")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat:<30} {count}")

    print("\n  Correct answer index distribution (across all questions):")
    idx_dist = defaultdict(int)
    for q in all_question_rows:
        if q["correct_answer_index"] is not None:
            idx_dist[q["correct_answer_index"]] += 1
    for idx, count in sorted(idx_dist.items()):
        print(f"    Index {idx}: {count} questions")

    print(f"\n  Output written to: {output_dir}")
    print("─" * 50)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI2D Dataset Metadata Extractor")
    parser.add_argument("--root",   required=True, help="Path to the ai2d root directory")
    parser.add_argument("--output", default=None,  help="Output directory (default: <root>/metadata_output)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out  = args.output if args.output else os.path.join(root, "metadata_output")

    print(f"\nAI2D Metadata Extractor")
    print(f"  Root:   {root}")
    print(f"  Output: {out}\n")

    run(root, out)