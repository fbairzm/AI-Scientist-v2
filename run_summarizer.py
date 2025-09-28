#!/usr/bin/env python3
"""Run the project's overall_summarize on a saved experiment folder.

Usage:
    python run_summarizer.py /path/to/experiment_folder

This will load stage folders named 'stage_{n}', read their journal.json files,
reconstruct Journal objects, call overall_summarize, and write four summary JSONs
into the experiment folder.
"""
import sys
import os
import json

from ai_scientist.treesearch.journal import Node, Journal
from ai_scientist.treesearch.log_summarization import overall_summarize


def load_stage_folders(base_path):
    """Find stage_* folders under base_path.

    The experiment layout sometimes places stages under logs/0-run/. We search
    base_path and one level below for directories named stage_*
    """
    stage_folders = []
    # direct children
    for entry in os.listdir(base_path):
        p = os.path.join(base_path, entry)
        if os.path.isdir(p) and entry.startswith("stage_"):
            stage_folders.append(p)
    # check logs and logs/0-run
    logs_dir = os.path.join(base_path, "logs")
    if os.path.isdir(logs_dir):
        for entry in os.listdir(logs_dir):
            p = os.path.join(logs_dir, entry)
            if os.path.isdir(p) and entry.startswith("stage_"):
                stage_folders.append(p)
            # check logs/0-run/
            if entry == "0-run":
                run_dir = p
                for sub in os.listdir(run_dir):
                    sp = os.path.join(run_dir, sub)
                    if os.path.isdir(sp) and sub.startswith("stage_"):
                        stage_folders.append(sp)

    # Deduplicate and sort by numeric suffix if possible
    unique = sorted(list(dict.fromkeys(stage_folders)))
    def sort_key(x):
        base = os.path.basename(x)
        parts = base.split("_")
        # try to find first numeric part
        for part in parts:
            if part.isdigit():
                return int(part)
        # fallback to name
        return base

    return sorted(unique, key=sort_key)


def reconstruct_journal(journal_data):
    id_to_node = {}
    for node_data in journal_data.get("nodes", []):
        if "actionable_insights_from_plots" in node_data:
            del node_data["actionable_insights_from_plots"]
        node = Node.from_dict(node_data)
        id_to_node[node.id] = node

    for node_id, parent_id in journal_data.get("node2parent", {}).items():
        child_node = id_to_node.get(node_id)
        parent_node = id_to_node.get(parent_id)
        if child_node and parent_node:
            child_node.parent = parent_node
            parent_node.children.add(child_node)

    journal = Journal()
    journal.nodes.extend(id_to_node.values())
    return journal


def main(exp_dir: str):
    if not os.path.isdir(exp_dir):
        print("Not a directory:", exp_dir)
        return 2

    stage_folders = load_stage_folders(exp_dir)
    if not stage_folders:
        print("No stage_* folders found under", exp_dir)
        return 2

    journals = []
    for idx, folder in enumerate(stage_folders, start=1):
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if not os.path.exists(journal_path):
            print(f"Warning: {journal_path} not found — skipping {stage_name}")
            continue
        with open(journal_path, "r") as f:
            journal_data = json.load(f)
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    if not journals:
        print("No journals loaded; nothing to summarize.")
        return 2

    print(f"Loaded {len(journals)} journals; calling overall_summarize()")
    try:
        draft, baseline, research, ablation = overall_summarize(journals)
    except Exception as e:
        print("Error while running overall_summarize:", e)
        import traceback

        traceback.print_exc()
        return 3

    def write_json(obj, name):
        outp = os.path.join(exp_dir, name)
        with open(outp, "w") as fw:
            json.dump(obj, fw, indent=2)
        print("Wrote", outp)

    if draft is not None:
        write_json(draft, "draft_summary.json")
    if baseline is not None:
        write_json(baseline, "baseline_summary.json")
    if research is not None:
        write_json(research, "research_summary.json")
    if ablation is not None:
        write_json(ablation, "ablation_summary.json")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_summarizer.py /path/to/experiment_dir")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
