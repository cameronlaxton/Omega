import os
import subprocess


def get_tracked_files():
    try:
        output = subprocess.check_output(["git", "ls-files"], text=True)
        return [f.replace("/", os.sep) for f in output.strip().split("\n") if f]
    except Exception as e:
        print(f"Error getting tracked files: {e}")
        return []


def validate_reports_metadata():
    tracked_files = get_tracked_files()
    reports_md = [
        f for f in tracked_files if f.startswith(f"reports{os.sep}") and f.endswith(".md")
    ]
    failed = False
    for md_file in reports_md:
        if not os.path.exists(md_file):
            continue
        with open(md_file, encoding="utf-8") as f:
            content = f.read()
            if "canonical: false" not in content:
                print(f"ERROR: {md_file} is missing 'canonical: false' metadata.")
                failed = True
    return not failed


def validate_terminology():
    banned_words = ["best bet", "tier a", "tier b", "engine-confirmed", "actionable bet"]
    tracked_files = get_tracked_files()
    files_to_scan = [f for f in tracked_files if f.endswith(".md") or f.endswith(".txt")]

    # Exclude historical/archived paths and reports
    exclude_paths = [
        "archive\\",
        "archive/",
        "docs\\history",
        "docs/history",
        "docs\\phase7",
        "docs/phase7",
        "plugins\\",
        "plugins/",
        "reports\\",
        "var/reports/",
        "inbox\\",
        "var/inbox/",
        "docs\\session_",
        "docs/session_",
        "docs\\qa",
        "docs/qa",
        "docs\\phase6\\HANDOFF",
    ]

    failed = False
    for filepath in files_to_scan:
        if any(excl in filepath for excl in exclude_paths):
            continue
        # A file can be tracked but deleted in the working tree (git status 'D').
        # Skip it here — its absence is surfaced by git/artifact-policy, not by a
        # crash in the terminology scan.
        if not os.path.exists(filepath):
            continue

        with open(filepath, encoding="utf-8") as f:
            try:
                content = f.read().lower()
                for word in banned_words:
                    if word.lower() in content:
                        # Check if the word is explicitly in a "banned list" or forbidden context
                        if (
                            "forbidden language" not in content
                            and "blocked phrases" not in content
                            and "banned terminology" not in content
                        ):
                            print(f"WARNING/ERROR: Banned terminology '{word}' found in {filepath}")
                            failed = True
            except UnicodeDecodeError:
                pass

    return not failed


def main():
    print("Running Repository State Validation...")
    reports_ok = validate_reports_metadata()
    terms_ok = validate_terminology()

    if reports_ok and terms_ok:
        print("SUCCESS: All repository state checks passed.")
        exit(0)
    else:
        print("FAILED: Repository state checks failed.")
        exit(1)


if __name__ == "__main__":
    main()
