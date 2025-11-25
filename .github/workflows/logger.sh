#!/usr/bin/env bash

set -euo pipefail

join_by() {
    local d="$1"; shift
    local first=1
    for f in "$@"; do
        if (( first )); then
            printf "%s" "$f"
            first=0
        else
            printf "%s%s" "$d" "$f"
        fi
    done
}

is_empty() {
    [[ "${1:-}" == "- " ]] && echo "true" || echo "false"
}

filter_commits_by_label() {
    local commits="$1"
    local label="$2"
    echo "$commits" | grep -Ei -- "$label" | \
        sed -E '/^\s*$/d' | \
        sed -E 's/^[[:space:]]*[-]?[[:space:]]*/- /' | \
        sed -E "s/[[:space:]]*$label\b//I"
}

filter_commits_exclude_label() {
    local commits="$1"
    local exclude_labels="$2"
    echo "$commits" | grep -Eiv -- "$exclude_labels" | \
        sed -E '/^\s*$/d' | \
        sed -E 's/^[[:space:]]*[-]?[[:space:]]*/- /'
}

filter_commits_by_tag_interval() {
    local tag_old="$1"
    local tag_new="$2"
    git log --merges "${tag_old}..${tag_new}" --format=%B 2>/dev/null | \
        grep -Ev "^Merge branch"
}

append_to_entry_with_label() {
    local content="$1"
    local file="$2"
    local label="$3"
    if [ "$(is_empty "$content")" = "false" ] && [ -n "$content" ]; then
        printf "### %s\n\n%s\n\n" "$label" "$content" >> "$file"
    fi
}

get_nth_recent_tag() {
    local n="$1"
    local tags
    tags=$(git for-each-ref --sort=-creatordate --format '%(refname:strip=2)' refs/tags --count="$n" | head -n "$n")
    local tag
    tag=$(echo "$tags" | sed -n "${n}p")
    if [[ -z "$tag" ]]; then
        echo "Error: Fewer than $n tags found." >&2
        exit 1
    fi
    echo "$tag"
}

# --- Main ---

tag_old=$(get_nth_recent_tag 2)
tag_new=$(get_nth_recent_tag 1)

merge_commits=$(filter_commits_by_tag_interval "$tag_old" "$tag_new")

# Section contents
features=$(filter_commits_by_label "$merge_commits" "#new")
enhancements=$(filter_commits_by_label "$merge_commits" "#enh")
maintenance=$(filter_commits_by_label "$merge_commits" "#maint")
changes=$(filter_commits_by_label "$merge_commits" "#api")
fixes=$(filter_commits_by_label "$merge_commits" "#bug")
documentation=$(filter_commits_by_label "$merge_commits" "#doc")

all_keywords=$(join_by "|" "#new" "#enh" "#maint" "#api" "#bug" "#doc" "#patch" "#minor" "#major")
uncategorized=$(filter_commits_exclude_label "$merge_commits" "$all_keywords")

# Prepare entry file
[ -f entry ] && rm -f entry
[ -f CHANGELOG.md ] && rm -f CHANGELOG.md

printf "## %s\n\n" "$tag_new" >> entry
append_to_entry_with_label "$features" entry ":rocket: New features"
append_to_entry_with_label "$enhancements" entry ":cake: Enhancements"
append_to_entry_with_label "$maintenance" entry ":wrench: Maintenance"
append_to_entry_with_label "$changes" entry ":warning: API changes"
append_to_entry_with_label "$fixes" entry ":bug: Bugfixes"
append_to_entry_with_label "$documentation" entry ":green_book: Documentation"
append_to_entry_with_label "$uncategorized" entry ":question: Uncategorized"

cat entry

printf "# Change log\n\n" > CHANGELOG.md
cat entry >> CHANGELOG.md
rm -f entry
