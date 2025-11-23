#!/usr/bin/env bash

# -------------------------------------------------------------------------- #
# Retrieves the most recent tag of the git project in the current working directory.
# Usage: get_most_recent_tag
# -------------------------------------------------------------------------- #
get_most_recent_tag() {
    get_nth_recent_tag 1
}

# -------------------------------------------------------------------------- #
# Retrieves the version number from the provided file location.
# Usage: get_version <version_file_path>
# -------------------------------------------------------------------------- #
get_version() {
    local version_loc="$1"
    if [[ -z "$version_loc" ]]; then
        echo "Error: version file path not provided" >&2
        return 1
    fi
    if [[ ! -f "$version_loc" ]]; then
        echo "Error: version file not found at $version_loc" >&2
        return 1
    fi
    grep -E -o "([0-9]{1,}\.)+[0-9]{1,}(.dev[0-9]{1,})?" "$version_loc" | head -n1
}

# -------------------------------------------------------------------------- #
# Retrieves the nth most recent tag of the git project.
# Usage: get_nth_recent_tag <n>
# -------------------------------------------------------------------------- #
get_nth_recent_tag() {
    local n="$1"
    if ! [[ "$n" =~ ^[0-9]+$ ]]; then
        echo "Error: Argument must be a positive integer" >&2
        return 1
    fi
    git fetch --tags --force --quiet
    local tags=($(git for-each-ref --sort=-creatordate --format '%(refname:strip=2)' refs/tags --count="$n"))
    if (( ${#tags[@]} < n )); then
        echo "Error: Less than $n tags found" >&2
        return 1
    fi
    echo "${tags[$((n-1))]}"
}

# -------------------------------------------------------------------------- #
# Bumps the version number based on release type ('patch', 'minor', 'major').
# Usage: bump_version <bump_type> <version_file_path>
# -------------------------------------------------------------------------- #
# bump_version() {
#     local bump_type="$1"
#     local version_loc="$2"
#     if [[ -z "$bump_type" || -z "$version_loc" ]]; then
#         echo "Usage: bump_version <patch|minor|major> <version_file_path>" >&2
#         return 1
#     fi
#     local version
#     version=$(get_version "$version_loc") || return 1
#     bump2version --current-version "$version" "$bump_type" "$version_loc" --verbose
# }

# -- Example usage for testing -- #
# version_loc="openpnm/__version__.py"
# get_most_recent_tag
# get_nth_recent_tag 2
# get_version "$version_loc"
# bump_version patch "$version_loc"
