#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Publish the rustmpeg-next workspace crates to crates.io in dependency order.
#
# After every successful upload the script polls the crates.io API until the new
# version is visible, so the next dependent crate resolves. Rate limits are
# handled explicitly: on HTTP 429 the ``Please try again after <date>`` hint is
# parsed and honoured, with exponential backoff (60 s, 120 s, 300 s, 600 s,
# capped) as a fallback. Any other failure stops the script and prints the exact
# cargo error.
#
# Usage:
#   scripts/publish_crates.sh [--dry-run]
#
# CARGO_REGISTRY_TOKEN must be set for a real publish (cargo reads it directly).

set -euo pipefail

CRATES=(
    rustmpeg-core
    rustmpeg-format
    rustmpeg-audio
    rustmpeg-image
    rustmpeg-video
    rustmpeg-vulkan
    rustmpeg-cli
    rustmpeg
)

MAX_RETRIES=6
INDEX_TIMEOUT_SECONDS="${INDEX_TIMEOUT_SECONDS:-600}"
BACKOFF_SECONDS=(60 120 300 600)
USER_AGENT="rustmpeg-publish-script/1.0 (+https://github.com/sysv86/rustmpeg-next)"

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

log() {
    printf '==> %s\n' "$*"
}

workspace_version() {
    cargo metadata --format-version 1 --no-deps |
        python3 -c 'import json, sys; d = json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"] == "rustmpeg-core"))'
}

VERSION="$(workspace_version)"
log "workspace version ${VERSION}"

published_version() {
    local crate="$1"
    curl -sS -o /dev/null -w '%{http_code}' \
        -H "User-Agent: ${USER_AGENT}" \
        "https://crates.io/api/v1/crates/${crate}/${VERSION}"
}

wait_for_index() {
    local crate="$1"
    local deadline=$(( $(date +%s) + INDEX_TIMEOUT_SECONDS ))
    log "waiting for ${crate} ${VERSION} in the crates.io index"
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if [ "$(published_version "${crate}")" = "200" ]; then
            log "${crate} ${VERSION} is indexed"
            return 0
        fi
        sleep 5
    done
    echo "timed out after ${INDEX_TIMEOUT_SECONDS}s waiting for ${crate} ${VERSION}" >&2
    return 1
}

# Extract an RFC 2822 timestamp from a crates.io rate-limit message.
retry_after_epoch() {
    local text="$1"
    local date_text
    date_text="$(printf '%s' "${text}" |
        grep -oE '[A-Z][a-z]{2}, [0-9]{1,2} [A-Z][a-z]{2} [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT' |
        head -n1 || true)"
    if [ -n "${date_text}" ]; then
        date -u -d "${date_text}" +%s 2>/dev/null || true
    fi
}

publish_crate() {
    local crate="$1"
    local attempt=0
    local output
    output="$(mktemp)"

    while :; do
        if [ "${DRY_RUN}" -eq 1 ]; then
            log "dry-run cargo publish -p ${crate}"
        else
            log "cargo publish -p ${crate} (attempt $((attempt + 1)))"
        fi

        local -a publish_args=(-p "${crate}")
        if [ "${DRY_RUN}" -eq 1 ]; then
            publish_args+=(--dry-run)
        fi

        if cargo publish "${publish_args[@]}" >"${output}" 2>&1; then
            cat "${output}"
            rm -f "${output}"
            return 0
        fi

        local message
        message="$(cat "${output}")"
        printf '%s\n' "${message}" >&2

        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "dry-run failed for ${crate}" >&2
            rm -f "${output}"
            return 1
        fi

        if ! printf '%s' "${message}" | grep -Eqi '(^|[^0-9])429([^0-9]|$)|too many requests|rate limit'; then
            echo "publish failed for ${crate}; not a rate limit, stopping" >&2
            rm -f "${output}"
            return 1
        fi

        if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
            echo "publish failed for ${crate} after ${MAX_RETRIES} retries" >&2
            rm -f "${output}"
            return 1
        fi

        local wait_seconds=0
        local retry_epoch
        retry_epoch="$(retry_after_epoch "${message}")"
        if [ -n "${retry_epoch}" ]; then
            local now
            now="$(date +%s)"
            if [ "${retry_epoch}" -gt "${now}" ]; then
                wait_seconds=$(( retry_epoch - now + 5 ))
            fi
        fi
        if [ "${wait_seconds}" -le 0 ]; then
            local index=$(( attempt < ${#BACKOFF_SECONDS[@]} ? attempt : ${#BACKOFF_SECONDS[@]} - 1 ))
            wait_seconds="${BACKOFF_SECONDS[${index}]}"
        fi

        attempt=$(( attempt + 1 ))
        log "rate limited; retrying ${crate} in ${wait_seconds}s (retry ${attempt}/${MAX_RETRIES})"
        rm -f "${output}"
        sleep "${wait_seconds}"
        output="$(mktemp)"
    done
}

for crate in "${CRATES[@]}"; do
    if [ "${DRY_RUN}" -eq 0 ] && [ "$(published_version "${crate}")" = "200" ]; then
        log "${crate} ${VERSION} is already published; skipping"
        continue
    fi
    publish_crate "${crate}"
    if [ "${DRY_RUN}" -eq 0 ]; then
        wait_for_index "${crate}"
    fi
done

log "all crates processed"
