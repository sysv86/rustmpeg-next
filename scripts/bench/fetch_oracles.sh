#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the benchmark oracles outside the repository.  The oracles are
# black-box reference tools; no oracle source is vendored, linked, or copied
# into rustmpeg.
#
# Usage:
#   scripts/bench/fetch_oracles.sh [--dest DIR] [--jobs N] [--force]
#                                  [--only LIST] [--skip LIST]
#                                  [--best-effort] [--build-timeout SECONDS]
#
#   LIST is a comma-separated subset of: svt,avm,dav2d,lc3
#   ("all" selects every oracle, "none" selects no oracle).
#
# Environment:
#   BENCH_OFFLINE=1   Do not clone or build.  Fail if a selected oracle is
#                     missing from DEST/bin.
#
# The script writes DEST/oracles.json with the resolved versions and the list
# of oracles that were skipped.  A skipped oracle is reported as SKIP and is
# never replaced by a different build or by an invented version.  Set
# LD_LIBRARY_PATH=DEST/bin when running liblc3 tools: they are built with a
# shared liblc3.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/bench/oracles.env
source "${here}/oracles.env"

dest="${BENCH_ORACLE_DIR:-${HOME}/.cache/rustmpeg-bench/oracles}"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
force=0
only="all"
skip_list=""
best_effort=0
build_timeout=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            dest="$2"
            shift 2
            ;;
        --jobs)
            jobs="$2"
            shift 2
            ;;
        --only)
            only="$2"
            shift 2
            ;;
        --skip)
            skip_list="$2"
            shift 2
            ;;
        --best-effort)
            best_effort=1
            shift
            ;;
        --build-timeout)
            build_timeout="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        -h|--help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

dest="$(mkdir -p "${dest}" && cd "${dest}" && pwd)"
bin="${dest}/bin"
src_root="${dest}/src"
build_root="${dest}/build"
log_root="${dest}/logs"
mkdir -p "${bin}" "${src_root}" "${build_root}" "${log_root}"
export PATH="${bin}:${PATH}"
export LD_LIBRARY_PATH="${bin}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GIT_TERMINAL_PROMPT=0

log() {
    printf '[oracles] %s\n' "$*" >&2
}

need_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing build tool: $1" >&2
        echo "install: sudo apt-get install -y build-essential git cmake ninja-build meson nasm yasm pkg-config libxxhash-dev python3 perl ffmpeg flac" >&2
        return 1
    fi
}

offline() {
    [[ "${BENCH_OFFLINE:-0}" == "1" ]]
}

limit() {
    if [[ ${build_timeout} -gt 0 ]]; then
        timeout "${build_timeout}" "$@"
    else
        "$@"
    fi
}

clone_at() {
    local url="$1" ref="$2" dir="$3"
    if [[ -d "${dir}/.git" && ${force} -eq 0 ]]; then
        log "source already present: ${dir}"
        return 0
    fi
    rm -rf "${dir}"
    log "cloning ${url} @ ${ref}"
    limit git clone --filter=blob:none --no-checkout "${url}" "${dir}" >/dev/null 2>&1 || return 1
    limit git -C "${dir}" fetch --depth 1 origin "${ref}" >/dev/null 2>&1 || return 1
    git -C "${dir}" checkout --detach FETCH_HEAD >/dev/null 2>&1 || return 1
}

# copy_first_executable NAME ROOT...
#
# Some CMake projects place binaries outside their build directory, so every
# candidate root is searched.  SVT-AV1 writes to <source>/Bin/<config> unless
# CMAKE_OUTPUT_DIRECTORY is overridden.
copy_first_executable() {
    local name="$1"
    shift
    local root found=""
    for root in "$@"; do
        [[ -d "${root}" ]] || continue
        found="$(find "${root}" -type f -name "${name}" -perm -u+x -print -quit)"
        if [[ -n "${found}" ]]; then
            break
        fi
    done
    if [[ -z "${found}" ]]; then
        echo "built target not found: ${name} under $*" >&2
        return 1
    fi
    cp "${found}" "${bin}/${name}"
    chmod +x "${bin}/${name}"
    log "installed ${bin}/${name}"
}

ORACLES=(svt avm dav2d lc3)

declare -A selected=()
declare -A want=()

select_oracle() {
    local name="$1"
    case "${name}" in
        svt|avm|dav2d|lc3) ;;
        *)
            echo "unknown oracle '${name}' (expected: ${ORACLES[*]})" >&2
            exit 2
            ;;
    esac
    selected["${name}"]=1
}

if [[ "${only}" == "all" ]]; then
    for name in "${ORACLES[@]}"; do
        select_oracle "${name}"
    done
elif [[ "${only}" == "none" ]]; then
    :
else
    IFS=',' read -r -a names <<<"${only}"
    for name in "${names[@]}"; do
        [[ -n "${name}" ]] && select_oracle "${name}"
    done
fi

if [[ -n "${skip_list}" ]]; then
    IFS=',' read -r -a names <<<"${skip_list}"
    for name in "${names[@]}"; do
        [[ -z "${name}" ]] && continue
        if [[ -z "${selected[${name}]:-}" ]]; then
            echo "--skip ${name}: oracle was not selected" >&2
            continue
        fi
        unset 'selected[$name]'
    done
fi

skipped=()
declare -A skipped_names=()

skip_oracle() {
    local name="$1" reason="$2"
    if [[ -n "${skipped_names[${name}]:-}" ]]; then
        return 0
    fi
    log "SKIP ${name}: ${reason}"
    skipped+=("${name}|${reason}")
    skipped_names["${name}"]=1
    want["${name}"]=0
}

for name in "${ORACLES[@]}"; do
    if [[ -n "${selected[${name}]:-}" ]]; then
        want["${name}"]=1
    else
        want["${name}"]=0
    fi
done

if offline; then
    for name in "${ORACLES[@]}"; do
        want["${name}"]=0
    done
    log "offline mode: using binaries already present in ${bin}"
fi

build_svt() {
    need_command git || return 1
    need_command cmake || return 1
    need_command ninja || return 1
    clone_at "https://github.com/AOMediaCodec/SVT-AV1.git" "${ORACLE_SVT_AV1_COMMIT}" "${src_root}/SVT-AV1" || return 1
    log "building SVT-AV1 ${ORACLE_SVT_AV1_REF}"
    rm -rf "${build_root}/SVT-AV1"
    cmake -S "${src_root}/SVT-AV1" -B "${build_root}/SVT-AV1" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_APPS=ON \
        -DCMAKE_OUTPUT_DIRECTORY="${build_root}/SVT-AV1/bin" \
        -DCMAKE_INSTALL_PREFIX="${dest}/prefix-svtav1" || return 1
    limit cmake --build "${build_root}/SVT-AV1" --target SvtAv1EncApp -j "${jobs}" || return 1
    copy_first_executable SvtAv1EncApp \
        "${build_root}/SVT-AV1/bin" \
        "${build_root}/SVT-AV1" \
        "${src_root}/SVT-AV1/Bin" || return 1
}

build_avm() {
    need_command git || return 1
    need_command cmake || return 1
    need_command ninja || return 1
    need_command perl || return 1
    clone_at "https://github.com/AOMediaCodec/avm.git" "${ORACLE_AVM_COMMIT}" "${src_root}/avm" || return 1
    log "building AVM ${ORACLE_AVM_REF} (avmenc; avmdec optional fallback)"
    rm -rf "${build_root}/avm"
    cmake -S "${src_root}/avm" -B "${build_root}/avm" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CCACHE=0 \
        -DENABLE_DOCS=0 \
        -DENABLE_TESTS=0 \
        -DENABLE_EXAMPLES=1 \
        -DENABLE_TOOLS=0 \
        -DCONFIG_AV2_ENCODER=1 \
        -DCONFIG_AV2_DECODER=1 || return 1
    limit cmake --build "${build_root}/avm" --target avmenc -j "${jobs}" || return 1
    copy_first_executable avmenc "${build_root}/avm" || return 1
    if limit cmake --build "${build_root}/avm" --target avmdec -j "${jobs}"; then
        copy_first_executable avmdec "${build_root}/avm" \
            || log "avmdec built but was not installed; dav2d remains the AV2 decoder"
    else
        log "avmdec target unavailable; dav2d remains the AV2 decoder"
    fi
}

build_dav2d() {
    need_command git || return 1
    need_command meson || return 1
    need_command ninja || return 1
    clone_at "https://code.videolan.org/videolan/dav2d.git" "${ORACLE_DAV2D_REF}" "${src_root}/dav2d" || return 1
    log "building dav2d ${ORACLE_DAV2D_REF:0:8}"
    rm -rf "${build_root}/dav2d"
    meson setup "${build_root}/dav2d" "${src_root}/dav2d" \
        --buildtype=release \
        --default-library=static \
        --wrap-mode=nodownload \
        -Denable_tests=false \
        -Denable_docs=false \
        -Denable_examples=false \
        -Denable_tools=true || return 1
    limit ninja -C "${build_root}/dav2d" -j "${jobs}" || return 1
    copy_first_executable dav2d "${build_root}/dav2d" || return 1
}

build_lc3() {
    need_command git || return 1
    need_command make || return 1
    clone_at "https://github.com/google/liblc3.git" "${ORACLE_LIBC3_COMMIT}" "${src_root}/liblc3" || return 1
    log "building liblc3 ${ORACLE_LIBC3_REF} tools"
    limit make -C "${src_root}/liblc3" -j "${jobs}" tools || return 1
    cp -a "${src_root}/liblc3/bin/." "${bin}/" || return 1
    chmod +x "${bin}/elc3" "${bin}/dlc3" || return 1
    log "installed elc3 and dlc3"
}

build_oracle() {
    local name="$1" builder="$2"
    if [[ ${want[${name}]} -eq 0 ]]; then
        return 0
    fi
    if [[ ${force} -eq 0 && -x "${bin}/$(oracle_primary "${name}")" ]]; then
        log "already built: ${name}"
        return 0
    fi
    local build_log="${log_root}/${name}-build.log"
    : >"${build_log}"
    if "${builder}" >>"${build_log}" 2>&1; then
        return 0
    fi
    if [[ ${best_effort} -eq 1 ]]; then
        tail -n 20 "${build_log}" >&2 || true
        skip_oracle "${name}" "build failed or exceeded the timeout (see ${build_log})"
        return 0
    fi
    echo "oracle build failed or timed out: ${name} (see ${build_log})" >&2
    tail -n 20 "${build_log}" >&2 || true
    exit 1
}

oracle_primary() {
    case "$1" in
        svt) echo SvtAv1EncApp ;;
        avm) echo avmenc ;;
        dav2d) echo dav2d ;;
        lc3) echo elc3 ;;
        *) echo "$1" ;;
    esac
}

oracle_binaries() {
    case "$1" in
        svt) echo "SvtAv1EncApp" ;;
        avm) echo "avmenc" ;;
        dav2d) echo "dav2d" ;;
        lc3) echo "elc3 dlc3" ;;
        *) echo "$1" ;;
    esac
}

build_oracle svt build_svt
build_oracle avm build_avm
build_oracle dav2d build_dav2d
build_oracle lc3 build_lc3

if ! offline; then
    # System packages: ffmpeg and flac come from the distribution image.
    if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v flac >/dev/null 2>&1; then
        echo "ffmpeg and flac are required (sudo apt-get install -y ffmpeg flac)" >&2
        exit 1
    fi
fi

for name in "${ORACLES[@]}"; do
    if [[ -z "${selected[${name}]:-}" || -n "${skipped_names[${name}]:-}" ]]; then
        continue
    fi
    missing=""
    read -r -a tools <<<"$(oracle_binaries "${name}")"
    for tool in "${tools[@]}"; do
        if [[ ! -x "${bin}/${tool}" ]]; then
            missing="${tool}"
            break
        fi
    done
    if [[ -n "${missing}" ]]; then
        if [[ ${best_effort} -eq 1 ]]; then
            skip_oracle "${name}" "binary ${missing} not installed"
            continue
        fi
        echo "required oracle missing: ${bin}/${missing}" >&2
        echo "run without BENCH_OFFLINE, or install the oracle binaries" >&2
        exit 1
    fi
done

version_line() {
    local tool="$1"
    shift
    if command -v "${tool}" >/dev/null 2>&1; then
        "${tool}" "$@" 2>&1 | head -n 1 || true
    else
        printf 'missing'
    fi
}

json_string() {
    printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

svt_version="$(version_line SvtAv1EncApp --version)"
dav2d_version="$(version_line dav2d --version)"
ffmpeg_version="$(version_line ffmpeg -version)"
flac_version="$(version_line flac --version)"

skipped_json="[]"
if [[ ${#skipped[@]} -gt 0 ]]; then
    parts=()
    for entry in "${skipped[@]}"; do
        name="${entry%%|*}"
        reason="${entry#*|}"
        parts+=("{\"oracle\": $(json_string "${name}"), \"reason\": $(json_string "${reason}")}")
    done
    skipped_json="[$(IFS=,; echo "${parts[*]}")]"
fi

cat >"${dest}/oracles.json" <<EOF
{
  "svt_av1_pin": "${ORACLE_SVT_AV1_REF}",
  "svt_av1_commit": "${ORACLE_SVT_AV1_COMMIT}",
  "svt_av1_version": $(json_string "${svt_version}"),
  "avm_pin": "${ORACLE_AVM_REF}",
  "avm_commit": "${ORACLE_AVM_COMMIT}",
  "avm_version": "AVM ${ORACLE_AVM_REF}",
  "dav2d_commit": "${ORACLE_DAV2D_REF}",
  "dav2d_version": $(json_string "${dav2d_version}"),
  "liblc3_pin": "${ORACLE_LIBC3_REF}",
  "liblc3_commit": "${ORACLE_LIBC3_COMMIT}",
  "liblc3_version": "liblc3 ${ORACLE_LIBC3_REF}",
  "ffmpeg": $(json_string "${ffmpeg_version}"),
  "flac": $(json_string "${flac_version}"),
  "skipped": ${skipped_json}
}
EOF

log "oracle directory: ${dest}"
log "version manifest: ${dest}/oracles.json"
if [[ ${#skipped[@]} -gt 0 ]]; then
    log "skipped oracles: $(printf '%s ' "${skipped[@]%%|*}")"
fi
