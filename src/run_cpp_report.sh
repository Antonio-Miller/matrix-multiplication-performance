#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
mkdir -p "${RESULTS_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RAW_OUT="${RESULTS_DIR}/cpp_report_${TS}.txt"
CSV_OUT="${RESULTS_DIR}/cpp_report_${TS}.csv"
PERF_OUT="${RESULTS_DIR}/cpp_perf_${TS}.txt"

SMALL_SIZES=(1024 1536 2048 2560 3072)
LARGE_SIZES=(4096 6144 8192 10240)
BLOCK_SIZES=(128 256 512)
THREADS_SCALING=(1 4 8 12 16 20 24)

MODE="${1:-full}"

print_usage() {
        cat <<EOF
Usage: ./run_cpp_report.sh [mode]

Modes:
    full       Run all benchmarks and perf collection (default)
    perf-only  Compile and run only perf collection for Part 1.4
    no-perf    Run all benchmarks but skip perf collection
EOF
}

if [[ "${MODE}" != "full" && "${MODE}" != "perf-only" && "${MODE}" != "no-perf" ]]; then
        print_usage
        exit 1
fi

perf_allowed() {
    set +e
    perf stat true >/dev/null 2>&1
    local status=$?
    set -e
    [[ "${status}" -eq 0 ]]
}

log() {
    echo "$*" | tee -a "${RAW_OUT}"
}

append_csv_from_output() {
    local output="$1"
    echo "${output}" | sed -n 's/^CSV[[:space:]]*:[[:space:]]*//p' >> "${CSV_OUT}"
}

run_case() {
    local op="$1"
    local n="$2"
    local extra="${3:-}"

    log ""
    log "----------------------------------------------------------------"
    log "RUN CASE | op=${op} | size=${n}${extra:+ | extra=${extra}}"
    log "----------------------------------------------------------------"

    local output
    if [[ -n "${extra}" ]]; then
        output="$(./multCPP <<EOF
${op}
${n}
${extra}
0
EOF
)"
    else
        output="$(./multCPP <<EOF
${op}
${n}
0
EOF
)"
    fi

    echo "${output}" >> "${RAW_OUT}"
    append_csv_from_output "${output}"
}

run_perf_case() {
    local label="$1"
    local op="$2"
    local n="$3"
    local extra="${4:-}"

    {
        echo ""
        echo "================================================================"
        echo "PERF CASE | ${label} | op=${op} | size=${n}${extra:+ | extra=${extra}}"
        echo "================================================================"
        echo ""
    } >> "${PERF_OUT}"

    local status=0
    if [[ -n "${extra}" ]]; then
        set +e
        perf stat ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
${extra}
0
EOF
        status=$?
        set -e
    else
        set +e
        perf stat ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
0
EOF
        status=$?
        set -e
    fi

    if [[ "${status}" -ne 0 ]]; then
        echo "perf stat failed with exit code ${status}." >> "${PERF_OUT}"
    fi
}

run_perf_cache_events_case() {
    local label="$1"
    local op="$2"
    local n="$3"
    local extra="${4:-}"

    {
        echo ""
        echo "================================================================"
        echo "PERF CACHE EVENTS | ${label} | op=${op} | size=${n}${extra:+ | extra=${extra}}"
        echo "================================================================"
        echo ""
    } >> "${PERF_OUT}"

    local events_primary="mem_load_retired.l1_miss,mem_load_retired.l2_miss"
    local events_fallback="cache-misses,cache-references"

    if [[ -n "${extra}" ]]; then
        if ! perf stat -e "${events_primary}" ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
${extra}
0
EOF
        then
            echo "Primary perf events unavailable; using fallback (${events_fallback})." >> "${PERF_OUT}"
            perf stat -e "${events_fallback}" ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
${extra}
0
EOF
        fi
    else
        if ! perf stat -e "${events_primary}" ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
0
EOF
        then
            echo "Primary perf events unavailable; using fallback (${events_fallback})." >> "${PERF_OUT}"
            perf stat -e "${events_fallback}" ./multCPP <<EOF >> "${PERF_OUT}" 2>&1
${op}
${n}
0
EOF
        fi
    fi
}

# Start fresh output files.
: > "${RAW_OUT}"
: > "${PERF_OUT}"
cat > "${CSV_OUT}" <<EOF
algorithm,size,threads,block_size,time_s,gflops
EOF

log "CPD Project 1 - C++ Automated Runner"
log "Timestamp      : ${TS}"
log "Mode           : ${MODE}"
log "Source folder  : ${SCRIPT_DIR}"
log "Raw output     : ${RAW_OUT}"
log "CSV output     : ${CSV_OUT}"
log "Perf output    : ${PERF_OUT}"

log ""
log "[1/6] Compiling C++ source (OpenMP)..."
(
    cd "${SCRIPT_DIR}"
    g++ -O2 -fopenmp -o multCPP mult.cpp
)
log "Compilation done."

if [[ "${MODE}" == "full" || "${MODE}" == "no-perf" ]]; then
    log ""
    log "[2/6] Part 1.1 and 1.2 (small sizes, C++):"
    for n in "${SMALL_SIZES[@]}"; do
        run_case 1 "${n}"
        run_case 2 "${n}"
    done

    log ""
    log "[3/6] Part 1.2 (large sizes, C++ line/ikj):"
    for n in "${LARGE_SIZES[@]}"; do
        run_case 2 "${n}"
    done

    log ""
    log "[4/6] Part 1.3 Block (sizes x block sizes):"
    for n in "${LARGE_SIZES[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            run_case 3 "${n}" "${b}"
        done
    done

    log ""
    log "[5/6] Part 2.1 (4 threads, sizes 1024..3072):"
    for n in "${SMALL_SIZES[@]}"; do
        run_case 4 "${n}" "4"
        run_case 5 "${n}" "4"
        run_case 6 "${n}" "4"
    done

    log ""
    log "[6/6] Part 2.2 (V2 scaling, size 8192):"
    for t in "${THREADS_SCALING[@]}"; do
        run_case 6 8192 "${t}"
    done
fi

if [[ "${MODE}" != "no-perf" ]]; then
    log ""
    log "Collecting perf data for Part 1.4 (representative size 4096)..."
    if perf_allowed; then
        run_perf_case "ijk" 1 4096
        run_perf_case "ikj" 2 4096
        run_perf_case "block-256" 3 4096 256
        run_perf_cache_events_case "ijk" 1 4096
        run_perf_cache_events_case "ikj" 2 4096
        run_perf_cache_events_case "block-256" 3 4096 256
    else
        log "Perf collection skipped: insufficient permissions (likely perf_event_paranoid)."
        {
            echo ""
            echo "Perf collection skipped: insufficient permissions."
            echo "Hint: check /proc/sys/kernel/perf_event_paranoid and CAP_PERFMON/CAP_SYS_ADMIN."
        } >> "${PERF_OUT}"
    fi
fi

{
    echo ""
    echo "NOTE: Report section 2.3 (for simd / collapse(2)) is not collected by this script"
    echo "because those variants are not implemented as separate menu options in mult.cpp yet."
} >> "${RAW_OUT}"

log ""
log "Done. Generated files:"
log "- ${RAW_OUT}"
log "- ${CSV_OUT}"
log "- ${PERF_OUT}"

printf "\nArtifacts created:\n%s\n%s\n%s\n" "${RAW_OUT}" "${CSV_OUT}" "${PERF_OUT}"
