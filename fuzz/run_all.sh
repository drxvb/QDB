#!/usr/bin/env sh
# Run every fuzz target in parallel for the same time, one libFuzzer process each.
#
# usage: fuzz/run_all.sh <seconds> <state-dir>
#   e.g. TMPDIR=/dev/shm sh fuzz/run_all.sh 900 /var/tmp/quickdb-fuzz
# FUZZ_TARGETS="store_api store_open" limits the run to some targets.
# Needs a nightly toolchain and cargo-fuzz. Corpora go to <state-dir>/corpus/<target>, crash,
# timeout and out-of-memory inputs to <state-dir>/artifacts/<target>/, logs to <state-dir>/log/.
# Exits 0 only if every target ran for the whole time without a finding.
set -u
SECONDS_PER_TARGET="${1:?seconds}"
STATE="${2:?state dir}"
cd "$(dirname "$0")"
TARGETS="${FUZZ_TARGETS:-snapshot_decode snapshot_mutate store_open store_api}"
cargo fuzz build -O -a || exit 2
mkdir -p "$STATE/log"
for t in $TARGETS; do
  mkdir -p "$STATE/corpus/$t" "$STATE/artifacts/$t"
done
# Seed the snapshot corpora with the valid fixtures (snapshot_decode without the checksum).
for f in ../crates/quickdb-native/tests/fixtures/snap/*.bin; do
  n=$(basename "$f" .bin)
  size=$(wc -c < "$f")
  head -c $((size - 4)) "$f" > "$STATE/corpus/snapshot_decode/seed-$n"
  printf '\003' > "$STATE/corpus/store_open/seed-mode3-$n"
  printf '\002' > "$STATE/corpus/store_open/seed-mode2-$n"
  tail -c +41 "$f" | head -c $((size - 44)) >> "$STATE/corpus/store_open/seed-mode2-$n"
done
printf '\000' > "$STATE/corpus/snapshot_mutate/seed-none"
PIDS=""
for t in $TARGETS; do
  cargo fuzz run -O -a "$t" "$STATE/corpus/$t" -- \
    -max_total_time="$SECONDS_PER_TARGET" -timeout=20 -rss_limit_mb=2048 -malloc_limit_mb=2048 \
    -artifact_prefix="$STATE/artifacts/$t/" -print_final_stats=1 \
    > "$STATE/log/$t.log" 2>&1 &
  PIDS="$PIDS $!"
done
FAILED=0
for job in $PIDS; do
  wait "$job" || FAILED=$((FAILED + 1))
done
for t in $TARGETS; do
  echo "== $t"
  grep -E "^Done [0-9]+ runs|stat::number_of_executed_units|stat::peak_rss_mb|SUMMARY|panicked at" "$STATE/log/$t.log" | head -5
  ls "$STATE/artifacts/$t"
done
# cargo-fuzz also creates empty default corpus and artifact directories here; remove them.
find corpus artifacts -depth -type d -empty -delete 2>/dev/null
echo "targets with a finding: $FAILED"
[ "$FAILED" = 0 ]
