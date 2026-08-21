#!/usr/bin/env bash
#
# Send N prediction requests at the gateway and tally which model version
# answered each one. This is the evidence for a canary split: the weights are a
# claim, and this is the measurement.
#
#   charts/load_test.sh                          # 200 requests at localhost:30080
#   charts/load_test.sh 500                      # 500 of them
#   charts/load_test.sh 20 localhost:30080 green # pin every request to green
#
# The inherited version of this file sent a bare GET with an `x-api-version`
# header and slept 200ms between requests, which took 40 seconds to produce 200
# lines of untallied output - against a chart whose Service pointed at a port
# nothing listened on. It now posts a real record to /predict, reads the
# X-Model-Version header the router stamps on every response, and counts.
set -euo pipefail

REQUESTS="${1:-200}"
TARGET="${2:-localhost:30080}"
PIN="${3:-}"

# One row of the automobile dataset, exactly as the data has it - `horsepower`
# as a string included. The model carries its own preprocessing, so nothing here
# has to know how it was trained.
read -r -d '' PAYLOAD <<'JSON' || true
{"records":[{"cylinders":8,"displacement":307.0,"horsepower":"130.0",
"weight":3504,"acceleration":12.0,"model year":70,"origin":1,
"car name":"chevrolet chevelle malibu"}]}
JSON

pin_header=()
if [ -n "$PIN" ]; then
  pin_header=(--header "x-api-version: ${PIN}")
fi

echo "POST http://${TARGET}/predict x${REQUESTS}${PIN:+ (pinned to ${PIN})}"

tally=$(
  for ((i = 1; i <= REQUESTS; i++)); do
    # -D - dumps the response headers to stdout; X-Model-Version is the one the
    # router stamps with the name of the version that answered.
    curl -s -o /dev/null -D - \
      --header 'Content-Type: application/json' \
      "${pin_header[@]+"${pin_header[@]}"}" \
      --data "$PAYLOAD" \
      "http://${TARGET}/predict" |
      tr -d '\r' |
      awk 'tolower($1) == "x-model-version:" { print $2 }'
  done | sort | uniq -c | sort -rn
)

echo "$tally"

total=$(echo "$tally" | awk '{ sum += $1 } END { print sum + 0 }')
if [ "$total" -gt 0 ]; then
  echo "$tally" | awk -v total="$total" '{ printf "%-8s %5d  %5.1f%%\n", $2, $1, 100 * $1 / total }'
fi
