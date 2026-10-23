#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/project-osrm/osrm-backend@sha256:43bbc1ff0d19b397406c94c8b79efca33dccdb77f8747e33d8cf25d70f7a88fe}"
ACTION="${1:-}"; CITY="$(echo "${2:-}" | tr '[:upper:]' '[:lower:]')"; MODE="$(echo "${3:-car}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$ACTION" || -z "$CITY" ]]; then
  echo "usage: bash osrm/manage_osrm.sh <build|start|stop|test> <nyc|tky|ca> [car|foot]"; exit 1
fi
OSRM_ROOT="${OSRM_ROOT:-$HOME/osrm-data}"; PORT="${PORT:-5000}"; DOCKER="${DOCKER:-docker}"; OSMIUM="${OSMIUM:-osmium}"
GEOFABRIK="https://download.geofabrik.de"

case "$CITY" in
  nyc) PBFS=("north-america/us/new-york-latest.osm.pbf"); BASE="new-york-latest"; TEST="-73.9857,40.7484;-73.9772,40.7527" ;;
  tky) PBFS=("asia/japan/kanto-latest.osm.pbf"); BASE="kanto-latest"; TEST="139.7672,35.6697;139.8051,35.7496" ;;
  ca)  PBFS=("north-america/us/california-latest.osm.pbf" "north-america/us/nevada-latest.osm.pbf"); BASE="ca_nv"; TEST="-122.3862,37.6164;-122.4076,37.7826" ;;
  *) echo "unknown city: $CITY (nyc | tky | ca)"; exit 1 ;;
esac
case "$MODE" in
  car)  LUA="/opt/car.lua";  PROFILE="driving" ;;
  foot) LUA="/opt/foot.lua"; PROFILE="foot" ;;
  *) echo "unknown mode: $MODE (car | foot)"; exit 1 ;;
esac
DIR="$OSRM_ROOT/${CITY}_${MODE}"; NAME="osrm-${CITY}-${MODE}"

build() {
  mkdir -p "$DIR"; cd "$DIR"
  for p in "${PBFS[@]}"; do wget -c -q --show-progress "$GEOFABRIK/$p" -O "$(basename "$p")"; done
  if [[ "$CITY" == ca ]]; then
    "$OSMIUM" merge california-latest.osm.pbf nevada-latest.osm.pbf -o ca_nv.osm.pbf --overwrite
  fi
  $DOCKER run --rm -t -v "$DIR:/data" "$IMAGE" osrm-extract -p "$LUA" "/data/$BASE.osm.pbf"
  $DOCKER run --rm -t -v "$DIR:/data" "$IMAGE" osrm-contract "/data/$BASE.osrm"
  echo "built $DIR/$BASE.osrm"
}

start() {
  [[ -f "$DIR/$BASE.osrm.hsgr" ]] || { echo "no graph in $DIR; run: bash osrm/manage_osrm.sh build $CITY $MODE"; exit 1; }
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  $DOCKER run -d --name "$NAME" -p "$PORT:5000" -v "$DIR:/data" "$IMAGE" \
    osrm-routed --algorithm ch --max-table-size 100000 "/data/$BASE.osrm"
  for _ in $(seq 1 60); do
    curl -s "http://localhost:$PORT/route/v1/$PROFILE/$TEST" | grep -q '"code":"Ok"' && { echo "$NAME ready at http://localhost:$PORT (profile $PROFILE)"; return; }
    sleep 5
  done
  echo "server did not answer within 5 min; check: $DOCKER logs $NAME"; exit 1
}

case "$ACTION" in
  build) build ;;
  start) start ;;
  stop)  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true; echo "stopped $NAME" ;;
  test)  curl -s "http://localhost:$PORT/route/v1/$PROFILE/$TEST?overview=false"; echo ;;
  *) echo "unknown action: $ACTION (build | start | stop | test)"; exit 1 ;;
esac
