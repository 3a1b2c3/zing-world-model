set -eu
ZING_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ZING_PYTHON=${ZING_PYTHON:-python3}
export PYTHONPATH="$ZING_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
exec "$ZING_PYTHON" -m zing_v0_5 "$@"
