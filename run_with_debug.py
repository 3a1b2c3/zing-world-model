#!/usr/bin/env python
import sys
import traceback

try:
    from zing_v0_5.main import main
    main()
except Exception as e:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
