#!/usr/bin/env python3

import qmmd
from qmmd.prep import require_amber

try:
    tleap_path = require_amber()
    print(f"It worked — tleap found at {tleap_path}")

except Exception as e:
    print("Amber is not loaded.")
    print("Actual error:")
    print(e)
