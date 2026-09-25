#!/usr/bin/env python3
"""Build the BV meta-hic OA-versus-OD joint summary."""
from __future__ import annotations

import sys

from joint_summary_b_oa_od import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--variant", "hic"]
    main()
