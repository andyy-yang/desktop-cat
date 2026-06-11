"""py2app entry point for OverlayCat (see packaging/setup_app.py)."""

import sys

sys.dont_write_bytecode = True

from overlay.app.main import main

main()
