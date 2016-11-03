#!/usr/bin/env python
"""
A small wrapper around pytest.

Sets up logging defaults.
"""

import pytest
import sys

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    sys.exit(pytest.main())
