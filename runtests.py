#!/usr/bin/env python
"""
A small wrapper around pytest.

Sets up logging defaults.
"""

import pytest

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    pytest.main()
