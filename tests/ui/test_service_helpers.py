"""Unit tests for service pagination/scan helpers (clamping + honesty)."""

from __future__ import annotations

from omega.ui.service import MAX_PAGE_SIZE, _paginate


def test_paginate_clamps_low_extremes():
    _, p = _paginate(list(range(5)), page=-5, page_size=-10)
    assert p.page == 1
    assert p.page_size == 1


def test_paginate_clamps_high_extremes():
    _, p = _paginate(list(range(5)), page=10**9, page_size=10**9)
    assert p.page == p.total_pages
    assert p.page_size == MAX_PAGE_SIZE


def test_paginate_empty_is_single_page():
    window, p = _paginate([], page=1, page_size=25)
    assert window == []
    assert p.total == 0
    assert p.total_pages == 1
    assert p.has_next is False
    assert p.has_prev is False


def test_paginate_scan_capped_passthrough():
    _, capped = _paginate(list(range(3)), page=1, page_size=2, scan_capped=True)
    assert capped.scan_capped is True
    _, uncapped = _paginate(list(range(3)), page=1, page_size=2)
    assert uncapped.scan_capped is False
