from datetime import date

import pytest

from pipeline import holidays


@pytest.mark.parametrize(
    "d,expected",
    [
        (date(2025, 1, 26), False),  # Republic Day (Sunday anyway, but holiday)
        (date(2025, 8, 15), False),  # Independence Day
        (date(2025, 10, 2), False),  # Gandhi Jayanti
        (date(2025, 1, 4), False),   # Saturday
        (date(2025, 1, 5), False),   # Sunday
        (date(2025, 1, 6), True),    # Monday, regular trading day
    ],
)
def test_is_trading_day(d, expected):
    assert holidays.is_trading_day(d, "NSE") is expected


def test_previous_trading_day_skips_weekend():
    # Monday 2025-01-06 → previous trading day should be Friday 2025-01-03
    assert holidays.previous_trading_day(date(2025, 1, 6), "NSE") == date(2025, 1, 3)


def test_next_trading_day_skips_weekend():
    # Friday 2025-01-03 → next trading day should be Monday 2025-01-06
    assert holidays.next_trading_day(date(2025, 1, 3), "NSE") == date(2025, 1, 6)


def test_trading_days_between_excludes_weekends_and_holidays():
    # Aug 11 to 17 2025: Aug 15 = Independence Day (Fri), Aug 16-17 = weekend
    days = holidays.trading_days_between(date(2025, 8, 11), date(2025, 8, 17), "NSE")
    assert date(2025, 8, 15) not in days  # Independence Day
    assert date(2025, 8, 16) not in days  # Saturday
    assert date(2025, 8, 17) not in days  # Sunday
    assert date(2025, 8, 11) in days
    assert date(2025, 8, 12) in days
    assert date(2025, 8, 13) in days
    assert date(2025, 8, 14) in days


def test_bse_same_calendar_as_nse():
    # NSE and BSE share trading days
    d = date(2025, 1, 6)
    assert holidays.is_trading_day(d, "NSE") == holidays.is_trading_day(d, "BSE")


@pytest.mark.parametrize(
    "d",
    [
        date(2010, 2, 6),    # Saturday
        date(2010, 5, 16),   # Sunday, gold ETFs only
        date(2013, 11, 14),  # the XBOM calendar had this closed
        date(2024, 1, 20),   # special live session, Saturday
        date(2024, 11, 1),   # Muhurat, Friday
        date(2025, 2, 1),    # budget, Saturday
        date(2025, 10, 21),  # Muhurat, Tuesday
        date(2026, 2, 1),    # budget, Sunday
    ],
)
def test_special_sessions_are_trading_days(d):
    assert holidays.is_trading_day(d, "NSE")
    assert d in holidays.trading_days_between(d, d, "NSE")


def test_closed_day_the_old_calendar_had_open():
    assert not holidays.is_trading_day(date(2013, 11, 15), "NSE")


def test_year_without_a_list_counts_weekdays():
    assert holidays.is_listed(date(2026, 12, 31))
    assert not holidays.is_listed(date(2027, 1, 1))
    assert holidays.is_trading_day(date(2027, 1, 1), "NSE")  # Friday
    assert not holidays.is_trading_day(date(2027, 1, 2), "NSE")  # Saturday
    assert holidays.next_trading_day(date(2026, 12, 31), "NSE") == date(2027, 1, 1)
    assert holidays.trading_days_between(date(2026, 12, 31), date(2027, 1, 4), "NSE") == [
        date(2026, 12, 31), date(2027, 1, 1), date(2027, 1, 4),
    ]


def test_weekday_after_special_sunday_previous_is_the_special_session():
    assert holidays.previous_trading_day(date(2026, 2, 2), "NSE") == date(2026, 2, 1)
