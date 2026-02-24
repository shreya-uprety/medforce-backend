"""
Data validators for patient information.

NHS number checksum, DOB range, email format.
"""

from __future__ import annotations

import re
from datetime import date, datetime


def validate_nhs_number(nhs: str) -> bool:
    """
    Validate an NHS number using the Modulus 11 checksum algorithm.

    NHS numbers are 10 digits. The last digit is a check digit.
    Multiply digits 1-9 by weights 10-2, sum them, take mod 11.
    If remainder is 0, check digit is 0. If remainder is 1, number is invalid.
    Otherwise check digit = 11 - remainder.
    """
    digits = nhs.replace(" ", "").replace("-", "")
    if not digits.isdigit() or len(digits) != 10:
        return False

    weights = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    remainder = total % 11
    if remainder == 0:
        expected_check = 0
    elif remainder == 1:
        return False  # invalid — no valid check digit
    else:
        expected_check = 11 - remainder

    return int(digits[9]) == expected_check


def validate_dob(dob_str: str) -> bool:
    """
    Validate a date of birth string.

    Must be a parseable date, in the past, and within 120 years of today.
    Accepts DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD formats.
    """
    parsed: date | None = None
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(dob_str.strip(), fmt).date()
            break
        except ValueError:
            continue

    if parsed is None:
        return False

    today = date.today()
    if parsed >= today:
        return False

    age_years = (today - parsed).days / 365.25
    if age_years > 120:
        return False

    return True


def validate_email(email: str) -> bool:
    """
    Validate an email address format.

    Checks basic structure and TLD length >= 2.
    """
    pattern = r"^[\w.+-]+@[\w-]+\.[\w.]+$"
    if not re.match(pattern, email.strip()):
        return False

    # TLD must be at least 2 characters
    tld = email.strip().rsplit(".", 1)[-1]
    if len(tld) < 2:
        return False

    return True
