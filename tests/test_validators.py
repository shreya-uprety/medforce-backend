"""
Tests for data validators: NHS number checksum, DOB range, email format.
"""

import pytest

from medforce.gateway.validators import validate_dob, validate_email, validate_nhs_number


class TestNHSNumberValidation:
    """NHS number uses Modulus 11 checksum algorithm."""

    def test_valid_nhs_number(self):
        # 943 476 5870 is a well-known valid test number
        assert validate_nhs_number("9434765870") is True

    def test_valid_nhs_number_with_spaces(self):
        assert validate_nhs_number("943 476 5870") is True

    def test_invalid_checksum(self):
        # Change last digit to make checksum fail
        assert validate_nhs_number("9434765871") is False

    def test_too_short(self):
        assert validate_nhs_number("12345") is False

    def test_too_long(self):
        assert validate_nhs_number("12345678901") is False

    def test_non_digits(self):
        assert validate_nhs_number("943476587A") is False

    def test_empty_string(self):
        assert validate_nhs_number("") is False

    def test_all_zeros(self):
        # 0000000000 — checksum: sum=0, remainder=0, check=0 → valid by algo
        assert validate_nhs_number("0000000000") is True

    def test_remainder_one_invalid(self):
        # A number where modulus 11 gives remainder 1 is always invalid
        # We need to find such a number. Let's compute:
        # digits d1..d9 with weights 10..2
        # If sum mod 11 == 1, no valid check digit exists
        # 1000000001: sum = 1*10 + 0*9 + ... + 0*2 = 10, 10 mod 11 = 10, check = 1
        # That's actually valid (check digit 1). Let's try another approach:
        # We just verify that our function catches known invalid numbers
        assert validate_nhs_number("1234567890") is False


class TestDOBValidation:
    def test_valid_dob_dd_mm_yyyy(self):
        assert validate_dob("15/03/1985") is True

    def test_valid_dob_dd_mm_yyyy_dash(self):
        assert validate_dob("15-03-1985") is True

    def test_valid_dob_iso(self):
        assert validate_dob("1985-03-15") is True

    def test_future_date_invalid(self):
        assert validate_dob("01/01/2099") is False

    def test_too_old(self):
        assert validate_dob("01/01/1800") is False

    def test_invalid_format(self):
        assert validate_dob("not-a-date") is False

    def test_empty_string(self):
        assert validate_dob("") is False

    def test_recent_date_valid(self):
        assert validate_dob("01/01/2020") is True

    def test_boundary_120_years(self):
        # A date 121 years ago should be invalid
        assert validate_dob("01/01/1900") is False


class TestEmailValidation:
    def test_valid_email(self):
        assert validate_email("john@example.com") is True

    def test_valid_email_with_plus(self):
        assert validate_email("john+tag@example.co.uk") is True

    def test_valid_email_with_dots(self):
        assert validate_email("first.last@company.org") is True

    def test_missing_at(self):
        assert validate_email("john.example.com") is False

    def test_missing_tld(self):
        assert validate_email("john@example") is False

    def test_tld_too_short(self):
        assert validate_email("john@example.c") is False

    def test_empty_string(self):
        assert validate_email("") is False

    def test_spaces_stripped(self):
        assert validate_email(" john@example.com ") is True

    def test_valid_nhs_email(self):
        assert validate_email("dr.patel@nhs.net") is True
