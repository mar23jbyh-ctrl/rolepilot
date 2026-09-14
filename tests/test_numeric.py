"""M4 的验证用例：统一数值入口（app/utils/numeric.py）。

覆盖方案 §6.1 B0 要求的六类输入：
非数字字符串 / None / NaN·inf / 布尔 / 越界 / 数字字符串。
"""

from __future__ import annotations

import math

from app.utils.numeric import (
    STATUS_CLAMPED_HIGH,
    STATUS_CLAMPED_LOW,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_OK,
    clamp,
    failed_keys,
    parse_number,
    safe_dimensions,
    safe_score,
)


# ---- 第 1 类：非数字字符串 ----

def test_invalid_string_returns_none_and_marks():
    result = parse_number("比较一般")
    assert result.value is None
    assert result.status == STATUS_INVALID
    assert result.is_failure is True
    # 关键契约：不提供"解析失败给中间分"的默认行为
    assert safe_score("比较一般") is None


def test_invalid_structured_value_rejected():
    assert parse_number({"value": 7}).status == STATUS_INVALID
    assert parse_number([7]).status == STATUS_INVALID


# ---- 第 2 类：None / 空值占位符 ----

def test_missing_values_are_distinguished_from_invalid():
    for raw in (None, "", "  ", "null", "None", "N/A", "nan", "无", "未知"):
        result = parse_number(raw)
        assert result.value is None, raw
        assert result.status == STATUS_MISSING, raw
    # missing 与 invalid 必须可区分：一个可以回退默认值，一个不可以
    assert parse_number(None).status != parse_number("abc").status


def test_safe_score_default_only_fills_missing_or_invalid():
    assert safe_score(None, default=5.0) == 5.0
    assert safe_score("abc", default=5.0) == 5.0
    assert safe_score(8, default=5.0) == 8.0
    assert safe_score("", default=None) is None


# ---- 第 3 类：NaN / inf ----

def test_non_finite_rejected():
    for raw in (float("nan"), float("inf"), float("-inf")):
        result = parse_number(raw)
        assert result.value is None
        assert result.status == STATUS_INVALID
    assert parse_number("inf").status == STATUS_INVALID


# ---- 第 4 类：布尔 ----

def test_bool_rejected_to_avoid_silent_float_conversion():
    # float(True) == 1.0 会把类型错误伪装成合法分数，必须拦掉
    for raw in (True, False):
        result = parse_number(raw)
        assert result.value is None
        assert result.status == STATUS_INVALID


# ---- 第 5 类：越界 ----

def test_out_of_range_is_clamped_and_flagged():
    low = parse_number(-3)
    assert low.value == 0.0
    assert low.status == STATUS_CLAMPED_LOW
    assert low.usable is True
    assert low.is_failure is False

    high = parse_number(85)
    assert high.value == 10.0
    assert high.status == STATUS_CLAMPED_HIGH


def test_custom_bounds():
    assert parse_number(1.5, low=1.0, high=1.2).status == STATUS_CLAMPED_HIGH
    assert parse_number(1.1, low=1.0, high=1.2).status == STATUS_OK
    assert clamp(5, 0, 1) == 1


# ---- 第 6 类：数字字符串 ----

def test_numeric_strings_accepted():
    assert parse_number("8").value == 8.0
    assert parse_number(" 7.5 ").value == 7.5
    assert parse_number("8").status == STATUS_OK
    # 带单位的字符串仍视为非法，不做模糊提取
    assert parse_number("8分").status == STATUS_INVALID


# ---- 合法输入 ----

def test_plain_numbers_pass_through():
    assert parse_number(0) == (0.0, STATUS_OK)
    assert parse_number(10) == (10.0, STATUS_OK)
    assert parse_number(6.5) == (6.5, STATUS_OK)
    assert math.isclose(safe_score(6.25), 6.25)


# ---- dimensions ----

def test_dimensions_omit_invalid_keys_instead_of_zeroing():
    result = safe_dimensions(
        {"d1": 7, "d2": "比较一般", "d3": None, "d4": 12},
        allowed_keys=["d1", "d2", "d3", "d4", "d5"],
    )
    # 非法/缺失的键不能出现在 values 里，否则下游会把"没评"当成 0 分
    assert result.values == {"d1": 7.0, "d4": 10.0}
    assert result.statuses["d2"] == STATUS_INVALID
    assert result.statuses["d3"] == STATUS_MISSING
    assert result.statuses["d4"] == STATUS_CLAMPED_HIGH
    # rubric 要求但模型没给的维度同样记为 missing
    assert result.statuses["d5"] == STATUS_MISSING
    assert result.invalid_keys == ["d2"]


def test_dimensions_handles_non_dict():
    missing = safe_dimensions(None)
    assert missing.values == {}
    assert missing.statuses == {"__all__": STATUS_MISSING}

    invalid = safe_dimensions("d1=7")
    assert invalid.values == {}
    assert invalid.statuses == {"__all__": STATUS_INVALID}


def test_dimensions_without_allowed_keys_keeps_all_valid():
    result = safe_dimensions({"a": 1, "b": 2.5})
    assert result.values == {"a": 1.0, "b": 2.5}
    assert result.invalid_keys == []


def test_failed_keys_reports_sorted_failures():
    statuses = {"d1": STATUS_OK, "d2": STATUS_INVALID, "d3": STATUS_MISSING}
    assert failed_keys(statuses) == ["d2", "d3"]
    assert failed_keys({"d1": STATUS_OK, "d2": STATUS_CLAMPED_HIGH}) == []
