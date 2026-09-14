"""统一的数值解析入口（M4）。

集中处理数值解析、合法区间和解析状态，供评分聚合等程序逻辑使用。

本模块的三条规则：

1. **非数字不给假值**：无法解析时返回 ``None`` 并带上状态码，
   让调用方显式决定"重试 / 跳过 / 记异常"，不补造中间分数。
2. **越界要留痕**：超出合法区间时 clamp 到边界，并把状态标成
   ``clamped_low`` / ``clamped_high``，便于统计"模型输出越界的频率"。
3. **缺失与非法要能分开**：``missing`` 表示模型没给这个字段
   （可回退默认值），``invalid`` 表示给了但解析不了（不该回退默认值）。

本模块是纯函数、无副作用，不调用任何 LLM。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, NamedTuple

# 单题分与各维度分的合法区间（与 app/models/schemas.py 的 Assessment 约束一致）
SCORE_MIN = 0.0
SCORE_MAX = 10.0

# ---- 解析状态码 ----
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_INVALID = "invalid"
STATUS_CLAMPED_LOW = "clamped_low"
STATUS_CLAMPED_HIGH = "clamped_high"

ALL_STATUSES = (
    STATUS_OK,
    STATUS_MISSING,
    STATUS_INVALID,
    STATUS_CLAMPED_LOW,
    STATUS_CLAMPED_HIGH,
)

# 这些字面量在语义上等价于"模型没给值"，与真正的非法值区分开
_MISSING_TOKENS = frozenset({"", "null", "none", "nan", "n/a", "na", "-", "无", "未知"})


class NumericResult(NamedTuple):
    """单个数值的解析结果。

    ``value`` 为 ``None`` 时表示"拿不到可用数值"，此时 ``status`` 一定是
    ``missing`` 或 ``invalid``；``clamped_*`` 表示值可用但已被夹到边界。
    """

    value: float | None
    status: str

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def usable(self) -> bool:
        """是否拿到了可用数值（含被 clamp 的情况）。"""

        return self.value is not None

    @property
    def is_failure(self) -> bool:
        """是否属于"必须上报"的失败（缺失或非法）。"""

        return self.status in {STATUS_MISSING, STATUS_INVALID}


class DimensionsResult(NamedTuple):
    """一组维度分的解析结果。

    - ``values``：可用的维度分（非法/缺失的键**不会**出现在这里，
      以避免下游把"没评"错当成"评了 0 分"）
    - ``statuses``：每个出现过的键的状态码
    - ``invalid_keys``：给了值但解析不了的键，供上层写进异常报告
    """

    values: dict[str, float]
    statuses: dict[str, str]
    invalid_keys: list[str]


def clamp(value: float, low: float = SCORE_MIN, high: float = SCORE_MAX) -> float:
    """把数值夹到 ``[low, high]``。"""

    if low > high:
        low, high = high, low
    return min(high, max(low, value))


def parse_number(
    raw: Any,
    *,
    low: float = SCORE_MIN,
    high: float = SCORE_MAX,
) -> NumericResult:
    """把任意输入解析成合法区间内的数值，并返回状态码。

    支持的输入：``int`` / ``float`` / 数字字符串。
    明确拒绝：``bool``（``float(True) == 1.0`` 会掩盖类型错误）、
    ``None``、空串、``"null"`` 这类占位符、``NaN``/``inf``、
    以及 dict / list 等结构。
    """

    if raw is None:
        return NumericResult(None, STATUS_MISSING)

    # bool 是 int 的子类，必须单独拦掉：True 会被 float() 静默变成 1.0
    if isinstance(raw, bool):
        return NumericResult(None, STATUS_INVALID)

    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() in _MISSING_TOKENS:
            return NumericResult(None, STATUS_MISSING)
        try:
            number = float(text)
        except (TypeError, ValueError):
            return NumericResult(None, STATUS_INVALID)
    elif isinstance(raw, (int, float)):
        number = float(raw)
    else:
        return NumericResult(None, STATUS_INVALID)

    if not math.isfinite(number):
        return NumericResult(None, STATUS_INVALID)

    if number < low:
        return NumericResult(clamp(number, low, high), STATUS_CLAMPED_LOW)
    if number > high:
        return NumericResult(clamp(number, low, high), STATUS_CLAMPED_HIGH)
    return NumericResult(number, STATUS_OK)


def safe_score(
    raw: Any,
    *,
    default: float | None = None,
    low: float = SCORE_MIN,
    high: float = SCORE_MAX,
) -> float | None:
    """取单个分数；拿不到时返回 ``default``（默认 ``None``，**不是 5.0**）。

    解析失败时保持缺失，由调用方显式处理，避免补造中间分数。
    """

    result = parse_number(raw, low=low, high=high)
    return result.value if result.value is not None else default


def safe_dimensions(
    raw: Any,
    *,
    allowed_keys: Iterable[str] | None = None,
    low: float = SCORE_MIN,
    high: float = SCORE_MAX,
) -> DimensionsResult:
    """解析一组维度分。

    - 传入非 dict：``None`` 记为 ``__all__=missing``，其他类型记为 ``__all__=invalid``
    - 非法/缺失的键不进入 ``values``（避免下游把"没评"当成 0 分）
    - 传入 ``allowed_keys`` 时只保留这些键（用于和 rubric 对齐）
    """

    if raw is None:
        return DimensionsResult({}, {"__all__": STATUS_MISSING}, [])
    if not isinstance(raw, dict):
        return DimensionsResult({}, {"__all__": STATUS_INVALID}, [])

    wanted = {str(key) for key in allowed_keys} if allowed_keys is not None else None
    values: dict[str, float] = {}
    statuses: dict[str, str] = {}
    invalid_keys: list[str] = []

    for key, value in raw.items():
        name = str(key)
        if wanted is not None and name not in wanted:
            continue
        result = parse_number(value, low=low, high=high)
        statuses[name] = result.status
        if result.value is None:
            if result.status == STATUS_INVALID:
                invalid_keys.append(name)
            continue
        values[name] = result.value

    if wanted is not None:
        # 记录 rubric 要求但模型没给的维度，供上层判"未覆盖"
        for name in wanted - set(statuses):
            statuses[name] = STATUS_MISSING

    return DimensionsResult(values, statuses, invalid_keys)


def is_failure_status(status: str) -> bool:
    """状态码是否属于"必须上报"的失败。"""

    return status in {STATUS_MISSING, STATUS_INVALID}


def failed_keys(statuses: dict[str, str]) -> list[str]:
    """从状态表里挑出所有失败项，按 key 排序便于比对。"""

    return sorted(key for key, status in statuses.items() if is_failure_status(status))
