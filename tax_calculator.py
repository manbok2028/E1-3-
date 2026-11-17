"""A deliberately simplified progressive-tax calculator for learning.

This module is *not* a Korean tax filing tool.  Its made-up brackets exist to
show the calculation flow: income minus deductions becomes a taxable base, and
each bracket is taxed only for its own range.  Real filing must use the current
official guidance and the taxpayer's complete circumstances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class TaxBracket:
    """One sample progressive range ending at upper_limit, or open-ended."""

    upper_limit: Optional[float]
    rate: float


@dataclass(frozen=True)
class TaxLine:
    """The taxable amount and tax produced by one bracket."""

    lower_limit: float
    upper_limit: Optional[float]
    taxable_amount: float
    rate: float
    tax: float


@dataclass(frozen=True)
class TaxCalculation:
    """A transparent learning result with every intermediate value."""

    gross_income: float
    deductions: float
    taxable_base: float
    lines: Sequence[TaxLine]
    estimated_tax: float

    @property
    def effective_rate(self) -> float:
        """Estimated tax divided by gross income; zero if income is zero."""

        return 0.0 if self.gross_income == 0 else self.estimated_tax / self.gross_income


# These are intentionally simple sample rates, not statutory Korean tax rates.
LEARNING_BRACKETS = (
    TaxBracket(10_000_000, 0.05),
    TaxBracket(30_000_000, 0.10),
    TaxBracket(None, 0.20),
)


def calculate_learning_tax(
    gross_income: float,
    deductions: float,
    brackets: Sequence[TaxBracket] = LEARNING_BRACKETS,
) -> TaxCalculation:
    """Calculate a sample progressive tax breakdown with validated inputs."""

    _validate_nonnegative_number(gross_income, "총소득")
    _validate_nonnegative_number(deductions, "공제액")
    if not brackets:
        raise ValueError("세율 구간이 하나 이상 필요합니다.")

    taxable_base = max(0.0, float(gross_income) - float(deductions))
    remaining = taxable_base
    lower_limit = 0.0
    lines: List[TaxLine] = []

    for position, bracket in enumerate(brackets):
        _validate_bracket(bracket, lower_limit, position == len(brackets) - 1)
        if bracket.upper_limit is None:
            amount = remaining
        else:
            width = bracket.upper_limit - lower_limit
            amount = min(remaining, width)
        tax = amount * bracket.rate
        lines.append(TaxLine(lower_limit, bracket.upper_limit, amount, bracket.rate, tax))
        remaining -= amount
        if remaining <= 0:
            break
        if bracket.upper_limit is not None:
            lower_limit = bracket.upper_limit

    if remaining > 0:
        raise ValueError("마지막 세율 구간은 상한이 없는 구간이어야 합니다.")

    return TaxCalculation(
        gross_income=float(gross_income),
        deductions=float(deductions),
        taxable_base=taxable_base,
        lines=tuple(lines),
        estimated_tax=sum(line.tax for line in lines),
    )


def _validate_nonnegative_number(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("{}은(는) 0 이상의 숫자여야 합니다.".format(label))


def _validate_bracket(bracket: TaxBracket, lower_limit: float, is_last: bool) -> None:
    if not isinstance(bracket, TaxBracket):
        raise ValueError("세율 구간 형식이 올바르지 않습니다.")
    if isinstance(bracket.rate, bool) or not isinstance(bracket.rate, (int, float)) or not 0 <= bracket.rate <= 1:
        raise ValueError("세율은 0 이상 1 이하의 숫자여야 합니다.")
    if bracket.upper_limit is None:
        if not is_last:
            raise ValueError("상한 없는 구간은 마지막에만 둘 수 있습니다.")
    elif isinstance(bracket.upper_limit, bool) or not isinstance(bracket.upper_limit, (int, float)) or bracket.upper_limit <= lower_limit:
        raise ValueError("각 구간 상한은 이전 구간보다 커야 합니다.")
