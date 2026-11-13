"""Request admission and monotonic deadlines for pilot trials."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic


class BudgetStop(RuntimeError):
    pass


@dataclass
class RequestBudget:
    action_limit: int
    call_limit: int
    seconds: int
    scope: str
    clock: Callable[[], float] = monotonic
    actions: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        self.deadline = self.clock() + self.seconds

    def check(self, *, action: bool = False, call: bool = False) -> None:
        if self.clock() >= self.deadline:
            raise BudgetStop(f"{self.scope}_time_limit")
        if action and self.actions >= self.action_limit:
            raise BudgetStop(f"{self.scope}_action_limit")
        if call and self.calls >= self.call_limit:
            raise BudgetStop(f"{self.scope}_model_call_limit")


@dataclass
class TrialBudget:
    trial: RequestBudget
    series: RequestBudget

    def check(self, *, action: bool = False, call: bool = False) -> None:
        self.series.check(action=action, call=call)
        self.trial.check(action=action, call=call)

    def start_call(self) -> None:
        self.check(call=True)
        self.series.calls += 1
        self.trial.calls += 1

    def start_action(self) -> None:
        self.check(action=True)
        self.series.actions += 1
        self.trial.actions += 1

    async def run[T](self, request: Callable[[], Awaitable[T]]) -> T:
        scope = min((self.series, self.trial), key=lambda budget: budget.deadline - budget.clock())
        remaining = scope.deadline - scope.clock()
        if remaining <= 0:
            raise BudgetStop(f"{scope.scope}_time_limit")
        timer = asyncio.timeout(remaining)
        try:
            async with timer:
                return await request()
        except TimeoutError as exc:
            if timer.expired():
                raise BudgetStop(f"{scope.scope}_time_limit") from exc
            raise


async def within_budget[T](budget: TrialBudget | None, request: Callable[[], Awaitable[T]]) -> T:
    return await request() if budget is None else await budget.run(request)
