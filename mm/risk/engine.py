from __future__ import annotations

from mm.core.interfaces.actor import AbstractRiskEngine
from mm.core.models.execution import Action, ActorState
from mm.core.models.market import MarketEvent


class PassThroughRiskEngine(AbstractRiskEngine):
    def validate(self, actions: list[Action], state: ActorState, event: MarketEvent) -> list[Action]:
        return actions

