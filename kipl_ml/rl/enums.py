from collections.abc import KeysView
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, TypeAlias, overload


class Actions(StrEnum):
    DO_NOTHING = "do_nothing"
    SEND_UP = "send_up"
    SEND_DOWN = "send_down"
    DELAY_UP = "delay_up"
    DELAY_DOWN = "delay_down"
    SELECTOR = "selector"
    CLIENT_BRICK_SELECT = "client_brick_select"
    SERVER_BRICK_SELECT = "server_brick_select"


class ActionHeadKeys(StrEnum):
    ACTION_SELECTION = "action_selection"
    SEND_COUNT_U = "send_count_u"
    SEND_BYPASS_U = "send_bypass_u"
    SEND_REPLACE_U = "send_replace_u"
    SEND_COUNT_D = "send_count_d"
    SEND_BYPASS_D = "send_bypass_d"
    SEND_REPLACE_D = "send_replace_d"
    SEND_TIME_U = "send_time_u"
    SEND_TIME_D = "send_time_d"
    DELAY_BINS_U = "delay_bins_u"
    DELAY_BYPASS_U = "delay_bypass_u"
    DELAY_REPLACE_U = "delay_replace_u"
    DELAY_BINS_D = "delay_bins_d"
    DELAY_BYPASS_D = "delay_bypass_d"
    DELAY_REPLACE_D = "delay_replace_d"


class EntropyKeys(StrEnum):
    SELECTION_ENTROPY = "entropy_selection"
    COND_ENTROPY = "cond_entropy"


AHKs = ActionHeadKeys


@dataclass
class ActSend:
    count: int
    after_steps: int
    bypass: bool = False
    replace: bool = False


@dataclass
class ActSendUp(ActSend):
    pass


@dataclass
class ActSendDown(ActSend):
    pass


@dataclass
class ActDelay:
    steps: int
    bypass: bool = False
    replace: bool = False


@dataclass
class ActDelayUp(ActDelay):
    pass


@dataclass
class ActDelayDown(ActDelay):
    pass


@dataclass
class ActDoNothing:
    bypass: bool = False
    replace: bool = False


@dataclass
class ActSelector:
    selected: int
    bypass: bool = False
    replace: bool = False


ActionType: TypeAlias = (
    ActSendUp | ActSendDown | ActDelayUp | ActDelayDown | ActDoNothing | ActSelector
)


@dataclass
class StepAction:
    time_bin: int
    _actions: dict[Actions, ActionType] = field(default_factory=dict)

    @property
    def acts(self) -> tuple[ActionType, ...]:
        return tuple(self._actions.values())

    @overload
    def __getitem__(self, key: Literal[Actions.SEND_UP]) -> ActSendUp: ...
    @overload
    def __getitem__(self, key: Literal[Actions.SEND_DOWN]) -> ActSendDown: ...
    @overload
    def __getitem__(self, key: Literal[Actions.DELAY_UP]) -> ActDelayUp: ...
    @overload
    def __getitem__(self, key: Literal[Actions.DELAY_DOWN]) -> ActDelayDown: ...
    @overload
    def __getitem__(self, key: Literal[Actions.DO_NOTHING]) -> ActDoNothing: ...
    @overload
    def __getitem__(self, key: Literal[Actions.SELECTOR]) -> ActSelector: ...
    @overload
    def __getitem__(self, key: Literal[Actions.CLIENT_BRICK_SELECT]) -> ActSelector: ...
    @overload
    def __getitem__(self, key: Literal[Actions.SERVER_BRICK_SELECT]) -> ActSelector: ...
    @overload
    def __getitem__(self, key: Actions) -> ActionType: ...

    def __getitem__(self, key: Actions) -> ActionType:
        return self._actions[key]

    def __setitem__(self, key: Actions, value: ActionType) -> None:
        self._actions[key] = value

    def __pop__(self, key: Actions) -> ActionType:
        return self._actions.pop(key)

    def __contains__(self, key: Actions) -> bool:
        return key in self._actions

    def keys(self) -> KeysView[Actions]:
        return self._actions.keys()

    def has(self, key: Actions) -> bool:
        return key in self._actions


class NoAction(StepAction):
    def __init__(self, time: int):
        super().__init__(time)
        self._actions = {Actions.DO_NOTHING: ActDoNothing()}


StepActions: TypeAlias = list[StepAction | NoAction]
