"""Channel message bus connecting agents.

Each downstream channel (data / pred / mech / critic / probe) is a function
that appends the incoming message to the target agent's memory, invokes the
agent handler, appends the reply, and logs the exchange to the shared
state. This guarantees every inter-agent exchange is recorded and
attributed, mirroring the bus design in Guo et al. SI.
"""

from __future__ import annotations

from typing import Protocol

from .state import Message, SharedState


class AgentLike(Protocol):
    name: str
    memory: list[Message]

    def handle(self, msg: Message, state: SharedState) -> Message:
        ...


class MessageBus:
    def __init__(self, state: SharedState, channel: str):
        self.state = state
        self.channel = channel

    def ask(self, agent: AgentLike, content: str,
            payload: dict | None = None) -> Message:
        incoming = Message(sender="central", content=content,
                           payload=payload or {})
        agent.memory.append(incoming)
        reply = agent.handle(incoming, self.state)
        agent.memory.append(reply)
        self.state.log_message(incoming)
        self.state.log_message(reply)
        self.state.log_exchange(self.channel, content, reply.content)
        return reply
