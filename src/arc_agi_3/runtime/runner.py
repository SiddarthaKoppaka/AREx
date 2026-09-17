"""Local environment loop over the callback-driven cognitive session."""

from arc_agi_3.adapters.protocols import EnvironmentAdapter

from .session import CognitiveSession
from .state import RunResult


class EpisodeRunner:
    def __init__(
        self,
        *,
        environment: EnvironmentAdapter,
        session: CognitiveSession,
    ) -> None:
        self.environment, self.session = environment, session
        self.io, self.events = session.io, session.io.events
        self.config, self.manifest = session.io.config, session.manifest

    def _episode(self, session: CognitiveSession) -> RunResult:
        observation = self.environment.reset()
        session.start(observation)
        while (action := session.next_action()) is not None:
            session.observe(self.environment.step(action))
        if session.result is None:
            raise RuntimeError("session stopped without a result")
        return session.result

    def run(self) -> RunResult:
        try:
            return self._episode(self.session)
        except Exception as error:
            return self.session.fail(error)
