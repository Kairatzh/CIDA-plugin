from typing import List, Dict, Any

class BaseCallbackHandler:
    """
    Base class for implementing CIDA agent event listeners.
    """
    def on_turn_start(self, turn_idx: int):
        pass

    def on_deliberation_start(self, rounds: int, variance: float):
        pass

    def on_deliberation_end(self, rounds_used: int):
        pass

    def on_deliberation_step(self, round_idx: int, disagreement: float):
        pass

    def on_rollback(self, retry_idx: int, reason: str):
        pass

    def on_uncertainty_high(self, uncertainty: float, threshold: float):
        pass

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        pass

    def on_tool_result(self, tool_name: str, result: str):
        pass

    def on_turn_end(self, response: str):
        pass

    def on_guardrail_violation(self, similarity: float):
        pass


class AgentCallbackManager:
    """
    Manages and invokes a list of callback handlers during agent execution.
    """
    def __init__(self, handlers: List[BaseCallbackHandler] = None):
        self.handlers = handlers or []

    def add_handler(self, handler: BaseCallbackHandler):
        self.handlers.append(handler)

    def on_turn_start(self, turn_idx: int):
        for handler in self.handlers:
            handler.on_turn_start(turn_idx)

    def on_deliberation_start(self, rounds: int, variance: float):
        for handler in self.handlers:
            handler.on_deliberation_start(rounds, variance)

    def on_deliberation_end(self, rounds_used: int):
        for handler in self.handlers:
            handler.on_deliberation_end(rounds_used)

    def on_deliberation_step(self, round_idx: int, disagreement: float):
        for handler in self.handlers:
            handler.on_deliberation_step(round_idx, disagreement)

    def on_rollback(self, retry_idx: int, reason: str):
        for handler in self.handlers:
            handler.on_rollback(retry_idx, reason)

    def on_uncertainty_high(self, uncertainty: float, threshold: float):
        for handler in self.handlers:
            handler.on_uncertainty_high(uncertainty, threshold)

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        for handler in self.handlers:
            handler.on_tool_call(tool_name, args, kwargs)

    def on_tool_result(self, tool_name: str, result: str):
        for handler in self.handlers:
            handler.on_tool_result(tool_name, result)

    def on_turn_end(self, response: str):
        for handler in self.handlers:
            handler.on_turn_end(response)

    def on_guardrail_violation(self, similarity: float):
        for handler in self.handlers:
            handler.on_guardrail_violation(similarity)


class StdOutCallbackHandler(BaseCallbackHandler):
    """
    A default callback handler that logs agent events to stdout in a clean format.
    """
    def on_rollback(self, retry_idx: int, reason: str):
        print(f"⚠️ [Latent Rollback Handler] Retry {retry_idx} initiated. Reason: {reason}")

    def on_uncertainty_high(self, uncertainty: float, threshold: float):
        print(f"🛡️ [CIDA Uncertainty Guard Handler] High latent uncertainty ({uncertainty:.4f} > {threshold:.4f})")

    def on_guardrail_violation(self, similarity: float):
        print(f"🛡️ [CIDA Guardrail Handler] Violation warning: similarity {similarity:.4f}")

    def on_deliberation_step(self, round_idx: int, disagreement: float):
        # Optional verbose trace printout
        pass
