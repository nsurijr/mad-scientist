"""HarnessHalt: the harness, not the agent, ends the session (hackathon patch P9).

Raised when strict replay misses the cache (the gateway answers HTTP 409), or when the model server,
gateway or tool cache fails. It derives from BaseException, so P8's handling of failing tool calls
(except Exception) never turns it into a message the agent could react to. projects/default.py
exits with HALT_EXIT_CODE, which harness/runner.py reports as a problem.
"""

HALT_EXIT_CODE = 3


class HarnessHalt(BaseException):
    pass
