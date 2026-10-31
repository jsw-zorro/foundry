# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SGLang general-plugin entry point for Foundry (group ``sglang.srt.plugins``).

SGLang imports this module in every process that runs ``load_plugins()``
(launcher entries, each scheduler process) whether or not Foundry is in use,
so it is a top-level module that imports nothing but the standard library:
importing ``foundry`` itself would load the native extension. Everything
else happens only when ``FOUNDRY_GRAPH_EXTENSION_CONFIG`` names a Foundry
TOML (see ``foundry.integration.sglang.plugin``).
"""

from __future__ import annotations

import os

CONFIG_ENV = "FOUNDRY_GRAPH_EXTENSION_CONFIG"


def load() -> None:
    if not os.environ.get(CONFIG_ENV):
        return
    # SGLang's load_plugins() logs a plugin's exception and goes on, which would
    # serve without Foundry (no archive on SAVE, native captures on LOAD) while
    # the config asks for it; SystemExit stops the process instead, as activate()
    # does for its own failures.
    try:
        from foundry.integration.sglang.plugin import activate

        activate(os.environ[CONFIG_ENV])
    except Exception as exc:
        raise SystemExit(
            f"[Foundry] sglang plugin failed to load ({CONFIG_ENV}={os.environ[CONFIG_ENV]}): "
            f"{exc!r}"
        ) from exc
