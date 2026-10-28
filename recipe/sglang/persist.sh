# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
# Sourced by the serve scripts: `foundry_persist save|load` sets the bash array PERSIST for
# `sglang serve` on whichever route the installed SGLang has (see README "Activation").
#   - SGLang with the Foundry adapter (sglang.srt.utils.foundry_adapter, e.g. foundry-org/sglang
#     foundry-dep): PERSIST = --cuda-graph-persistence <mode> --cuda-graph-persistence-config <toml>.
#   - SGLang without it (upstream main): Foundry loads as an SGLang plugin. PERSIST stays empty and
#     FOUNDRY_GRAPH_EXTENSION_CONFIG names a copy of the TOML with `mode = "<mode>"` added (the plugin
#     reads the mode from the TOML; the shared foundry.toml has none). The copy is written to
#     FOUNDRY_PLUGIN_TOML_DIR (default: the directory of the TOML).
# FOUNDRY_TOML overrides the TOML (default: foundry.toml next to the scripts).
foundry_persist() {
  local mode=$1 toml=${FOUNDRY_TOML:-$SCRIPT_DIR/foundry.toml}
  [[ $mode == save || $mode == load ]] || { echo "foundry_persist: mode must be save or load" >&2; return 2; }
  [[ $mode == load ]] && export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}"
  # Locate the adapter without importing sglang (find_spec of a top-level package runs no code).
  if python3 -c 'import importlib.util as u, os, sys; s = u.find_spec("sglang"); a = s and s.origin and os.path.join(os.path.dirname(s.origin), "srt", "utils", "foundry_adapter"); sys.exit(not (a and (os.path.isfile(a + ".py") or os.path.isdir(a))))' 2>/dev/null; then
    PERSIST=(--cuda-graph-persistence "$mode" --cuda-graph-persistence-config "$toml")
    return
  fi
  local dir=${FOUNDRY_PLUGIN_TOML_DIR:-$(dirname "$toml")}
  local plugin_toml=$dir/.foundry.$mode.toml
  { echo "mode = \"$mode\""; grep -Ev '^[[:space:]]*mode[[:space:]]*=' "$toml"; } > "$plugin_toml" || return 1
  export FOUNDRY_GRAPH_EXTENSION_CONFIG=$plugin_toml
  PERSIST=()
  echo "[recipe] sglang has no Foundry adapter: plugin route, FOUNDRY_GRAPH_EXTENSION_CONFIG=$plugin_toml" >&2
}
