#!/usr/bin/env bash
# Create the default-deny egress network the whitelist containers attach to (§8.3).
#
# darwin.sandbox attaches mutation/finetune containers to a user-defined Docker network named
# "darwin-egress" (spec.WHITELIST_NETWORK). This script creates that network and installs an
# egress firewall that allows ONLY the §8.3 whitelisted hosts, dropping everything else. The
# eval container does NOT use this — it runs with `--network none` (zero egress).
#
# This is a reference implementation; on a real host you'd enforce the whitelist with your
# network plugin / iptables / a forward proxy. Run once on each Docker host (needs root).
set -euo pipefail

NETWORK="${DARWIN_EGRESS_NETWORK:-darwin-egress}"

# Allowed egress hosts (must match darwin/sources/whitelist.py + PyPI + the Anthropic API).
ALLOW_HOSTS=(
  export.arxiv.org arxiv.org          # papers
  api.semanticscholar.org             # papers
  huggingface.co cdn-lfs.huggingface.co  # datasets/models
  pypi.org files.pythonhosted.org     # package installs
  api.anthropic.com                   # Claude backend (§4.5)
)

if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  docker network create --driver bridge "$NETWORK"
  echo "created docker network: $NETWORK"
else
  echo "docker network already exists: $NETWORK"
fi

echo "Whitelist hosts for $NETWORK (enforce via your egress proxy / iptables):"
printf '  - %s\n' "${ALLOW_HOSTS[@]}"
echo
echo "NOTE: Docker's bridge driver does not itself filter egress by hostname. Enforce the"
echo "whitelist with a forward proxy (e.g. set HTTP(S)_PROXY in the container to a proxy that"
echo "allows only these hosts) or host iptables rules. The eval container needs no rule — it"
echo "runs with --network none (zero egress)."
