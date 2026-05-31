# Objectives

Central research question (ARCHITECTURE.md §1.1): can an AI system independently improve its
own code and architecture — discovering, testing, and keeping ideas that move it toward
state-of-the-art — with no human writing the improvements?

Current high-level direction: bootstrap the evolutionary loop on a small/cheap base model
with the Claude mutation backend to prove it produces fitness gains at all (Phase 4), before
trusting the local backend (Phase 5) or scaling to the 32B target (Phase 6).

_This file is rewritten each generation by the global-memory pass (§7.4)._
