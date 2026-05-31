"""Git checkpointing & the "green" contract (ARCHITECTURE.md §4.4).

Each offspring container is a Git repo on a branch `offspring/<id>`. "Green" is concrete:
- a commit is green iff the smoke test (§4.4.1) exited 0 on its tree;
- green commits are recorded two ways — the commit-message prefix `darwin-green:` **and** a
  moving `last-green` tag that advances to each new green commit.

"Revert to last green" = `git reset --hard last-green`. If the `last-green` tag is absent at
the end of the window (no green commit was produced — likely the weaker local backend), the
**zero-green fallback** (§4.3) applies: reset to the base commit, which is the unchanged clone
of survivor S and green by construction. The final genome is therefore *always* a green commit
— finetuning never runs on broken code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GREEN_PREFIX = "darwin-green:"
LAST_GREEN_TAG = "last-green"
_GIT_USER = ("DARWIN", "darwin@local")


class GitCheckpointer:
    """Owns the offspring repo's branch, green commits, and the last-green tag."""

    def __init__(self, repo: Path | str):
        self.repo = Path(repo)
        self.base_commit: str | None = None  # the clone-of-S baseline (zero-green fallback)

    # ------------------------------------------------------------------ git plumbing
    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            capture_output=True,
            text=True,
            check=check,
        )

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    def has_last_green(self) -> bool:
        return self._git("tag", "-l", LAST_GREEN_TAG).stdout.strip() == LAST_GREEN_TAG

    def last_green(self) -> str | None:
        if not self.has_last_green():
            return None
        return self._git("rev-parse", LAST_GREEN_TAG).stdout.strip()

    # ------------------------------------------------------------------ lifecycle
    def init_offspring(self, offspring_id: str, parent_survivor: str = "unknown") -> str:
        """Make the genome dir a repo on branch `offspring/<id>` with a base commit.

        The base commit captures the cloned-from-S tree (green by construction); it is the
        zero-green fallback target. Returns the base commit sha.
        """
        if not (self.repo / ".git").exists():
            self._git("init")
            self._git("config", "user.name", _GIT_USER[0])
            self._git("config", "user.email", _GIT_USER[1])

        # Ensure a base commit exists before branching (`checkout -B` needs a born HEAD), then
        # point the offspring branch at it. The base captures the clone-of-S tree.
        if self._git("rev-parse", "HEAD", check=False).returncode != 0:
            self._git("add", "-A")
            self._git(
                "commit",
                "--allow-empty",
                "-m",
                f"darwin: offspring {offspring_id} cloned from {parent_survivor}",
            )
        self._git("checkout", "-B", f"offspring/{offspring_id}")
        self.base_commit = self.head()
        return self.base_commit

    def commit_green(self, summary: str) -> str:
        """Commit the current tree as a green checkpoint and advance `last-green`.

        Called only after a passing smoke test (§4.4). Returns the new commit sha. Uses
        `--allow-empty` so a green re-run with no further edits still advances the tag.
        """
        self._git("add", "-A")
        self._git("commit", "--allow-empty", "-m", f"{GREEN_PREFIX} {summary}")
        sha = self.head()
        self._git("tag", "-f", LAST_GREEN_TAG, sha)
        return sha

    def revert_to_last_green(self) -> str:
        """Hard-reset to the last green commit, or the base clone if there is none (§4.3)."""
        target = LAST_GREEN_TAG if self.has_last_green() else self._require_base()
        self._git("reset", "--hard", target)
        return self.head()

    def finalize_genome(self) -> tuple[str, bool]:
        """Ensure HEAD is a green commit; return (final_sha, fell_back_to_clone).

        fell_back_to_clone is True when no green commit was produced this window, so the
        offspring is the unchanged clone of S (→ `mutation_failed`, §4.3).
        """
        if self.has_last_green():
            self._git("reset", "--hard", LAST_GREEN_TAG)
            return self.head(), False
        self._git("reset", "--hard", self._require_base())
        return self.head(), True

    def _require_base(self) -> str:
        if self.base_commit is None:
            raise RuntimeError("init_offspring() must run before reverting")
        return self.base_commit
