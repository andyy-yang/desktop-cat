"""Brain facade: ties needs, selector and persistence together behind the
tick() contract. Emits a PlayClip only when the activity changes, plus the
click-reaction interlude (reaction clip, then resume of the exact prior clip).
Event semantics are deterministic: pet/drag only feed needs, click reacts (or
wakes), double_click is a scene shuffle — a uniform pick over every
schedulable clip except the one playing, bypassing all gates."""

import math
import random

from .activities import Activity, Catalog, LOOP_BY_CATEGORY
from .clock import Clock
from .commands import BrainEvent, PlayClip
from .needs import NeedsModel
from .selector import ActivitySelector
from .store import PersistenceStore

REACTION_COOLDOWN_S = 10.0
STATE_VERSION = 1


class Brain:
    def __init__(self, clock: Clock, rng: random.Random, catalog: Catalog,
                 store: PersistenceStore):
        self._clock = clock
        self._rng = rng
        self._store = store
        saved = store.load()
        if saved is None:
            self._needs = NeedsModel(rng)
        else:
            self._needs = NeedsModel.from_dict(saved["needs"], rng)
            self._needs.fast_forward(max(clock.wall() - saved["wall"], 0.0))
        self._selector = ActivitySelector(list(catalog.activities), rng)
        self._durations = dict(catalog.durations)
        self._reaction_clips = tuple(catalog.reaction_clips)
        # double-click shuffle pool: every clip of every schedulable activity
        # (reaction clips are never in catalog.activities), in catalog order
        self._shuffle_pool: tuple[tuple[str, Activity, str | None], ...] = tuple(
            (clip, activity, motion)
            for activity in catalog.activities
            for clip, motion in zip(activity.clips, activity.motions))
        self._current: Activity | None = None
        self._current_clip: str | None = None
        self._current_motion: str | None = None
        self._deadline = 0.0
        self._reaction_until: float | None = None
        self._last_reaction_at: float | None = None
        self._last_now: float | None = None
        self._last_wall = clock.wall()

    def tick(self, events: list[BrainEvent]) -> PlayClip | None:
        now = self._clock.now()
        wall = self._clock.wall()
        prev_wall = self._last_wall
        dt = now - self._last_now if self._last_now is not None else 0.0
        self._last_now = now
        self._last_wall = wall

        woke = any(e.kind == "wake" for e in events)
        if woke:
            # system slept between ticks: monotonic dt is meaningless here
            self._needs.fast_forward(max(wall - prev_wall, 0.0))
        else:
            self._needs.tick(dt, wall)
        for event in events:
            self._needs.apply_event(event.kind)

        clicked = any(e.kind == "click" for e in events)
        double_clicked = any(e.kind == "double_click" for e in events)
        sleeping = self._current is not None and self._current.category == "sleep"

        reaction_cut_short = False
        if self._reaction_until is not None:
            if double_clicked:
                # "do something else" overrides the interlude
                self._reaction_until = None
                reaction_cut_short = True
            elif now < self._reaction_until:
                return None
            else:
                # reaction done: resume the exact prior clip in place — no teleporting
                self._reaction_until = None
                return self._resume_command(now)

        if (clicked and not double_clicked and not sleeping
                and self._current is not None and self._reaction_clips
                and (self._last_reaction_at is None
                     or now - self._last_reaction_at >= REACTION_COOLDOWN_S)):
            clip = self._rng.choice(self._reaction_clips)
            duration = self._durations[clip]
            self._reaction_until = now + duration
            self._last_reaction_at = now
            return PlayClip(clip=clip, loop="once", min_seconds=duration)

        if double_clicked:
            # scene shuffle: uniform over every schedulable clip except the
            # one playing — an explicit user command, so sleep's hysteresis
            # gate and all cooldowns are bypassed
            pool = [entry for entry in self._shuffle_pool
                    if entry[0] != self._current_clip]
            if not pool:
                # a lone schedulable clip cannot visibly change; mid-reaction
                # at least resume the prior clip in place
                if reaction_cut_short:
                    return self._resume_command(now)
                return None
            if sleeping:
                # waking implicitly: cooldown sleep re-entry so the next
                # natural reconsideration cannot insta-return to it
                self._selector.wake(wall)
            clip, activity, motion = self._rng.choice(pool)
            self._selector.force(activity, wall)
            duration = self._sample_duration(activity)
            self._deadline = now + duration
            return self._switch_to(activity, clip, motion, duration)

        # pet/drag_start/drag_end only feed needs: a carried or petted cat
        # resumes exactly what it was doing; click during the reaction
        # cooldown is brain-side silence (runtime boops)
        reconsider = self._current is None or now >= self._deadline
        if self._current is not None:
            if sleeping and clicked:
                reconsider = True
                self._selector.wake(wall)
            elif woke:
                reconsider = True
                self._selector.release()
        if not reconsider:
            return None

        chosen = self._selector.choose(self._needs.needs, wall)
        self._deadline = now + self._sample_duration(chosen)
        if chosen is self._current:
            return None
        duration = self._deadline - now
        if chosen.category == "walk":
            motion = self._rng.choice(sorted(set(chosen.motions)))
            clip = self._rng.choice(
                [c for c, m in zip(chosen.clips, chosen.motions) if m == motion])
        else:
            motion = None
            clip = self._rng.choice(chosen.clips)
        return self._switch_to(chosen, clip, motion, duration)

    def _switch_to(self, activity: Activity, clip: str, motion: str | None,
                   duration: float) -> PlayClip:
        self._current = activity
        self._needs.set_mode(activity.category)
        self._current_clip = clip
        self._current_motion = motion
        return PlayClip(
            clip=clip,
            loop=LOOP_BY_CATEGORY[activity.category],
            min_seconds=duration,
            motion=motion,
        )

    def _resume_command(self, now: float) -> PlayClip:
        return PlayClip(
            clip=self._current_clip,
            loop=LOOP_BY_CATEGORY[self._current.category],
            min_seconds=max(self._deadline - now, 0.0),
            motion=self._current_motion,
        )

    def shutdown(self) -> None:
        self._store.save({
            "version": STATE_VERSION,
            "wall": self._clock.wall(),
            "needs": self._needs.to_dict(),
            "activity": self._current.name if self._current is not None else None,
        })

    def _sample_duration(self, activity: Activity) -> float:
        lo = math.log(activity.min_s)
        hi = math.log(activity.max_s)
        if hi == lo:
            return activity.min_s
        mu = (lo + hi) / 2.0
        sigma = (hi - lo) / 4.0
        sampled = self._rng.lognormvariate(mu, sigma)
        return min(max(sampled, activity.min_s), activity.max_s)
