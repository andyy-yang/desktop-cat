# Overlay Cat — User Manual

Your cat, on your desktop. Overlay Cat is a borderless, transparent, always-on-top window
whose only visible (and clickable) part is the cat itself — built frame-by-frame from real
footage of him. He sleeps, loafs, grooms, plays and reacts on his own schedule; everywhere
he isn't, your clicks pass straight through to whatever is underneath.

## Installing

1. Drag `dist/OverlayCat.app` into `/Applications` (or run it from anywhere — it is fully
   self-contained, clips included).
2. Double-click it. That's it.

First-launch notes:
- Built on this machine, the app carries no quarantine flag, so there is no Gatekeeper
  ceremony locally.
- If you copy it to **another** Mac, the transfer adds quarantine and the ad-hoc signature
  will be blocked on a normal double-click. Right-click the app > **Open** > **Open** once;
  after that it launches normally.
- It no longer asks for access to your Documents folder. Clips ship inside the bundle and
  state lives in `~/Library/Application Support/OverlayCat/`.

The app has no Dock icon — it lives entirely in the menu bar (see below).

## Living with the cat

| You do | He does |
|---|---|
| Single click | Plays a reaction (looks up at you, raises a paw). 10 s cooldown — clicks inside the cooldown get a little boop bounce instead. Afterward he resumes exactly what he was doing, right where he was. |
| Double-click | Changes his activity on the spot. **Available now.** |
| Drag | Carry him anywhere on screen. He mildly disapproves (affection dips). In the **current build** being carried won't change his mind about what he was doing. |
| Pet (slow strokes over his fur) | Affection rises. No fireworks — you'll notice it over time in warmer, more sociable choices. |
| Click while he's asleep | Wakes him up (never a reaction). Freshly woken, he'll usually stretch into a groom. |

The cat is only "solid" where his fur actually is: hover over a paw and he takes the click,
hover over the empty corner of his window and you click straight through to your desktop.

## Why he does what he does

He runs on five needs — **energy, hunger, playfulness, affection, cleanliness** — that drift
over real time, plus a circadian rhythm tied to your local clock. Each activity is scored by
how well it serves the most pressing need, with a touch of randomness and a penalty for
repeating himself, so he stays believable without being a metronome. At night (roughly 23:00
to 08:00) sleep pressure runs almost 3x higher, so he sleeps more and longer; after waking he
is biased toward grooming for a couple of minutes, like the real thing. Playing burns energy
faster, grooming restores cleanliness, and petting tops up affection. His needs persist
between runs — quit for a few hours and he comes back hungrier and scruffier.

## The menu bar 🐱

Click the cat face in your menu bar:

- **Pause / Resume** — freezes him (animation, wandering brain and all) and brings him back.
- **Size** — pick how big he is on screen. **Available now.**
- **Quit Overlay Cat** — saves his state and says goodnight.

## What he knows how to do today

20 clips, all from real footage of him:

- **Sleep (2):** curled-up sleep; dozing loaf on the carpet.
- **Idle (9):** front-facing loaf; loaf in the dark; sunlit loaf on the carpet; loaf with a
  tail flick in the doorway; loaf gazing around and settling toward a doze; sitting on the
  cat tree gazing out the window; sitting on the carpet looking at you; sitting with his
  tail wrapped; sitting seen from behind.
- **Groom (5):** licking a paw on the chair; licking a paw on the treadmill; full paw-wash
  while curled on the couch; two flank-grooming sessions on the chair.
- **Play (2):** rearing up at the feather toy; wrestling a toy on the towel.
- **Reactions (2, click-only):** looking up at the camera with a paw raised; sitting and
  tracking you on the rug.

## Footprint and state

- **RAM:** roughly 40–140 MB depending on the current clip. **CPU:** under 1%.
- **State** (his needs, so he remembers how hungry he is):
  `~/Library/Application Support/OverlayCat/state.json`
- **Factory-reset the cat:** quit him, delete that `state.json`, relaunch. He comes back
  well-rested and reasonably clean.

## Known quirks

- On some clips the window sits a few points below the bottom of the screen, so he can look
  slightly sunken into the Dock area.
- The play clips can shimmer: the feather rear-up has one synthesized motion-blur frame
  mid-lunge, and the back-and-forth loop replays the leap in reverse. A fix is in progress.
- He cannot walk yet — none of the your source videos contain a usable side-on walk. The 3D
  variant (`clips_3d`) walks fine, but the real him needs **one piece of filming: a single
  side-profile walk pass on a dark floor**, and the wandering is already wired up waiting
  for it.
