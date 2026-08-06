"""A small procedural 2D platformer built with pygame.

Run it with:
    python main.py

For automated testing/CI, a headless smoke test is available:
    python main.py --smoke-test

Importing this module (e.g. from a test file) does NOT open a window or start
the game loop -- everything that touches pygame's display/audio/event system
happens inside main() (or the individual init_*() functions it calls), never
at import time. Pure logic (level generation, save/load, upgrade costs, etc.)
can be imported and exercised directly.
"""

import argparse
import array
import colorsys
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pygame

# ---------------- paths ----------------
# Resolve every file path relative to this script, not the current working
# directory, so the game works the same whether you run it from its own
# folder, a shortcut, or `python /some/other/place/main.py`.
BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR
HERO_SPRITESHEET_PATH = ASSET_DIR / "hero_spritesheet.png"


def _app_dir():
    """Where the save file should live -- not necessarily wherever a packaged build
    happens to unpack its bundled code/assets to internally (BASE_DIR above, which stays
    correct for HERO_SPRITESHEET_PATH regardless of platform: Nuitka extracts bundled
    data files there, so that lookup is right exactly as it is).

    Plain `python main.py` (not compiled): BASE_DIR is already the right, visible
    folder -- use it as-is, on every platform.

    Windows, compiled (`__compiled__` is a global Nuitka only defines in a compiled
    module, so this whole function body below is a no-op otherwise): next to the real,
    user-visible .exe, so a packaged game is just "two files sitting together" like a
    portable app is expected to be on Windows. A `--onefile` build unpacks to a fresh
    throwaway temp folder every single launch, so BASE_DIR there is useless for anything
    meant to persist -- NUITKA_ONEFILE_DIRECTORY is the environment variable Nuitka
    itself sets to the directory containing the actual .exe the user launched, confirmed
    empirically (not guessed) by compiling a one-line diagnostic script and inspecting
    its output. sys.argv[0]'s directory is the fallback for a `--standalone` build (no
    onefile unpacking, so no such env var -- sys.argv[0] is already the real exe path
    there) or on the off chance the env var is ever unset.

    Linux/macOS, compiled: deliberately NOT "next to the binary" the way Windows is --
    an AppImage mounts itself read-only at a fresh temp path every launch (the same
    "unpacks somewhere throwaway" problem as Windows onefile, just via squashfs+FUSE
    instead), and a Flatpak's install location isn't even writable by the app at all.
    Follows the XDG Base Directory spec instead (~/.local/share/<app>, or
    $XDG_DATA_HOME if set) -- what every well-behaved Linux app does, portable-app
    conventions don't really exist there, and it keeps working no matter how the game
    ends up packaged (AppImage, Flatpak, a plain standalone folder, ...)."""
    if "__compiled__" in globals():
        if sys.platform == "win32":
            onefile_dir = os.environ.get("NUITKA_ONEFILE_DIRECTORY")
            if onefile_dir:
                return Path(onefile_dir)
            return Path(sys.argv[0]).resolve().parent
        xdg_data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        data_dir = Path(xdg_data_home) / "PygamePlatformer"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    return BASE_DIR


SAVE_FILE = str(_app_dir() / "save.txt")  # next to the real .exe once packaged -- see _app_dir()

# ---------------- core constants ----------------
TS = 40  # tile size in pixels
SCREEN_W, SCREEN_H = 800, 440

GRAVITY = 0.8
MAX_FALL_SPEED = 20
JUMP_STRENGTH = -15
MOVE_SPEED = 5
COYOTE_TIME = 120        # ms grace period to jump after walking off a ledge
JUMP_BUFFER = 120        # ms a jump press is remembered before landing
HIT_INVINCIBILITY_MS = 1000  # grace period after a shield-absorbed hit before you can be hit again
RESPAWN_INVINCIBILITY_MS = 1500  # grace period after respawning, so spawning next to a hazard/
                                  # enemy (e.g. a charger that immediately locks on) can't chain
                                  # straight into another death before you've even had a chance to move
STOMP_TOLERANCE = 6      # px of forgiveness on the "was above the enemy" stomp check
AUTOSAVE_INTERVAL_MS = 8000  # how often unsaved coin/score progress gets auto-flushed to disk
COIN_VALUE = 10
STOMP_REWARD = 50
RARE_COIN_VALUE = 50
SAVE_VERSION = 5  # bumped: trails were reordered/rebalanced so price correlates with bonus
                  # strength (see TRAILS) -- equipped_trail_id positions shifted, so an
                  # old save's equipped trail is reset to "none" on load (ownership is
                  # untouched, since trail_owned_* is keyed by name, not position)

# ---------------- fixed-timestep simulation architecture ----------------
# Every velocity/acceleration constant in this file (GRAVITY, MOVE_SPEED, JUMP_STRENGTH,
# DASH_SPEED, the sine-wave "self.t += 0.08" phase steps on flying enemies, Camera's
# "* 0.15" smoothing factor, Particle's "+= 0.25" gravity, ...) is written as a plain
# per-update-call increment, the same way it always has been. What makes the *game*
# frame-rate independent is not rewriting all of those into per-second units and
# multiplying by a variable dt -- it's guaranteeing that every gameplay update() call
# happens at a strictly constant rate (SIM_HZ, below) no matter how fast or slow the
# player's monitor/GPU can render, so "one update() call" always represents exactly the
# same fixed slice of game time it always implicitly did.
#
# Concretely (see run_fixed_updates()/game_loop() further down): rendering happens as
# often as the FPS cap (or the display) allows, but gameplay simulation only advances in
# fixed SIM_DT_MS chunks, accumulated against real elapsed time. At a steady 60 FPS this
# is indistinguishable from the old "one update per rendered frame" behavior. Below 60
# FPS, more than one fixed tick runs per rendered frame to keep the simulation in lock-
# step with real time (so the game never runs in slow motion just because rendering
# fell behind) -- capped at MAX_CATCHUP_TICKS so a huge stall (window drag, breakpoint,
# a slow asset load) can't make the game try to frantically replay minutes of missed
# time at once (the classic fixed-timestep "spiral of death"; any backlog beyond the cap
# is simply dropped, same as pausing briefly would look from the player's perspective).
# Above 60 FPS, most rendered frames run zero fixed ticks. Naively redrawing the latest
# simulated state on those frames would only repaint the same positions until the next
# tick lands, which looks stuttery well above SIM_HZ -- so draw_playing_frame() instead
# blends each rendered sprite (and the camera) between its pre-tick and post-tick
# position using the leftover accumulator as a 0..1 blend factor (`alpha`, returned by
# run_fixed_updates()). This never touches the actual simulated .rect -- it's a
# render-only offset computed fresh every frame -- so it cannot desync physics/collision
# no matter how it's tuned. See _snapshot_render_positions()/_interp_rect().
#
# Every gameplay system's *timers* (coyote time, jump buffer, hit invincibility, dash
# cooldown, hazard cycles, particle/popup lifetimes, camera shake, autosave, ...) read
# _now_ms() instead of pygame.time.get_ticks() directly, for exactly one reason: when
# several fixed ticks run back-to-back to catch up (the low-FPS case above), they execute
# in a fraction of a millisecond of *real* wall-clock time, so back-to-back
# pygame.time.get_ticks() reads would return nearly the same value for all of them --
# collapsing "5 ticks (~83ms of game time) elapsed" down to "~0ms elapsed" and quietly
# breaking every elapsed-time-based timer during exactly the frame-rate range (well under
# 60 FPS) this whole system exists to make correct. The offset only ever grows, by
# exactly SIM_DT_MS per extra tick within a single burst, and holds steady between
# bursts -- so it never resets mid-session, but it also never needs to: nothing compares
# a _now_ms() reading against a raw pygame.time.get_ticks() one, only against another
# stored _now_ms() reading, and a shared constant bias cancels out of every such
# subtraction. Direct calls in tests that never go through run_fixed_updates() (i.e.
# almost all of them -- calling Player.update() etc. directly, not via game_loop()) see
# offset permanently at 0, so _now_ms() is simply pygame.time.get_ticks() for them.
# game_loop() zeroes both this and the accumulator on entry, so catch-up bursts from an
# earlier test/session can never bleed into the next one.
SIM_HZ = 60                  # fixed simulation rate; every constant above is implicitly
                              # tuned for "one update() call = 1/SIM_HZ of a second" and
                              # would need retuning if this ever changed
SIM_DT_MS = 1000.0 / SIM_HZ   # ms of *game* time simulated per fixed update tick
MAX_CATCHUP_TICKS = 5         # hard cap on fixed ticks per rendered frame -- see above
_TICK_EPSILON_MS = 1e-6        # float-rounding slack for the accumulator loop; see run_fixed_updates()

# ---------------- FPS cap (rendering only -- see Settings) ----------------
# How often run_fixed_updates()/draw_playing_frame() get called (i.e. how often the
# window repaints) is entirely separate from SIM_HZ above: this only throttles
# pygame.time.Clock.tick() in game_loop(), which in turn controls how much real time
# elapses between iterations and therefore how "chunky" the render alpha (see
# draw_playing_frame()) can get. 0 means uncapped, matching pygame.time.Clock.tick(0)'s
# own "no limit" convention -- chosen deliberately so fps_cap can be handed straight to
# clock.tick() with no translation.
FPS_CAP_UNLIMITED = 0
FPS_CAP_DEFAULT = 60
FPS_CAP_MIN = 1
FPS_CAP_MAX = 1000     # generous ceiling for hand-typed values; no real display exceeds this
FPS_CAP_STEP = 5        # per-click increment for the Settings screen's -/+ buttons

_tick_time_offset_ms = 0.0  # nudged forward during a multi-tick catch-up burst; see _now_ms()
_accumulator_ms = 0.0        # real ms of simulation time owed but not yet run; see run_fixed_updates()


def _now_ms():
    """The clock every gameplay system (Player, enemies, hazards, particles, popups,
    camera shake, autosave, ...) reads instead of pygame.time.get_ticks() directly --
    see the fixed-timestep architecture comment above for exactly why. Purely cosmetic,
    non-gameplay UI animation (e.g. the Trails menu's swatch color preview) should keep
    using pygame.time.get_ticks() directly instead: those run continuously while
    browsing menus, when the fixed-tick simulation (and this clock, functionally) isn't
    advancing at all.

    Always an int, like pygame.time.get_ticks() itself -- _tick_time_offset_ms
    accumulates in SIM_DT_MS (a repeating fraction, 1000/60) steps, and plenty of
    existing gameplay code divides an elapsed-time reading and uses it as a list index
    (e.g. HERO_DIE_FRAMES[elapsed // DIE_ANIM_INTERVAL]), which requires an int just like
    it always got from pygame.time.get_ticks() before this function existed."""
    return int(pygame.time.get_ticks() + _tick_time_offset_ms)


# ---------------- upgrade balance table ----------------
# Every purchasable effect is a small per-level increment rather than one big unlock, and
# every upgrade has a hard max level -- see UPGRADE_MAX_LEVELS below. Keeping the *rate*
# of each effect here (instead of scattered through Player/gameplay code) is what makes
# this tunable without hunting through the whole file.
SPEED_PER_LEVEL = 0.4          # move speed bonus per Speed Boots level (max +2.4 at level 6)
JUMP_PER_LEVEL = 0.5           # jump power bonus per Spring Boots level (max +2.5 at level 5)
COYOTE_BONUS_PER_LEVEL = 30    # ms added to COYOTE_TIME per Ledge Grace level
JUMP_BUFFER_BONUS_PER_LEVEL = 30  # ms added to JUMP_BUFFER per Quick Reflex level
COIN_BONUS_PER_LEVEL = 0.10    # +% coin pickup value per Coin Charm level (capped at 3 levels)
STOMP_BONUS_PER_LEVEL = 0.10   # +% stomp reward per Heavy Boots level (capped at 4 levels)
SCORE_MULT_PER_LEVEL = 0.10    # +% SCORE ONLY (not wallet) per Trophy Polish level
SHOP_DISCOUNT_PER_LEVEL = 0.05  # -% off every shop price per Bargainer level (capped at 2 levels)
LUCKY_CHARM_CHANCE_PER_LEVEL = 0.08  # chance any given coin tile spawns as a rare coin instead

MAGNET_RANGES = {1: 70, 2: 110, 3: 160, 4: 220}   # px pull radius by Coin Magnet level
MAGNET_PULL_SPEED = 7

GLIDE_MAX_FRAMES = 45           # ~0.75s of glide "stamina", refills the instant you land
GLIDE_FALL_SPEED = 3            # capped fall speed while gliding (vs. MAX_FALL_SPEED=20)
GLIDE_EXTRA_FRAMES_PER_LEVEL = 25  # bonus stamina from Feather Fall level 2+ ("Aerodynamics")

ENEMY_RADAR_RANGE_BONUS_PER_LEVEL = 220  # extra px of Danger Sense range from level 2 ("Keen Senses")

DASH_SPEED = 14
DASH_DURATION_MS = 160
DASH_COOLDOWN_MS = 1500
DASH_COOLDOWN_MIN_MS = 900       # floor so dash_cooldown levels can't be reduced to near-zero
DASH_COOLDOWN_PER_LEVEL = 300

CHARGE_TRIGGER_RANGE = 220       # px; ChargerEnemy speeds up toward the player within this range
CHARGE_SPEED = 6
DIVE_TRIGGER_RANGE = 150         # px; DiveEnemy swoops down at the player within this range
DIVE_SPEED = 5

PROJECTILE_SPEED = 6
PROJECTILE_LIFETIME_MS = 3000    # a stray shot despawns after this long even if it hits nothing
TURRET_FIRE_INTERVAL_MIN_MS = 1400
TURRET_FIRE_INTERVAL_MAX_MS = 2200
TURRET_TELEGRAPH_MS = 450        # visible wind-up before firing, so a shot is always dodgeable

WIND_ZONE_PUSH = 1.6             # px/frame lateral push -- well under MOVE_SPEED, always resistible

# -- Mastery-tier upgrade balance (see the "Mastery" category below) --
COMBO_MOMENTUM_PER_LEVEL = 0.15  # +% stomp reward per consecutive aerial stomp (chain), per level
COMBO_MOMENTUM_CAP = 0.50        # hard ceiling on the chain bonus regardless of level/chain length

# -- New enemy behavior tuning --
SPLITTER_STOMP_SCALE = 0.6    # SplitterEnemy's own reward, reduced since its shards pay out the rest
SHARD_STOMP_SCALE = 0.35      # each ShardEnemy's reward (0.6 + 0.35 + 0.35 = 1.3x a normal stomp overall)
SHARD_SPEED_BONUS = 2         # shards move faster than their parent, escalating the encounter
HOMING_TRIGGER_RANGE = 320    # px; HomingFlyerEnemy only tracks the player within this range
HOMING_MAX_SPEED = 3          # deliberately well under MOVE_SPEED so walking away always outpaces it
SPITTER_RANGE = 260           # px; SpitterEnemy only lobs shots at a player within this range
SPITTER_FIRE_INTERVAL_MIN_MS = 1700
SPITTER_FIRE_INTERVAL_MAX_MS = 2600
LOB_GRAVITY = 0.35            # gravity applied to LobbedProjectile -- an arc, not a straight line
LOB_SPEED_X = 4.5
LOB_SPEED_Y = -8.5

# -- New hazard tuning --
CRUMBLE_STAND_MS = 260     # how long the player must stand on it before it starts shaking
CRUMBLE_SHAKE_MS = 500     # visible shake/warning before it collapses
CRUMBLE_GONE_MS = 2200     # how long it stays gone before reforming
CRUMBLE_PARK_Y = -100000   # parked far off the level's actual bounds while "gone" -- guarantees it
                           # can never collide with anything without needing a separate group/flag
                           # in every collision loop that already iterates the `platforms` group
GAS_VENT_CYCLE_MS = 2400
GAS_VENT_WARN_MS = 450        # safe, but visibly building up -- the telegraph
GAS_VENT_ACTIVE_MS = 1300     # dangerous: the gas column is up
GAS_VENT_COLUMN_TILES = 1.6   # the active hazard column's height above the vent, in tiles
ICE_ACCEL = 0.10   # how quickly horizontal velocity blends toward the target speed while on ice
                    # (1.0 would be an instant snap, i.e. normal ground friction)

# Shop upgrades, bought with coins (a separate spendable balance from the permanent
# lifetime Score). Each entry's "costs" list has one price per level; buying maxes out
# once you've bought len(costs) times. "category" groups rows onto shop pages.
UPGRADES = [
    # -- Movement --
    {"key": "speed", "name": "Speed Boots", "category": "Movement",
     "desc": f"+{SPEED_PER_LEVEL:g} move speed per level", "costs": [60, 100, 150, 220, 320, 450]},
    {"key": "jump", "name": "Spring Boots", "category": "Movement",
     "desc": f"+{JUMP_PER_LEVEL:g} jump power per level", "costs": [70, 120, 190, 280, 380]},
    {"key": "coyote", "name": "Ledge Grace", "category": "Movement",
     "desc": f"+{COYOTE_BONUS_PER_LEVEL}ms grace to jump after walking off a ledge", "costs": [90, 150, 220]},
    {"key": "jump_buffer", "name": "Quick Reflex", "category": "Movement",
     "desc": f"+{JUMP_BUFFER_BONUS_PER_LEVEL}ms buffered-jump window before landing", "costs": [90, 150, 220]},
    {"key": "dash", "name": "Dash", "category": "Movement",
     "desc": "Unlocks a quick horizontal dash (Shift)", "costs": [600]},
    {"key": "dash_cooldown", "name": "Quick Dash", "category": "Movement",
     "desc": f"Requires Dash. -{DASH_COOLDOWN_PER_LEVEL}ms dash cooldown per level", "costs": [300, 500]},
    # -- Survival --
    {"key": "shield", "name": "Shield Charges", "category": "Survival",
     "desc": "+1 hit absorbed (shield, recharges each life/level)", "costs": [150, 260, 420, 650]},
    {"key": "vitality", "name": "Extra Heart", "category": "Survival",
     "desc": "+1 extra hit survived after your shields run out", "costs": [400, 800, 1300]},
    {"key": "extra_jump", "name": "Air Jump", "category": "Survival",
     "desc": "Level 1: double jump. Level 2: triple jump. Level 3: quadruple jump.", "costs": [350, 700, 1100]},
    {"key": "glide", "name": "Feather Fall", "category": "Survival",
     "desc": f"Hold Down while falling to glide. Lv2: +{GLIDE_EXTRA_FRAMES_PER_LEVEL} more glide frames.",
     "costs": [500, 700]},
    {"key": "checkpoint", "name": "Checkpoint Flag", "category": "Survival",
     "desc": "Unlocks mid-level respawn checkpoints on level 5+", "costs": [450]},
    {"key": "enemy_radar", "name": "Danger Sense", "category": "Survival",
     "desc": "Arrows point toward nearby enemies. Lv2: much longer range.", "costs": [300, 450]},
    # -- Economy --
    {"key": "magnet", "name": "Coin Magnet", "category": "Economy",
     "desc": "Nearby coins fly to you (range grows per level)", "costs": [120, 220, 350, 500]},
    {"key": "coin_bonus", "name": "Coin Charm", "category": "Economy",
     "desc": f"+{COIN_BONUS_PER_LEVEL:.0%} coin value per level", "costs": [200, 350, 550]},
    {"key": "stomp_bonus", "name": "Heavy Boots", "category": "Economy",
     "desc": f"+{STOMP_BONUS_PER_LEVEL:.0%} stomp reward per level", "costs": [200, 350, 550, 850]},
    {"key": "lucky_charm", "name": "Lucky Charm", "category": "Economy",
     "desc": f"+{LUCKY_CHARM_CHANCE_PER_LEVEL:.0%} chance a coin is a rare {RARE_COIN_VALUE}-coin", "costs": [250, 400, 600, 850]},
    {"key": "shop_discount", "name": "Bargainer", "category": "Economy",
     "desc": f"-{SHOP_DISCOUNT_PER_LEVEL:.0%} off every shop price per level", "costs": [500, 900]},
    {"key": "score_multiplier", "name": "Trophy Polish", "category": "Economy",
     "desc": f"+{SCORE_MULT_PER_LEVEL:.0%} score per level (score only, not coins)", "costs": [400, 750, 1200]},
    # -- Mastery -- higher-tier upgrades that build on an already-owned mechanic or lean
    # into the deeper hazard/enemy variety added alongside them, rather than just being
    # another copy of an existing effect.
    {"key": "second_wind", "name": "Second Wind", "category": "Mastery",
     "desc": "Once per life, a killing hit grants a moment of invincibility instead.",
     "costs": [900]},
    {"key": "combo_momentum", "name": "Combo Momentum", "category": "Mastery",
     "desc": f"Consecutive stomps without landing grow in reward, up to +{COMBO_MOMENTUM_CAP:.0%}.",
     "costs": [500, 850]},
    {"key": "hazard_sense", "name": "Hazard Sense", "category": "Mastery",
     "desc": "Requires Danger Sense. Also flags nearby gas vents/crumbling floors.", "costs": [500]},
]
UPGRADE_DEFAULTS = {u["key"]: 0 for u in UPGRADES}
UPGRADE_MAX_LEVELS = {u["key"]: len(u["costs"]) for u in UPGRADES}
UPGRADE_CATEGORIES = ["Movement", "Survival", "Economy", "Mastery"]

# ---------------- cosmetic trail system ----------------
# Completely separate from upgrades: each trail is a one-time coin purchase (not leveled)
# that you then equip/unequip freely at no extra cost. Trails are primarily cosmetic --
# most have no gameplay effect at all -- but a few carry a small, capped bonus that stays
# active even while the trail's visual effect is hidden (see trail_visible/Trail Menu).
# "style" picks which of the shared particle-rendering treatments in
# maybe_spawn_trail_particles() it uses (circle-shaped "rainbow"/"solid"/"spark", plus
# "square"/"ring"/"comet" for a genuinely different silhouette, not just a recolor) --
# new trails only ever need a new table row, never new drawing code.
#
# Every trail grants a real (small, capped) gameplay bonus now -- none are purely
# cosmetic -- and the table is ordered cheapest-to-priciest with the bonus strength
# rising to match: buying up through the list is a genuine progression, not just a
# wardrobe choice, and the price you paid always tells you roughly how strong the
# trail is. Each bonus *type* appears twice (a cheaper "lesser" version and a pricier
# "greater" version, e.g. Toxic then Stars for rare_coin_chance) so the ladder reads as
# 8 progression pairs rather than 16 unrelated one-offs.
#
# This is position-sensitive (see TRAIL_KEY_TO_ID/_clamp_save_data: equipped_trail_id is
# a 1-based *position* in this list) -- SAVE_VERSION was bumped when this table was
# last reordered so old saves don't end up with the wrong trail silently equipped (see
# its comment). Any *future* new trail should still be appended at the end rather than
# inserted in the middle, unless you're deliberately doing another full rebalance pass
# (bump SAVE_VERSION again if so).
TRAILS = [
    {"key": "rainbow", "name": "Rainbow", "price": 150, "style": "rainbow",
     "desc": "Cycles smoothly through every color. (+2% score)", "bonus": ("score_mult", 0.02)},
    {"key": "aurora", "name": "Aurora", "price": 180, "style": "ring",
     "colors": [(90, 220, 200), (120, 160, 255), (180, 120, 255)],
     "desc": "Shimmering auroral rings ripple outward. (+3% score)", "bonus": ("score_mult", 0.03)},
    {"key": "toxic", "name": "Toxic", "price": 210, "style": "solid",
     "colors": [(140, 220, 60), (90, 180, 40), (180, 255, 110)],
     "desc": "A radioactive haze. (+3% rare coin chance)", "bonus": ("rare_coin_chance", 0.03)},
    {"key": "stars", "name": "Stars", "price": 240, "style": "spark",
     "colors": [(255, 255, 255), (210, 210, 255), (255, 240, 190)],
     "desc": "Tiny stars twinkle in your wake. (+4% rare coin chance)", "bonus": ("rare_coin_chance", 0.04)},
    {"key": "gold_sparkles", "name": "Gold Sparkles", "price": 270, "style": "spark",
     "colors": [(255, 223, 60), (255, 250, 210), (230, 180, 40)],
     "desc": "Glimmering gold dust. (+4% coin value)", "bonus": ("coin_bonus", 0.04)},
    {"key": "confetti", "name": "Confetti", "price": 300, "style": "square",
     "colors": [(255, 105, 180), (255, 215, 60), (90, 200, 255), (120, 255, 140)],
     "desc": "Chunky bursts of colorful confetti. (+5% coin value)", "bonus": ("coin_bonus", 0.05)},
    {"key": "purple_magic", "name": "Purple Magic", "price": 330, "style": "spark",
     "colors": [(180, 90, 255), (230, 150, 255), (140, 60, 220)],
     "desc": "Violet sparkles widen your coin pull. (+5% magnet range)", "bonus": ("coin_radius", 0.05)},
    {"key": "void", "name": "Void", "price": 360, "style": "ring",
     "colors": [(40, 20, 60), (90, 40, 140), (15, 10, 25)],
     "desc": "Space bends behind you, pulling coins closer. (+6% magnet)", "bonus": ("coin_radius", 0.06)},
    {"key": "shadow", "name": "Shadow", "price": 390, "style": "solid",
     "colors": [(60, 55, 70), (90, 80, 100), (40, 35, 50)],
     "desc": "A creeping dark haze. (+6% stomp reward)", "bonus": ("stomp_bonus", 0.06)},
    {"key": "neon_green", "name": "Neon Green", "price": 420, "style": "solid",
     "colors": [(120, 255, 90), (60, 255, 150), (170, 255, 120)],
     "desc": "A bright acid-green afterglow. (+7% stomp reward)", "bonus": ("stomp_bonus", 0.07)},
    {"key": "ice", "name": "Ice", "price": 450, "style": "solid",
     "colors": [(150, 220, 255), (210, 245, 255), (110, 190, 240)],
     "desc": "A frosty shimmer follows you. (+7% jump power)", "bonus": ("jump", 0.07)},
    {"key": "smoke", "name": "Smoke", "price": 480, "style": "solid",
     "colors": [(120, 120, 130), (170, 170, 180), (90, 90, 100)],
     "desc": "A drifting grey haze. (+8% jump power)", "bonus": ("jump", 0.08)},
    {"key": "fire", "name": "Fire", "price": 510, "style": "solid",
     "colors": [(255, 120, 30), (255, 60, 20), (255, 180, 60)],
     "desc": "Warm embers trail behind every step. (+8% move speed)", "bonus": ("speed", 0.08)},
    {"key": "neon_blue", "name": "Neon Blue", "price": 540, "style": "solid",
     "colors": [(0, 255, 200), (0, 180, 255), (80, 220, 255)],
     "desc": "A bright cyan afterglow. (+9% move speed)", "bonus": ("speed", 0.09)},
    {"key": "lightning", "name": "Lightning", "price": 570, "style": "spark",
     "colors": [(255, 245, 140), (230, 230, 255), (255, 255, 255)],
     "desc": "Crackling arcs of static. (+9% dash speed)", "bonus": ("dash_speed", 0.09)},
    {"key": "comet", "name": "Comet", "price": 600, "style": "comet",
     "colors": [(255, 160, 40), (255, 220, 80), (255, 255, 255)],
     "desc": "A blazing comet streaks behind you. (+10% dash speed)", "bonus": ("dash_speed", 0.10)},
]
TRAIL_DEFAULTS = {t["key"]: 0 for t in TRAILS}
TRAILS_BY_KEY = {t["key"]: t for t in TRAILS}
TRAIL_KEY_TO_ID = {t["key"]: i + 1 for i, t in enumerate(TRAILS)}  # 1-based; 0 means "none equipped"

SHOP_CATEGORIES = UPGRADE_CATEGORIES + ["Trails", "Statistics", "Coming Soon"]
SHOP_PAGES = {cat: [u["key"] for u in UPGRADES if u["category"] == cat] for cat in UPGRADE_CATEGORIES}
SHOP_PAGES["Trails"] = [t["key"] for t in TRAILS]

GOAL_COLOR = (255, 215, 0)
SPIKE_COLOR = (180, 180, 190)
SPIKE_TALL_EXTRA_HEIGHT = 10   # px a "tall" spike variant's hitbox extends above a normal tile
SPIKE_TALL_CHANCE = 0.15       # purely cosmetic variety; well under safe jump-clearance margins
ENEMY_COLOR = (90, 40, 140)
FLYER_COLOR = (200, 80, 200)
JUMPER_COLOR = (210, 120, 40)
CHARGER_COLOR = (205, 55, 55)
SHIELDED_COLOR = (80, 180, 190)
DIVE_COLOR = (120, 60, 160)
COIN_COLOR = (255, 223, 60)
RARE_COIN_COLOR = (80, 220, 255)
LAVA_COLOR = (255, 90, 20)
CHECKPOINT_COLOR = (90, 220, 140)
TURRET_COLOR = (160, 60, 60)
PROJECTILE_COLOR = (255, 210, 90)
WIND_ZONE_COLOR = (170, 220, 255)
SPLITTER_COLOR = (60, 150, 90)
SHARD_COLOR = (110, 210, 140)
HOMING_COLOR = (220, 90, 160)
SPITTER_COLOR = (150, 100, 60)
CRUMBLE_COLOR = (140, 110, 90)
CRUMBLE_WARNING_COLOR = (220, 90, 70)
ICE_COLOR = (200, 230, 250)
ICE_TOP_COLOR = (230, 245, 255)
GAS_VENT_COLOR = (130, 200, 110)
GAS_VENT_ACTIVE_COLOR = (90, 220, 130)
WHITE = (255, 255, 255)

# Environment palettes the level cycles through as you climb the level count, purely for
# visual variety over an infinite run -- gameplay-critical colors (spikes, enemies, coins)
# stay fixed across all of them so hazards are always recognizable at a glance.
BIOME_PALETTES = [
    {  # day
        "sky_top": (135, 206, 235), "sky_bottom": (200, 230, 250),
        "hill": (110, 190, 110), "ground": (90, 60, 40), "ground_top": (60, 140, 60),
        "cloud": (255, 255, 255),
    },
    {  # sunset
        "sky_top": (255, 150, 90), "sky_bottom": (255, 205, 140),
        "hill": (190, 120, 80), "ground": (100, 65, 45), "ground_top": (190, 110, 60),
        "cloud": (255, 235, 220),
    },
    {  # night
        "sky_top": (20, 20, 55), "sky_bottom": (60, 45, 95),
        "hill": (35, 35, 70), "ground": (45, 38, 55), "ground_top": (80, 65, 100),
        "cloud": (150, 150, 190),
    },
    {  # dawn
        "sky_top": (255, 190, 140), "sky_bottom": (255, 225, 190),
        "hill": (140, 165, 115), "ground": (95, 68, 48), "ground_top": (130, 150, 85),
        "cloud": (255, 240, 225),
    },
    {  # cavern -- deep, muted underground tones
        "sky_top": (25, 22, 38), "sky_bottom": (55, 45, 70),
        "hill": (45, 42, 60), "ground": (60, 50, 65), "ground_top": (90, 75, 100),
        "cloud": (95, 85, 110),
    },
    {  # volcanic -- dark ash sky with hot embers underfoot
        "sky_top": (40, 15, 15), "sky_bottom": (90, 35, 25),
        "hill": (60, 30, 25), "ground": (55, 35, 30), "ground_top": (200, 90, 40),
        "cloud": (120, 90, 80),
    },
    {  # crystal -- pale, glassy pastels
        "sky_top": (210, 235, 255), "sky_bottom": (235, 245, 255),
        "hill": (170, 200, 220), "ground": (120, 130, 150), "ground_top": (200, 225, 240),
        "cloud": (255, 255, 255),
    },
]


def get_biome(level_num):
    return BIOME_PALETTES[((level_num - 1) // 3) % len(BIOME_PALETTES)]


SAMPLE_RATE = 44100

LEVEL_ROWS = 11
LEVEL_HEIGHT = LEVEL_ROWS * TS  # rows never change, so this stays fixed across all levels
LEVEL_TRANSITION_MS = 1200  # how long the "Level N complete!" banner shows before advancing


# ---------------- level generation (tile map) ----------------
# G=ground  .=empty  P=start  X=goal  ^=spike  ~=lava pool  M=moving spike  Z=timed spike  c=coin
# E=patrol enemy  J=jumper enemy  C=charger enemy  S=shielded enemy  D=dive enemy  U=turret enemy
# ==one-way platform  K=checkpoint flag  >=wind zone (pushes right)  <=wind zone (pushes left)
# B=crumbling floor  I=icy floor  V=gas vent  T=splitter enemy  L=spitter enemy  H=homing flyer
LEVEL_WIDTH_TILE_MILESTONE = 20   # every N levels, the width cap is allowed to grow a bit
LEVEL_WIDTH_TILE_MILESTONE_BONUS = 8
LEVEL_WIDTH_TILE_CAP_BASE = 70
LEVEL_WIDTH_TILE_CAP_MAX = 110    # absolute ceiling no matter how high level_num goes

# ---------------- difficulty curve ----------------
# Every generation parameter below (pit width, spacing, hazard/enemy density) derives from
# this single smooth 0..1 curve instead of several independently-stepping formulas, so
# difficulty climbs in one coherent ramp rather than a patchwork of milestones. It
# saturates but never reaches exactly 1, so very late levels keep climbing, just slowly.
DIFFICULTY_CURVE_SCALE = 25.0   # levels; higher = slower ramp


def _difficulty(level_num):
    return 1 - 1 / (1 + level_num / DIFFICULTY_CURVE_SCALE)


# "Phase" is flavor, not the numeric curve: it gates WHICH extra tricks (hazard combos,
# spike clusters) can appear at all. The continuous curve above controls HOW MUCH of
# everything there is; the phase controls WHAT KINDS of arrangements are allowed to show up.
EARLY_GAME_MAX_LEVEL = 8    # forgiving: narrow pits, wide spacing, no combos/clusters
MID_GAME_MAX_LEVEL = 25     # timing challenges (hazard combos, spike clusters) start here


def _game_phase(level_num):
    if level_num <= EARLY_GAME_MAX_LEVEL:
        return "early"
    if level_num <= MID_GAME_MAX_LEVEL:
        return "mid"
    return "late"


CHARGER_MIN_LEVEL = 10
SHIELDED_MIN_LEVEL = 15
DIVE_MIN_LEVEL = 20
LAVA_MIN_LEVEL = 12
MOVING_SPIKE_MIN_LEVEL = 15
TIMED_SPIKE_MIN_LEVEL = 18
TURRET_MIN_LEVEL = 25
WIND_ZONE_MIN_LEVEL = 20
CHECKPOINT_MIN_LEVEL = 5
CHECKPOINT_SPACING_TILES = 30  # a level needs at least this many tiles per checkpoint added
CHECKPOINT_MAX_COUNT = 3       # hard cap regardless of how long the level gets
CRUMBLE_MIN_LEVEL = 9
SPLITTER_MIN_LEVEL = 11
ICE_MIN_LEVEL = 13
GAS_VENT_MIN_LEVEL = 16
SPITTER_MIN_LEVEL = 19
HOMING_MIN_LEVEL = 23


def generate_level(level_num, rows=LEVEL_ROWS):
    """Procedurally build level `level_num` (1, 2, 3, ... forever). Same level_num always
    produces the same layout (seeded RNG), and difficulty (pit width, hazard density, enemy
    speed/variety, level width) scales up with level_num but is capped so late levels stay
    merely hard, not literally unbeatable. Pure function -- no pygame dependency, safe to
    call at import time or from tests without any display/audio setup."""
    width_cap = min(LEVEL_WIDTH_TILE_CAP_BASE + (level_num // LEVEL_WIDTH_TILE_MILESTONE) * LEVEL_WIDTH_TILE_MILESTONE_BONUS,
                     LEVEL_WIDTH_TILE_CAP_MAX)
    cols = min(44 + level_num * 3, width_cap)
    rng = random.Random(1000 + level_num)

    floor = ["G"] * cols
    feature = ["."] * cols
    feature[1] = "P"

    difficulty = _difficulty(level_num)
    phase = _game_phase(level_num)

    # Pit width starts pinned at its narrowest (2 tiles) through the whole early game and
    # only widens toward the existing safety cap of 4 as difficulty climbs -- "easier
    # jumps" early is a real narrower cap here, not just fewer hazards around it.
    max_pit_w = min(2 + round(difficulty * 2), 4)
    # Spacing between features: wide/forgiving early ("wider platforms"), tightening
    # smoothly toward the old floor as difficulty rises, instead of hitting that floor
    # abruptly by level ~12 the way the old step formula did.
    min_gap = max(5, round(10 - difficulty * 5))

    target_spikes = min(1 + level_num, round(2 + difficulty * 20))
    target_enemies = min(1 + level_num // 2, round(1 + difficulty * 10))
    enemy_speed = min(2 + level_num // 4, round(2 + difficulty * 5))

    # flying/jumping enemies only start showing up once the player has seen the basic
    # patroller for a bit, and each new enemy archetype is folded in gradually at its own
    # level milestone so early levels stay simple and later levels keep feeling new.
    enemy_types = ["E"]
    if level_num >= 3:
        enemy_types += ["E", "F", "J"]
    if level_num >= CHARGER_MIN_LEVEL:
        enemy_types += ["C"]
    if level_num >= SPLITTER_MIN_LEVEL:
        enemy_types += ["T"]
    if level_num >= SHIELDED_MIN_LEVEL:
        enemy_types += ["S"]
    if level_num >= SPITTER_MIN_LEVEL:
        enemy_types += ["L"]
    if level_num >= DIVE_MIN_LEVEL:
        enemy_types += ["D"]
    if level_num >= HOMING_MIN_LEVEL:
        enemy_types += ["H"]
    if level_num >= TURRET_MIN_LEVEL:
        enemy_types += ["U"]

    hazard_types = ["^"]
    if level_num >= LAVA_MIN_LEVEL:
        hazard_types += ["~"]
    if level_num >= MOVING_SPIKE_MIN_LEVEL:
        hazard_types += ["M"]
    if level_num >= GAS_VENT_MIN_LEVEL:
        hazard_types += ["V"]
    if level_num >= TIMED_SPIKE_MIN_LEVEL:
        hazard_types += ["Z"]

    # Hazard "combos" (a spike immediately followed by an enemy) and spike "clusters"
    # (2 adjacent hazard tiles instead of 1) are mid/late-game-only variety. Neither
    # touches the pit-width cap or the min_gap spacing rule above -- they only change
    # WHAT fills an already-safe, already-spaced-out placement slot, so they add timing
    # pressure and visual variety without ever creating a new impossible arrangement.
    combo_chance = 0.0 if phase == "early" else min(0.35, max(0.0, difficulty - 0.2) * 0.6)
    cluster_chance = 0.0 if phase == "early" else min(0.4, difficulty * 0.5)

    pos = 6
    end_zone = cols - 4
    spikes_placed = 0
    enemies_placed = 0
    pits = []
    spike_band = 0.15 + difficulty * 0.20  # slice of rolls resolving to a hazard: 15% -> 35%

    while pos < end_zone:
        roll = rng.random()
        step = rng.randint(min_gap, min_gap + 3)
        if roll < 0.3:
            w = rng.randint(2, max_pit_w)
            if pos + w < end_zone:
                for i in range(pos, pos + w):
                    floor[i] = "."
                pits.append((pos, w))
                pos += w + step
                continue
        elif roll < 0.3 + spike_band and spikes_placed < target_spikes:
            cluster_w = 2 if (cluster_chance and rng.random() < cluster_chance and pos + 1 < end_zone) else 1
            hazard_choice = rng.choice(hazard_types)
            for i in range(cluster_w):
                if pos + i < end_zone:
                    feature[pos + i] = hazard_choice
            spikes_placed += 1
            if combo_chance and rng.random() < combo_chance and enemies_placed < target_enemies:
                enemy_pos = pos + cluster_w + 1
                if enemy_pos < end_zone:
                    feature[enemy_pos] = rng.choice(enemy_types)
                    enemies_placed += 1
                    pos = enemy_pos
        elif roll < 0.3 + spike_band + 0.25 and enemies_placed < target_enemies:
            feature[pos] = rng.choice(enemy_types)
            enemies_placed += 1
        pos += step

    feature[cols - 2] = "X"

    # Mid-level checkpoint flag(s) on solid, hazard-free ground, once levels are long
    # enough to make one meaningful -- and, for longer levels, more than one: a checkpoint
    # is added for roughly every CHECKPOINT_SPACING_TILES of usable level length, capped at
    # CHECKPOINT_MAX_COUNT, spaced evenly across the level so a death late in a long level
    # never means redoing the whole thing from scratch. Only actually changes respawn
    # behavior once the Checkpoint Flag upgrade is purchased (see Player.update) -- placing
    # the tiles here is gated purely on level_num so generate_level stays deterministic and
    # upgrade-agnostic. Touching each one in turn just keeps moving the respawn point
    # forward (see Player.update()'s checkpoint loop), so placement order doesn't matter.
    if level_num >= CHECKPOINT_MIN_LEVEL:
        usable_span = max(1, end_zone - 6)
        checkpoint_count = min(CHECKPOINT_MAX_COUNT, max(1, usable_span // CHECKPOINT_SPACING_TILES))
        for k in range(1, checkpoint_count + 1):
            target = 6 + usable_span * k // (checkpoint_count + 1)
            for offset in range(usable_span // 2 + 1):
                for cand in (target - offset, target + offset):
                    if 6 <= cand < end_zone and floor[cand] == "G" and feature[cand] == ".":
                        feature[cand] = "K"
                        break
                else:
                    continue
                break

    # Wind zones: a short run of tiles that nudge the player sideways. Mid/late-game only,
    # always placed over solid, hazard/enemy-free ground (never over a pit -- being pushed
    # while already trying to clear a gap is exactly the kind of "unfair" combination the
    # generator otherwise avoids), and WIND_ZONE_PUSH is small enough that holding the
    # opposing direction always beats it (see Player.update()), so it's a nudge, not a trap.
    if level_num >= WIND_ZONE_MIN_LEVEL:
        wind_direction = 1 if rng.random() < 0.5 else -1
        wind_w = rng.randint(3, 5)
        wind_start = None
        for attempt_start in range(8, end_zone - wind_w):
            if all(floor[i] == "G" and feature[i] == "." for i in range(attempt_start, attempt_start + wind_w)):
                wind_start = attempt_start
                break
        if wind_start is not None:
            wind_char = ">" if wind_direction > 0 else "<"
            for i in range(wind_start, wind_start + wind_w):
                feature[i] = wind_char

    # coins: a risky one hovering over most pits, plus a scatter along open floor
    for pit_start, pit_w in pits:
        if rng.random() < 0.7:
            feature[pit_start + pit_w // 2] = "c"
    for i in range(6, end_zone):
        if floor[i] == "G" and feature[i] == "." and rng.random() < 0.12:
            feature[i] = "c"

    # Crumbling floor: rare, isolated floor-tile hazards that alter footing instead of
    # dealing damage -- pressure to keep moving rather than an instant-death hazard. Only
    # ever replaces a "G" tile that's solidly in the middle of a floor run (both neighbors
    # are still plain floor, not a pit) with nothing already on it, so it can never be the
    # one tile standing between the player and clearing a gap, or double up with an
    # existing hazard/coin/enemy. The neighbor check also makes two crumbling tiles ever
    # landing next to each other impossible: converting floor[i] makes floor[i] no longer
    # "G", so the very next iteration's floor[i-1]=="G" check already excludes it.
    if level_num >= CRUMBLE_MIN_LEVEL and phase != "early":
        for i in range(8, end_zone - 2):
            if (floor[i] == "G" and floor[i - 1] == "G" and floor[i + 1] == "G" and feature[i] == "."
                    and rng.random() < 0.05):
                floor[i] = "B"

    # Icy floor: a short slippery patch (see Player.update()'s on_ice blending) placed
    # over solid ground with a plain-floor margin on both sides -- like wind zones, never
    # right up against a pit, so a slide can't carry the player straight into a gap.
    if level_num >= ICE_MIN_LEVEL and phase != "early":
        ice_w = rng.randint(3, 5)
        for attempt_start in range(8, end_zone - ice_w):
            margin_lo, margin_hi = attempt_start - 2, attempt_start + ice_w + 2
            if margin_lo >= 0 and margin_hi <= cols and all(
                    floor[j] == "G" and feature[j] == "." for j in range(margin_lo, margin_hi)):
                for j in range(attempt_start, attempt_start + ice_w):
                    floor[j] = "I"
                break

    # optional elevated bonus platforms: wide floating islands 2 tiles above the floor,
    # lined with coins. Never required to finish the level -- just an opportunistic
    # detour for players who want the extra coins, and the only source of verticality in
    # an otherwise flat floor-plus-pits level.
    #
    # These are one-way platforms (tile "="), not solid ground: jumping toward one at
    # normal running speed almost always either clips its underside/left edge like a wall
    # or sails clean over the top before descending back down -- the horizontal speed is
    # just too fast relative to the jump arc for a solid block to be landable without
    # pixel-perfect timing. One-way semantics (collide only when falling onto the top,
    # pass through freely from below/the side) fixes the wall-clipping, but empirically
    # (measured by simulating hundreds of jump-timing/approach-distance combinations) even
    # then a *narrow* platform is essentially never actually landed on -- the ~2-3-tile
    # height window where the player's arc is above the platform only lines up with a
    # 2-3 tile-wide target by luck. Making the platform wide (6-9 tiles) instead of precise
    # is what actually makes it reliably reachable, since it turns "land in a narrow spot at
    # the right instant" into "land somewhere in a wide zone during a wide time window."
    platform_row = ["."] * cols
    bonus_coin_row = ["."] * cols
    p = 10
    while p < end_zone - 3:
        if rng.random() < 0.2:
            w = rng.randint(6, 9)
            if p + w < end_zone and all(floor[i] == "G" for i in range(p, p + w)):
                for i in range(p, p + w):
                    platform_row[i] = "="
                for i in range(p + 1, p + w - 1, 2):
                    bonus_coin_row[i] = "c"
                p += w + rng.randint(6, 10)
                continue
        p += rng.randint(4, 7)

    # A second, higher-elevation "staircase" tier: a smaller one-way platform floating
    # above a stretch of the low tier, offering extra bonus coins for climbing further.
    # Mid/late game only, and only ever placed directly above a span that's already "="
    # in platform_row -- guaranteeing it's always reachable by hopping up from a platform
    # the player can already stand on, never floating with no way up to it.
    high_platform_row = ["."] * cols
    high_coin_row = ["."] * cols
    if rows >= 7 and phase != "early":
        p = 10
        while p < end_zone - 3:
            if all(ch == "=" for ch in platform_row[p:p + 4]) and rng.random() < 0.35:
                w = rng.randint(3, 5)
                if p + w < end_zone and all(ch == "=" for ch in platform_row[p:p + w]):
                    for i in range(p, p + w):
                        high_platform_row[i] = "="
                    for i in range(p + 1, p + w - 1, 2):
                        high_coin_row[i] = "c"
                    p += w + rng.randint(6, 10)
                    continue
            p += rng.randint(4, 7)

    blank_row = "." * cols
    rows_list = [blank_row] * (rows - 2)
    if rows >= 4 and any(ch != "." for ch in platform_row):
        rows_list[rows - 3] = "".join(platform_row)
    if rows >= 5 and any(ch != "." for ch in bonus_coin_row):
        rows_list[rows - 4] = "".join(bonus_coin_row)
    if rows >= 6 and any(ch != "." for ch in high_platform_row):
        rows_list[rows - 5] = "".join(high_platform_row)
    if rows >= 7 and any(ch != "." for ch in high_coin_row):
        rows_list[rows - 6] = "".join(high_coin_row)

    level_map = rows_list + ["".join(feature), "".join(floor)]
    return level_map, enemy_speed


# ---------------- sprites ----------------
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h, ground_color=None, ground_top_color=None):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(ground_color or BIOME_PALETTES[0]["ground"])
        pygame.draw.rect(self.image, ground_top_color or BIOME_PALETTES[0]["ground_top"], (0, 0, w, 8))
        self.rect = self.image.get_rect(topleft=(x, y))


class OneWayPlatform(pygame.sprite.Sprite):
    """A thin floating ledge: solid only when the player falls onto its top surface,
    otherwise fully passable (no wall-clipping the side, no bonking the underside).
    See the comment in generate_level() for why regular solid platforms don't work here."""

    def __init__(self, x, y, w, color=None):
        super().__init__()
        h = 14
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(self.image, color or BIOME_PALETTES[0]["ground_top"], (0, 0, w, h), border_radius=4)
        self.rect = self.image.get_rect(topleft=(x, y))


class InvisibleWall(pygame.sprite.Sprite):
    """A solid, unrendered boundary. build_level() places one just past each end of every
    level so the player can never walk off the generated map into empty space -- most
    importantly just past the goal, where jumping clean over the finish flag used to run
    you straight off the right edge of the level into the void. Solid to collision (lives
    in the `platforms` group, so the normal wall-resolution code handles it for free) but
    fully transparent, so it never appears on screen."""

    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)  # left fully transparent
        self.rect = self.image.get_rect(topleft=(x, y))


class IceFloor(Platform):
    """A solid floor tile (inherits Platform's rect/solid-collision semantics, so no new
    collision code is needed anywhere) that's also slippery. Player.update() detects this
    with a plain `isinstance(p, IceFloor)` check against the same `platforms` group it
    already receives every frame -- deliberately not a separate sprite group, so adding
    this hazard needed zero changes to build_level()/load_level()'s signatures. Visually a
    pale, glassy tint instead of the normal ground color."""

    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h, ICE_COLOR, ICE_TOP_COLOR)


class CrumblingPlatform(pygame.sprite.Sprite):
    """A floor tile that collapses a moment after the player stands on it, then reforms
    a couple seconds later -- pressure to keep moving rather than an instant-death hazard
    like a spike. Lives permanently in the `platforms` group (so it's part of the normal
    solid-collision sweep for the player AND every ground-patrolling enemy, for free) and
    cycles solid -> shaking (telegraph) -> gone -> solid. While "gone" its rect is parked
    far outside the level (CRUMBLE_PARK_Y) instead of being removed from the group --
    every existing rect-based collision check already treats "far away" as "not
    colliding" for free, with no special-casing needed anywhere else."""

    def __init__(self, x, y, w, h, ground_color=None, ground_top_color=None):
        super().__init__()
        self._home_rect = pygame.Rect(x, y, w, h)
        self._solid_img = self._build_image(w, h, ground_color or CRUMBLE_COLOR,
                                             ground_top_color or BIOME_PALETTES[0]["ground_top"])
        self._shake_img = self._build_image(w, h, CRUMBLE_WARNING_COLOR, CRUMBLE_WARNING_COLOR)
        self.image = self._solid_img
        self.rect = self._home_rect.copy()
        self.state = "solid"        # "solid" -> "shaking" -> "gone" -> back to "solid"
        self._state_since = 0
        self._stand_since = None    # tick the player first stepped onto it, or None

    @staticmethod
    def _build_image(w, h, base_color, top_color):
        surf = pygame.Surface((w, h))
        surf.fill(base_color)
        pygame.draw.rect(surf, top_color, (0, 0, w, 8))
        for cx in (w // 3, w * 2 // 3):  # crack lines -- signals "breakable" at a glance
            pygame.draw.line(surf, (0, 0, 0), (cx, 8), (cx - 4, h - 4), 2)
        return surf

    def update(self, player_rect=None, *_args, **_kwargs):
        now = _now_ms()
        if self.state == "solid":
            self.image = self._solid_img
            standing_on = (player_rect is not None
                           and self.rect.left < player_rect.centerx < self.rect.right
                           and abs(player_rect.bottom - self.rect.top) <= 6)
            if standing_on:
                if self._stand_since is None:
                    self._stand_since = now
                elif now - self._stand_since >= CRUMBLE_STAND_MS:
                    self.state = "shaking"
                    self._state_since = now
            else:
                self._stand_since = None
        elif self.state == "shaking":
            self.image = self._shake_img if (now // 80) % 2 == 0 else self._solid_img
            if now - self._state_since >= CRUMBLE_SHAKE_MS:
                self.state = "gone"
                self._state_since = now
                self._stand_since = None
                self.rect = pygame.Rect(self._home_rect.x, CRUMBLE_PARK_Y, self._home_rect.w, self._home_rect.h)
        elif self.state == "gone":
            if now - self._state_since >= CRUMBLE_GONE_MS:
                self.state = "solid"
                self.rect = self._home_rect.copy()


CHECKPOINT_TRIGGER_WIDTH_TILES = 3  # trigger column is this many tiles wide, centered on the flag


class Checkpoint(pygame.sprite.Sprite):
    """A flag planted on solid ground. `rect` is just the visual flag graphic; the actual
    activation check uses `trigger_rect`, a tall column spanning the full level height so
    the player can't dodge a checkpoint by jumping over or above it -- touching the flag,
    sailing over it at the peak of a jump, or just walking through its column all count.
    Whether touching the trigger actually moves your respawn point is decided in
    Player.update() based on whether the Checkpoint Flag upgrade has been bought, so the
    tile can be placed on every long-enough level (see generate_level) without generation
    needing to know anything about the player's upgrades.

    `activate()` is called once by Player.update(), the first time the player's respawn
    point actually moves to this flag -- swaps to a brighter gold flag so it's obvious at
    a glance which checkpoint is currently active (useful now that a single long level can
    have several, see CHECKPOINT_MAX_COUNT). Calling it again is a harmless no-op."""

    def __init__(self, x, y):
        super().__init__()
        w, h = TS, TS
        self.activated = False
        self._img_inactive = self._build_image(w, h, CHECKPOINT_COLOR)
        self._img_active = self._build_image(w, h, (255, 223, 60))
        self.image = self._img_inactive
        self.rect = self.image.get_rect(topleft=(x, y))
        trigger_w = TS * CHECKPOINT_TRIGGER_WIDTH_TILES
        self.trigger_rect = pygame.Rect(self.rect.centerx - trigger_w // 2, 0, trigger_w, LEVEL_HEIGHT)

    @staticmethod
    def _build_image(w, h, flag_color):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pole_x = w // 2 - 2
        pygame.draw.rect(surf, (90, 70, 50), (pole_x, 4, 4, h - 8))
        pygame.draw.polygon(surf, flag_color, [(pole_x + 4, 4), (pole_x + 22, 12), (pole_x + 4, 20)])
        return surf

    def activate(self):
        if not self.activated:
            self.activated = True
            self.image = self._img_active


class WindZone(pygame.sprite.Sprite):
    """A single tile of a wind zone -- generate_level() lays several of these side by
    side to form a wider zone (see WIND_ZONE_MIN_LEVEL). Purely a trigger area (`rect`)
    plus a `push` value Player.update() adds to horizontal velocity while overlapping it;
    WIND_ZONE_PUSH is small enough relative to MOVE_SPEED that holding the opposing
    direction always beats it, so it's a nudge, never something you're helplessly blown
    around by. The drawn arrows show which way it pushes."""

    def __init__(self, x, y, direction):
        super().__init__()
        w, h = TS, TS
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        base_color = (*WIND_ZONE_COLOR, 70)
        self.image.fill(base_color)
        for i in range(2):
            cx = w // 2 + (i - 0.5) * 14
            tip = (cx + 7 * direction, h // 2)
            back1 = (cx - 7 * direction, h // 2 - 6)
            back2 = (cx - 7 * direction, h // 2 + 6)
            pygame.draw.polygon(self.image, (*WIND_ZONE_COLOR, 190), [tip, back1, back2])
        self.rect = self.image.get_rect(topleft=(x, y))
        self.push = WIND_ZONE_PUSH * direction


class Spike(pygame.sprite.Sprite):
    """`extra_height` optionally makes this a visually taller spike variant (see
    SPIKE_TALL_EXTRA_HEIGHT/SPIKE_TALL_CHANCE) -- it only extends upward into the empty
    row above, never widens the footprint, and stays a small fraction of the player's full
    jump height, so it adds visual variety without threatening jump clearance. The peak's
    horizontal position within the tile is randomized slightly per-instance (via the plain
    unseeded `random` module, like every other purely cosmetic wobble in this file --
    Coin's bob phase, Particle's velocity, etc. -- so it never perturbs the seeded RNG
    stream generate_level()/build_level() use for actual layout decisions)."""

    def __init__(self, x, y, w, h, extra_height=0):
        super().__init__()
        total_h = h + extra_height
        peak_ratio = random.uniform(0.18, 0.34)
        self.image = pygame.Surface((w, total_h), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, SPIKE_COLOR, [(0, total_h), (w // 2, int(total_h * peak_ratio)), (w, total_h)])
        self.rect = self.image.get_rect(topleft=(x, y - extra_height))
        self.hitbox = self.rect.inflate(-14, -10)
        self.hitbox.bottom = self.rect.bottom


class HazardPool(pygame.sprite.Sprite):
    """A wider, flat instant-hazard (lava/acid) -- mechanically identical to Spike (touch
    it, take_hit() fires) but visually distinct so late-level hazard variety is readable
    at a glance rather than just "more of the same spike."."""

    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        top = int(h * 0.35)
        pygame.draw.rect(self.image, LAVA_COLOR, (0, top, w, h - top))
        pygame.draw.polygon(self.image, (255, 180, 60), [(0, top), (w // 2, int(h * 0.1)), (w, top)])
        self.rect = self.image.get_rect(topleft=(x, y))
        self.hitbox = self.rect.inflate(-6, -4)
        self.hitbox.bottom = self.rect.bottom


class MovingSpike(pygame.sprite.Sprite):
    """A Spike that patrols back and forth over a short range -- same instant-hazard
    hitbox/collision as Spike, just not stationary. Lives in the same `spikes` group as
    plain Spike/HazardPool, so no new collision-handling code is needed anywhere else;
    only its own position needs a per-frame update() (a no-op on the other hazard types
    via pygame.sprite.Sprite's default update(), so calling spikes.update() every frame
    is always safe)."""

    def __init__(self, x, y, w, h, travel=TS * 2, speed=2):
        super().__init__()
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, SPIKE_COLOR, [(0, h), (w // 2, int(h * 0.25)), (w, h)])
        self.rect = self.image.get_rect(topleft=(x, y))
        self.base_x = x
        self.travel = travel
        self.vel_x = speed
        self.hitbox = self.rect.inflate(-14, -10)
        self.hitbox.bottom = self.rect.bottom

    def update(self, *_args, **_kwargs):
        self.rect.x += self.vel_x
        if self.rect.x > self.base_x + self.travel or self.rect.x < self.base_x - self.travel:
            self.vel_x *= -1
        self.hitbox = self.rect.inflate(-14, -10)
        self.hitbox.bottom = self.rect.bottom


TIMED_SPIKE_CYCLE_MS = 2600
TIMED_SPIKE_EXTENDED_MS = 1100    # dangerous
TIMED_SPIKE_WARN_MS = 500         # safe, but visibly about to extend
# retracted (safe, no warning) for whatever's left of the cycle: 1000ms here


class TimedSpike(pygame.sprite.Sprite):
    """A spike that cycles retracted (safe, flush with the floor) -> warning (safe, but
    visibly rising/flashing) -> extended (dangerous) -> back to retracted. Lives in the
    same `spikes` group as Spike/HazardPool/MovingSpike, so no new collision-handling
    code is needed -- only the hitbox is swapped for an empty one while safe. A random
    per-instance phase offset means a row of these doesn't all extend in unison, so
    there's always a safe moment to cross if you wait and watch."""

    def __init__(self, x, y, w, h):
        super().__init__()
        self.w, self.h = w, h
        self.phase_offset = random.randint(0, TIMED_SPIKE_CYCLE_MS)
        self._retracted_img = self._build_image(w, h, 0.85, SPIKE_COLOR, bump=True)
        self._warning_img = self._build_image(w, h, 0.6, (255, 200, 60), bump=True)
        self._extended_img = self._build_image(w, h, 0.25, SPIKE_COLOR, bump=False)
        self.image = self._retracted_img
        self.rect = self.image.get_rect(topleft=(x, y))
        self.hitbox = pygame.Rect(0, 0, 0, 0)

    @staticmethod
    def _build_image(w, h, peak_ratio, color, bump):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        if bump:
            # low, flush bump -- reads as "not dangerous right now"
            bump_h = max(3, int(h * (1 - peak_ratio) * 0.5))
            pygame.draw.rect(surf, color, (0, h - bump_h, w, bump_h), border_radius=2)
        else:
            pygame.draw.polygon(surf, color, [(0, h), (w // 2, int(h * peak_ratio)), (w, h)])
        return surf

    def update(self, *_args, **_kwargs):
        t = (_now_ms() + self.phase_offset) % TIMED_SPIKE_CYCLE_MS
        retracted_ms = TIMED_SPIKE_CYCLE_MS - TIMED_SPIKE_EXTENDED_MS - TIMED_SPIKE_WARN_MS
        if t < retracted_ms:
            self.image = self._retracted_img
            self.hitbox = pygame.Rect(0, 0, 0, 0)
        elif t < retracted_ms + TIMED_SPIKE_WARN_MS:
            self.image = self._warning_img
            self.hitbox = pygame.Rect(0, 0, 0, 0)  # still safe -- this is the telegraph
        else:
            self.image = self._extended_img
            self.hitbox = self.rect.inflate(-14, -10)
            self.hitbox.bottom = self.rect.bottom


class GasVent(pygame.sprite.Sprite):
    """A floor-level vent that cycles idle -> warning (rising bubbles) -> active (a tall
    gas column dangerous to touch) -> back to idle. Lives in the same `spikes` group as
    every other hazard, so no new collision-handling code is needed -- only the hitbox
    toggles between empty (safe) and the full tall image (dangerous). Unlike TimedSpike
    (which threatens at foot level, cleared with a sidestep), the column reaches well
    above the vent's own tile, so the only way past it while active is to be airborne
    above it -- a well-timed jump, not just lateral movement, which is what makes this a
    genuinely different obstacle rather than a recolored timed spike."""

    def __init__(self, x, y, w, h):
        super().__init__()
        column_h = int(h * GAS_VENT_COLUMN_TILES)
        self.phase_offset = random.randint(0, GAS_VENT_CYCLE_MS)
        self._idle_img = self._build_image(w, h, column_h, 0)
        self._warn_img = self._build_image(w, h, column_h, 1)
        self._active_img = self._build_image(w, h, column_h, 2)
        self.image = self._idle_img
        self.rect = self.image.get_rect(topleft=(x, y - column_h))
        self.hitbox = pygame.Rect(0, 0, 0, 0)

    @staticmethod
    def _build_image(w, h, column_h, stage):
        total_h = h + column_h
        surf = pygame.Surface((w, total_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (90, 90, 100), (w // 2 - 6, total_h - 10, 12, 10), border_radius=2)  # nozzle
        if stage == 1:
            for i in range(3):
                pygame.draw.circle(surf, GAS_VENT_COLOR, (w // 2, max(2, total_h - 14 - i * 10)), 3)
        elif stage == 2:
            pygame.draw.rect(surf, (*GAS_VENT_ACTIVE_COLOR, 150), (w // 4, 0, w // 2, total_h - 10))
        return surf

    def update(self, *_args, **_kwargs):
        t = (_now_ms() + self.phase_offset) % GAS_VENT_CYCLE_MS
        idle_ms = GAS_VENT_CYCLE_MS - GAS_VENT_ACTIVE_MS - GAS_VENT_WARN_MS
        if t < idle_ms:
            self.image = self._idle_img
            self.hitbox = pygame.Rect(0, 0, 0, 0)
        elif t < idle_ms + GAS_VENT_WARN_MS:
            self.image = self._warn_img
            self.hitbox = pygame.Rect(0, 0, 0, 0)  # still safe -- this is the telegraph
        else:
            if self.hitbox.height == 0:
                spawn_particles(self.rect.centerx, self.rect.top, GAS_VENT_ACTIVE_COLOR, count=6, speed=2)
            self.image = self._active_img
            self.hitbox = self.rect.copy()


def patrol_bounce_walls(rect, vel_x, platforms):
    """Shove `rect` back out of any platform it just walked into and reverse direction.
    Shared by every ground-patrolling enemy so the wall-bounce behaves identically.
    Pure geometry helper -- takes plain values/collections, no globals, easy to unit test."""
    for p in platforms:
        if rect.colliderect(p.rect):
            if vel_x > 0:
                rect.right = p.rect.left
            else:
                rect.left = p.rect.right
            return -vel_x
    return vel_x


def patrol_at_ledge(rect, vel_x, platforms):
    """True if there's no ground a step ahead in the direction of travel -- i.e. about
    to walk off an edge, which should also trigger a turnaround."""
    ahead_x = rect.centerx + (12 if vel_x > 0 else -12)
    probe = pygame.Rect(ahead_x - 1, rect.bottom + 2, 2, 4)
    return not any(probe.colliderect(p.rect) for p in platforms)


def simple_patrol_step(rect, vel_x, platforms):
    """The 'walk, bounce off walls, turn around at ledges' pattern shared by every enemy
    whose whole movement is just that with no extra behavior layered on top (SplitterEnemy,
    ShardEnemy). Enemy/ChargerEnemy/ShieldedEnemy predate this helper and interleave their
    own extra logic around the same two primitives (patrol_bounce_walls/patrol_at_ledge)
    directly, so they're left as-is rather than retrofitted."""
    rect.x += vel_x
    vel_x = patrol_bounce_walls(rect, vel_x, platforms)
    if patrol_at_ledge(rect, vel_x, platforms):
        vel_x *= -1
    return vel_x


def _make_facing_sprite(w, h, color, facing):
    """Small shared enemy-face renderer: a solid body plus a white eye near the leading
    edge so patrol direction is readable at a glance, not just inferable from motion."""
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(color)
    eye_x = w - 11 if facing > 0 else 5
    eye_y = max(6, h // 3)
    pygame.draw.circle(surf, WHITE, (eye_x + 3, eye_y), 4)
    pygame.draw.circle(surf, (20, 20, 25), (eye_x + 3 + (1 if facing > 0 else -1), eye_y), 2)
    return surf


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = ENEMY_COLOR
        self._img = {-1: _make_facing_sprite(TS - 4, TS - 6, ENEMY_COLOR, -1),
                     1: _make_facing_sprite(TS - 4, TS - 6, ENEMY_COLOR, 1)}
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(x + 2, y + 6))

    def update(self, platforms, player_rect=None):
        self.rect.x += self.vel_x
        self.vel_x = patrol_bounce_walls(self.rect, self.vel_x, platforms)
        if patrol_at_ledge(self.rect, self.vel_x, platforms):
            self.vel_x *= -1
        self.image = self._img[-1 if self.vel_x < 0 else 1]


class ChargerEnemy(pygame.sprite.Sprite):
    """Patrols slowly like Enemy, but speeds up and locks onto the player's direction
    once they're roughly on the same floor and within CHARGE_TRIGGER_RANGE -- still
    respects walls/ledges exactly like a normal patroller (patrol_bounce_walls /
    patrol_at_ledge always run), so it can never wall-clip or walk off into a pit while
    charging, just close the distance faster and more aggressively than a plain Enemy."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = CHARGER_COLOR
        self.base_speed = speed
        self._img = {-1: _make_facing_sprite(TS - 4, TS - 6, CHARGER_COLOR, -1),
                     1: _make_facing_sprite(TS - 4, TS - 6, CHARGER_COLOR, 1)}
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(x + 2, y + 6))
        self.charging = False

    def update(self, platforms, player_rect=None):
        self.charging = False
        if player_rect is not None:
            same_row = abs(player_rect.centery - self.rect.centery) < TS * 1.5
            dx = player_rect.centerx - self.rect.centerx
            if same_row and abs(dx) < CHARGE_TRIGGER_RANGE:
                self.charging = True
                self.vel_x = CHARGE_SPEED if dx > 0 else -CHARGE_SPEED

        self.rect.x += self.vel_x
        self.vel_x = patrol_bounce_walls(self.rect, self.vel_x, platforms)
        if patrol_at_ledge(self.rect, self.vel_x, platforms):
            self.vel_x *= -1
            self.charging = False
        if not self.charging:
            self.vel_x = self.base_speed if self.vel_x > 0 else -self.base_speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]


class FlyingEnemy(pygame.sprite.Sprite):
    """Hovers back and forth around its spawn point with a sine-wave bob, ignoring the
    ground entirely -- gives the player something to worry about mid-air, not just underfoot."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = FLYER_COLOR
        self._img = {-1: _make_facing_sprite(TS - 8, TS - 12, FLYER_COLOR, -1),
                     1: _make_facing_sprite(TS - 8, TS - 12, FLYER_COLOR, 1)}
        self.base_x = x + 4
        self.base_y = y + 6
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(self.base_x, self.base_y))
        self.range = TS * 3
        self.t = random.uniform(0, 6.28)

    def update(self, platforms, player_rect=None):
        self.rect.x += self.vel_x
        if abs(self.rect.x - self.base_x) > self.range:
            self.vel_x *= -1
        self.t += 0.08
        self.rect.y = self.base_y + int(6 * math.sin(self.t))
        self.image = self._img[-1 if self.vel_x < 0 else 1]


class DiveEnemy(pygame.sprite.Sprite):
    """A bat-like flier: hovers like FlyingEnemy until the player passes underneath and
    within range, then swoops down toward them before drifting back up to its hover
    altitude. Still just another sprite in the `enemies` group -- stomping, spikes/hit
    detection, etc. all work on it exactly like every other enemy, no special-casing
    needed anywhere else."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = DIVE_COLOR
        self._img = {-1: _make_facing_sprite(TS - 8, TS - 12, DIVE_COLOR, -1),
                     1: _make_facing_sprite(TS - 8, TS - 12, DIVE_COLOR, 1)}
        self.base_x = x + 4
        self.base_y = y + 6
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(self.base_x, self.base_y))
        self.range = TS * 3
        self.diving = False

    def update(self, platforms, player_rect=None):
        if player_rect is not None and not self.diving:
            dx = abs(player_rect.centerx - self.rect.centerx)
            if dx < DIVE_TRIGGER_RANGE and player_rect.centery > self.rect.centery:
                self.diving = True

        if self.diving:
            target_y = player_rect.centery if player_rect is not None else self.base_y
            if self.rect.centery < target_y - 6:
                self.rect.y += DIVE_SPEED
            else:
                self.diving = False
        elif self.rect.y != self.base_y:
            step = 3 if self.rect.y < self.base_y else -3
            self.rect.y += step
            if abs(self.rect.y - self.base_y) <= 3:
                self.rect.y = self.base_y

        self.rect.x += self.vel_x
        if abs(self.rect.x - self.base_x) > self.range:
            self.vel_x *= -1
        self.image = self._img[-1 if self.vel_x < 0 else 1]


class JumperEnemy(pygame.sprite.Sprite):
    """Patrols like Enemy but hops periodically, so stomping it needs better timing."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = JUMPER_COLOR
        self._img = {-1: _make_facing_sprite(TS - 4, TS - 6, JUMPER_COLOR, -1),
                     1: _make_facing_sprite(TS - 4, TS - 6, JUMPER_COLOR, 1)}
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(x + 2, y + 6))
        self.vel_y = 0
        self.on_ground = False
        self.next_hop = _now_ms() + random.randint(400, 1200)

    def update(self, platforms, player_rect=None):
        now = _now_ms()
        if self.on_ground and now >= self.next_hop:
            self.vel_y = -10
            self.on_ground = False
            self.next_hop = now + random.randint(800, 1600)

        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

        self.rect.x += self.vel_x
        self.vel_x = patrol_bounce_walls(self.rect, self.vel_x, platforms)

        self.rect.y += int(self.vel_y)
        self.on_ground = False
        for p in platforms:
            if self.rect.colliderect(p.rect) and self.vel_y > 0:
                self.rect.bottom = p.rect.top
                self.vel_y = 1
                self.on_ground = True

        if self.on_ground and patrol_at_ledge(self.rect, self.vel_x, platforms):
            self.vel_x *= -1

        self.image = self._img[-1 if self.vel_x < 0 else 1]


class ShieldedEnemy(pygame.sprite.Sprite):
    """Patrols like Enemy but survives one stomp: the first stomp cracks its shield
    (swaps to a dimmer sprite and pays a small consolation reward instead of the full
    stomp reward) and the second stomp actually kills it. Player.update() reads
    `hits_remaining` to decide whether a stomp kills or just cracks -- see the stomp
    branch there."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = SHIELDED_COLOR
        self.hits_remaining = 2
        self._img_shielded = {-1: _make_facing_sprite(TS - 4, TS - 6, SHIELDED_COLOR, -1),
                               1: _make_facing_sprite(TS - 4, TS - 6, SHIELDED_COLOR, 1)}
        cracked_color = tuple(max(0, c - 70) for c in SHIELDED_COLOR)
        self._img_cracked = {-1: _make_facing_sprite(TS - 4, TS - 6, cracked_color, -1),
                              1: _make_facing_sprite(TS - 4, TS - 6, cracked_color, 1)}
        self.vel_x = -speed
        self.image = self._img_shielded[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(x + 2, y + 6))

    def update(self, platforms, player_rect=None):
        self.rect.x += self.vel_x
        self.vel_x = patrol_bounce_walls(self.rect, self.vel_x, platforms)
        if patrol_at_ledge(self.rect, self.vel_x, platforms):
            self.vel_x *= -1
        imgs = self._img_shielded if self.hits_remaining > 1 else self._img_cracked
        self.image = imgs[-1 if self.vel_x < 0 else 1]


class SplitterEnemy(pygame.sprite.Sprite):
    """Patrols like Enemy, but the killing stomp doesn't remove it from play -- instead
    it bursts into two smaller, faster ShardEnemy fragments that scurry off in opposite
    directions (see spawn_shards(), which Player._handle_stomp() calls right before
    killing it). Turns one threat into two, rewarding chasing the fragments down instead
    of treating the stomp as "done" -- a genuinely different stomp interaction than
    ShieldedEnemy's simple "survives one extra hit"."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = SPLITTER_COLOR
        self.reward_scale = SPLITTER_STOMP_SCALE
        self._img = {-1: _make_facing_sprite(TS - 4, TS - 6, SPLITTER_COLOR, -1),
                     1: _make_facing_sprite(TS - 4, TS - 6, SPLITTER_COLOR, 1)}
        self.vel_x = -speed
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(topleft=(x + 2, y + 6))
        self._speed = speed

    def update(self, platforms, player_rect=None):
        self.vel_x = simple_patrol_step(self.rect, self.vel_x, platforms)
        self.image = self._img[-1 if self.vel_x < 0 else 1]

    def spawn_shards(self):
        shard_speed = self._speed + SHARD_SPEED_BONUS
        left = ShardEnemy(self.rect.centerx, self.rect.top + 4, -shard_speed)
        right = ShardEnemy(self.rect.centerx, self.rect.top + 4, shard_speed)
        return [left, right]


class ShardEnemy(pygame.sprite.Sprite):
    """A small, fast fragment left behind by a stomped SplitterEnemy. Doesn't split
    further (no spawn_shards method, so Player._handle_stomp() just kills it outright) and
    pays a reduced stomp reward via reward_scale, the same mechanism SplitterEnemy itself
    uses -- together a stomped Splitter plus both of its shards pays out
    SPLITTER_STOMP_SCALE + 2 * SHARD_STOMP_SCALE, a bit more than a normal enemy in total,
    which is the reward for the extra effort of running two fragments down."""

    def __init__(self, x, y, vel_x):
        super().__init__()
        self.color = SHARD_COLOR
        self.reward_scale = SHARD_STOMP_SCALE
        w, h = TS - 16, TS - 18
        self._img = {-1: _make_facing_sprite(w, h, SHARD_COLOR, -1),
                     1: _make_facing_sprite(w, h, SHARD_COLOR, 1)}
        self.vel_x = vel_x
        self.image = self._img[-1 if self.vel_x < 0 else 1]
        self.rect = self.image.get_rect(midtop=(x, y))

    def update(self, platforms, player_rect=None):
        self.vel_x = simple_patrol_step(self.rect, self.vel_x, platforms)
        self.image = self._img[-1 if self.vel_x < 0 else 1]


class HomingFlyerEnemy(pygame.sprite.Sprite):
    """Hovers with the same sine-wave bob as FlyingEnemy, but -- unlike FlyingEnemy's
    fixed back-and-forth patrol -- slowly drifts to line up under/near the player's
    current x position whenever they're within HOMING_TRIGGER_RANGE. HOMING_MAX_SPEED is
    deliberately well under MOVE_SPEED, so simply walking away always outpaces it: it
    pressures positioning (standing still under one is never truly safe), it doesn't trap."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = HOMING_COLOR
        self._img = {-1: _make_facing_sprite(TS - 8, TS - 12, HOMING_COLOR, -1),
                     1: _make_facing_sprite(TS - 8, TS - 12, HOMING_COLOR, 1)}
        self.base_y = y + 6
        self.vel_x = 0
        self.facing = -1
        self.image = self._img[self.facing]
        self.rect = self.image.get_rect(topleft=(x + 4, self.base_y))
        self.home_speed = min(HOMING_MAX_SPEED, max(1, speed // 2))
        self.t = random.uniform(0, 6.28)

    def update(self, platforms, player_rect=None):
        if player_rect is not None:
            dx = player_rect.centerx - self.rect.centerx
            if abs(dx) > HOMING_TRIGGER_RANGE:
                self.vel_x = 0
            else:
                self.vel_x = self.home_speed if dx > 2 else (-self.home_speed if dx < -2 else 0)
        self.rect.x += self.vel_x
        if self.vel_x != 0:
            self.facing = 1 if self.vel_x > 0 else -1
        self.t += 0.08
        self.rect.y = self.base_y + int(6 * math.sin(self.t))
        self.image = self._img[self.facing]


class Projectile(pygame.sprite.Sprite):
    """A slow-moving shot fired by a TurretEnemy. Travels in a straight horizontal line
    (no arc), dies on hitting solid ground/a wall, or after PROJECTILE_LIFETIME_MS even
    if it hits nothing -- so a missed shot never lingers forever. Not stompable and not
    part of the `spikes` group (it needs its own per-frame movement and wall-collision,
    unlike the mostly-static spikes group); Player.update() checks the `projectiles`
    global directly, the same way it already reads `camera`/`popups`/etc."""

    def __init__(self, x, y, direction):
        super().__init__()
        w, h = 12, 6
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(self.image, PROJECTILE_COLOR, (0, 0, w, h))
        self.rect = self.image.get_rect(center=(x, y))
        self.vel_x = PROJECTILE_SPEED * direction
        self.spawn_time = _now_ms()

    def update(self, platforms=()):
        self.rect.x += self.vel_x
        if _now_ms() - self.spawn_time > PROJECTILE_LIFETIME_MS:
            self.kill()
            return
        for p in platforms:
            if self.rect.colliderect(p.rect):
                self.kill()
                return


class LobbedProjectile(pygame.sprite.Sprite):
    """An arcing shot fired by a SpitterEnemy -- gravity (LOB_GRAVITY) pulls it into a
    parabola instead of Projectile's flat line, so dodging means judging where it will
    land, not just stepping off its travel line. Lives in the same `projectiles` global
    as Projectile (Player.update() and update_and_draw_playing() already iterate that
    group generically), so no new group or call site is needed anywhere -- only its own
    update() differs. Despawns on hitting solid ground/a wall or after
    PROJECTILE_LIFETIME_MS, same lifetime discipline as Projectile."""

    def __init__(self, x, y, direction):
        super().__init__()
        w, h = 10, 10
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.circle(self.image, PROJECTILE_COLOR, (w // 2, h // 2), w // 2)
        self.rect = self.image.get_rect(center=(x, y))
        self.vel_x = LOB_SPEED_X * direction
        self.vel_y = LOB_SPEED_Y
        self.spawn_time = _now_ms()

    def update(self, platforms=()):
        self.vel_y += LOB_GRAVITY
        self.rect.x += int(self.vel_x)
        self.rect.y += int(self.vel_y)
        if _now_ms() - self.spawn_time > PROJECTILE_LIFETIME_MS:
            self.kill()
            return
        for p in platforms:
            if self.rect.colliderect(p.rect):
                self.kill()
                return


class TurretEnemy(pygame.sprite.Sprite):
    """A stationary installation that periodically fires a Projectile horizontally at
    whichever side the player is currently on. Always telegraphs (a visible color flash)
    TURRET_TELEGRAPH_MS before firing, and the shot travels in a flat, fixed-height line
    -- both mean it's always dodgeable by moving away or jumping over the shot, never an
    instant unavoidable hit. Not stompable like other enemies (has no hits_remaining
    quirks either) -- landing on it or touching it from the side both just hurt you via
    the normal side-collision path in Player.update()."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = TURRET_COLOR
        self._img_idle = _make_facing_sprite(TS - 6, TS - 10, TURRET_COLOR, 1)
        flash_color = tuple(min(255, c + 90) for c in TURRET_COLOR)
        self._img_telegraph = _make_facing_sprite(TS - 6, TS - 10, flash_color, 1)
        self.image = self._img_idle
        self.rect = self.image.get_rect(topleft=(x + 3, y + 10))
        self.telegraphing = False
        self.next_fire = _now_ms() + random.randint(600, TURRET_FIRE_INTERVAL_MAX_MS)

    def update(self, platforms=(), player_rect=None):
        global projectiles
        now = _now_ms()
        self.telegraphing = now >= self.next_fire - TURRET_TELEGRAPH_MS
        self.image = self._img_telegraph if self.telegraphing else self._img_idle
        if now >= self.next_fire:
            direction = 1
            if player_rect is not None and player_rect.centerx < self.rect.centerx:
                direction = -1
            projectiles.add(Projectile(self.rect.centerx, self.rect.centery, direction))
            self.next_fire = now + random.randint(TURRET_FIRE_INTERVAL_MIN_MS, TURRET_FIRE_INTERVAL_MAX_MS)
            self.telegraphing = False


class SpitterEnemy(pygame.sprite.Sprite):
    """A stationary ground installation, like TurretEnemy, but lobs an arcing
    LobbedProjectile (gravity-affected) instead of a flat-line shot -- dodging means
    judging where the arc will land, a different read than ducking a turret's shot. Only
    fires while the player is within SPITTER_RANGE and roughly at or below vent height
    (it doesn't lob shots at someone high overhead), and always telegraphs
    TURRET_TELEGRAPH_MS before firing, same as TurretEnemy."""

    def __init__(self, x, y, speed=2):
        super().__init__()
        self.color = SPITTER_COLOR
        self._img_idle = _make_facing_sprite(TS - 6, TS - 10, SPITTER_COLOR, 1)
        flash_color = tuple(min(255, c + 90) for c in SPITTER_COLOR)
        self._img_telegraph = _make_facing_sprite(TS - 6, TS - 10, flash_color, 1)
        self.image = self._img_idle
        self.rect = self.image.get_rect(topleft=(x + 3, y + 10))
        self.telegraphing = False
        self.next_fire = _now_ms() + random.randint(600, SPITTER_FIRE_INTERVAL_MAX_MS)

    def update(self, platforms=(), player_rect=None):
        global projectiles
        now = _now_ms()
        in_range = (player_rect is not None
                    and abs(player_rect.centerx - self.rect.centerx) < SPITTER_RANGE
                    and player_rect.centery > self.rect.top - TS)
        self.telegraphing = in_range and now >= self.next_fire - TURRET_TELEGRAPH_MS
        self.image = self._img_telegraph if self.telegraphing else self._img_idle
        if in_range and now >= self.next_fire:
            direction = 1 if player_rect.centerx >= self.rect.centerx else -1
            projectiles.add(LobbedProjectile(self.rect.centerx, self.rect.top, direction))
            self.next_fire = now + random.randint(SPITTER_FIRE_INTERVAL_MIN_MS, SPITTER_FIRE_INTERVAL_MAX_MS)
            self.telegraphing = False


class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        size = TS - 16
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(self.image, COIN_COLOR, (size // 2, size // 2), size // 2)
        pygame.draw.circle(self.image, (255, 250, 210), (size // 2, size // 2), max(1, size // 2 - 3), 2)
        self.base_rect = self.image.get_rect(topleft=(x + 8, y + 8))
        self.rect = self.base_rect.copy()
        self.t = random.uniform(0, 6.28)
        self.value = COIN_VALUE
        self.color = COIN_COLOR


class RareCoin(Coin):
    """Worth more than a normal coin and visually distinct (cyan gem vs. gold circle).
    Only ever spawned by build_level(), at a chance controlled by the Lucky Charm
    upgrade -- see load_level()'s rare_coin_chance parameter."""

    def __init__(self, x, y):
        super().__init__(x, y)
        size = self.image.get_width()
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pts = [(size // 2, 0), (size, size // 3), (size * 3 // 4, size), (size // 4, size), (0, size // 3)]
        pygame.draw.polygon(self.image, RARE_COIN_COLOR, pts)
        pygame.draw.polygon(self.image, (255, 255, 255), pts, 2)
        self.value = RARE_COIN_VALUE
        self.color = RARE_COIN_COLOR

    def update(self):
        self.t += 0.12
        self.rect.x = self.base_rect.x
        self.rect.y = self.base_rect.y + int(3 * math.sin(self.t))


class Particle:
    """`shape` picks how draw() renders the fading dot: "circle" (default, used by every
    combat/pickup effect and the original 3 trail styles), "square" (Confetti trail), or
    "ring" (an unfilled outline -- Aurora/Void trails). Purely a rendering choice --
    physics (velocity/gravity/fade) is identical for every shape, so new trail visuals
    never need a new update() path, only a new draw() branch."""

    def __init__(self, x, y, color, speed=3, life=400, shape="circle"):
        self.x = x
        self.y = y
        self.vel = pygame.Vector2(random.uniform(-speed, speed), random.uniform(-speed * 1.5, -0.5))
        self.color = color
        self.spawn_time = _now_ms()
        self.life = life
        self.shape = shape

    def alive(self):
        return _now_ms() - self.spawn_time < self.life

    def update(self):
        self.x += self.vel.x
        self.y += self.vel.y
        self.vel.y += 0.25

    def draw(self, surf, camera):
        age = (_now_ms() - self.spawn_time) / self.life
        alpha = max(0, 255 - int(255 * age))
        size = max(1, int(4 * (1 - age)))
        s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        if self.shape == "square":
            pygame.draw.rect(s, (*self.color, alpha), (0, 0, size * 2, size * 2))
        elif self.shape == "ring":
            pygame.draw.circle(s, (*self.color, alpha), (size, size), size, width=max(1, size // 2))
        else:
            pygame.draw.circle(s, (*self.color, alpha), (size, size), size)
        screen_x = self.x - camera.x + camera.shake_ox - size
        screen_y = self.y + camera.shake_oy - size
        surf.blit(s, (screen_x, screen_y))


class Popup:
    """Little floating '+10' / '+50' text that drifts up and fades out."""

    def __init__(self, x, y, text, color=WHITE, life=700):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.spawn_time = _now_ms()
        self.life = life

    def alive(self):
        return _now_ms() - self.spawn_time < self.life

    def draw(self, surf, camera, font):
        age = (_now_ms() - self.spawn_time) / self.life
        y_off = -22 * age
        alpha = max(0, 255 - int(255 * age))
        txt = font.render(self.text, True, self.color)
        txt.set_alpha(alpha)
        screen_x = self.x - camera.x + camera.shake_ox - txt.get_width() // 2
        screen_y = self.y + camera.shake_oy + y_off
        surf.blit(txt, (screen_x, screen_y))


particles = []
popups = []


def spawn_particles(x, y, color, count=10, speed=3, shape="circle"):
    for _ in range(count):
        particles.append(Particle(x, y, color, speed, shape=shape))


TRAIL_PARTICLE_INTERVAL_MS = 55  # how often a trail particle spawns while moving
_last_trail_particle_time = 0


def maybe_spawn_trail_particles(player_rect, player_state):
    """Cosmetic-only: spawns one small particle behind the player using the equipped
    trail's style/colors, reusing the existing Particle/spawn_particles machinery instead
    of a separate rendering system. A no-op whenever there's no trail equipped, the trail
    is set to Invisible, or the player is dead -- gameplay bonuses (see
    _trail_bonus_value) are completely independent of this and keep working regardless."""
    global _last_trail_particle_time
    if not equipped_trail_key or not trail_visible or player_state == "dead":
        return
    trail = TRAILS_BY_KEY.get(equipped_trail_key)
    if not trail:
        return
    now = _now_ms()
    if now - _last_trail_particle_time < TRAIL_PARTICLE_INTERVAL_MS:
        return
    _last_trail_particle_time = now

    style = trail["style"]
    if style == "rainbow":
        hue = (now / 3000) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))
    else:
        color = random.choice(trail["colors"])

    # "spark"/"square"/"comet" spawn 2 particles per tick (a denser burst); "comet" also
    # renders both as plain circles -- its distinct look comes from spawning a fresh
    # bright "core" every tick that overlaps the still-fading previous one, reading as a
    # streaking tail rather than individual dots. "ring" stays a single particle since an
    # outline shape already reads as more visually busy than a filled dot.
    count = 2 if style in ("spark", "square", "comet") else 1
    shape = {"square": "square", "ring": "ring"}.get(style, "circle")
    spawn_particles(player_rect.centerx, player_rect.bottom - 4, color, count=count, speed=1.6, shape=shape)


def render_text_fit(font, text, color, max_width):
    """Render `text` with `font`, shrinking the whole rendered surface down (preserving
    aspect ratio, same smoothscale trick draw_glow_text() already uses for its blur) if
    it would otherwise be wider than `max_width`. A no-op plain render whenever it
    already fits. This is the safety net behind every shop/trail row description: hand-
    picking strings that are "short enough" is exactly how those rows silently overflowed
    past their status label (or off the row entirely) before, so row-drawing code should
    never blit raw font.render() output for user-facing copy without going through this."""
    surf = font.render(text, True, color)
    if surf.get_width() <= max_width:
        return surf
    scale = max_width / surf.get_width()
    new_size = (max_width, max(1, int(surf.get_height() * scale)))
    return pygame.transform.smoothscale(surf, new_size)


def draw_hud_text(surf, text, pos, pad=4, font=None, color=WHITE):
    """Blit HUD text on a translucent dark backing so it stays readable over any
    background (white text on a white cloud is otherwise invisible). `text` may be a
    plain string (rendered internally with `font`/`color`) or an already-rendered
    Surface for callers that need to measure it before choosing where to place it.
    Either way, the rendered Surface is returned so the caller can query its size."""
    text_surf = (font or small_font).render(text, True, color) if isinstance(text, str) else text
    bg = pygame.Surface((text_surf.get_width() + pad * 2, text_surf.get_height() + pad * 2), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    surf.blit(bg, (pos[0] - pad, pos[1] - pad))
    surf.blit(text_surf, pos)
    return text_surf


# ---------------- upgrade-scaled reward helpers ----------------
# Small, capped percentage bonuses (never a flat unlimited multiplier) so buying these
# upgrades meaningfully helps the economy without letting it run away -- each is capped
# by its own UPGRADE_MAX_LEVELS entry (coin_bonus/stomp_bonus/score_multiplier all top
# out well under +50% total). Kept as free functions (not Player methods) since they only
# need the `upgrades` global, and buy_upgrade()/Player.update() both use them.
def _effective_coin_value(base_value):
    bonus = COIN_BONUS_PER_LEVEL * upgrades["coin_bonus"] + _trail_bonus_value("coin_bonus")
    return max(1, round(base_value * (1 + bonus)))


def _effective_stomp_reward():
    bonus = STOMP_BONUS_PER_LEVEL * upgrades["stomp_bonus"] + _trail_bonus_value("stomp_bonus")
    return max(1, round(STOMP_REWARD * (1 + bonus)))


def _effective_score_gain(base_value):
    # Trophy Polish (score_multiplier) only inflates the Score readout, never the
    # spendable wallet or the lifetime stat_coins counter -- see Player.update(). The
    # "score_mult" trail bonus (Rainbow, Aurora) stacks the same way.
    bonus = SCORE_MULT_PER_LEVEL * upgrades["score_multiplier"] + _trail_bonus_value("score_mult")
    return max(1, round(base_value * (1 + bonus)))


def _effective_glide_max():
    """Feather Fall level 1 just unlocks gliding at the base GLIDE_MAX_FRAMES; level 2
    ("Aerodynamics") adds extra stamina on top. A function (not a flat constant) since
    both respawn() and Player.update()'s landing-refill both need the current value."""
    if upgrades["glide"] <= 1:
        return GLIDE_MAX_FRAMES
    return GLIDE_MAX_FRAMES + GLIDE_EXTRA_FRAMES_PER_LEVEL * (upgrades["glide"] - 1)


def _trail_bonus_value(bonus_key):
    """The equipped trail's small gameplay bonus, if it has one of type `bonus_key` --
    0.0 otherwise. Deliberately ignores trail_visible: a trail's bonus (see the TRAILS
    table) stays active even while its visual effect is hidden, only equipping/unequipping
    the trail itself turns the bonus on or off."""
    if not equipped_trail_key:
        return 0.0
    trail = TRAILS_BY_KEY.get(equipped_trail_key)
    if trail and trail["bonus"] and trail["bonus"][0] == bonus_key:
        return trail["bonus"][1]
    return 0.0


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.rect = pygame.Rect(x, y, 30, 44)  # collision box stays fixed; art is drawn on top of it
        # Sub-pixel horizontal position -- see update() for why this exists. update()
        # overwrites self.rect.x from this every frame, so any code that moves the
        # player by setting self.rect.x/left/right/topleft directly (outside of
        # update()'s own collision resolution) MUST also set self.pos_x to match, or
        # the next frame's update() will silently snap rect.x back to the stale value.
        # respawn() already does this; that's the only other place the player is moved.
        self.pos_x = float(x)
        self.facing = 1
        self.run_frame_index = 0
        self.last_anim_tick = 0
        self.die_frame_index = 0
        self.image = HERO_IDLE_RIGHT
        self.start_pos = (x, y)
        self.checkpoint_pos = None  # set by touching a Checkpoint once the upgrade is owned
        self.vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.last_ground_time = 0
        self.jump_buffer_time = -99999
        self.state = "alive"
        self.death_time = 0
        self.air_jumps_remaining = 1
        self.shield_max = 0
        self.shield_charges = 0
        self.hearts_max = 0
        self.hearts = 0
        self.invincible_until = 0
        self.glide_frames_left = GLIDE_MAX_FRAMES
        self.dash_ready_time = 0
        self.dashing_until = 0
        self.dash_facing = 1  # direction locked in by try_dash(); see update()
        self.stomp_combo = 0       # consecutive aerial stomps without touching ground (Combo Momentum)
        self.second_wind_used = False  # Second Wind: at most one save per life
        self._ice_vel = 0.0        # blended horizontal velocity while standing on an IceFloor tile

    def _update_animation(self):
        now = _now_ms()
        if self.state == "dead":
            elapsed = now - self.death_time
            self.die_frame_index = min(len(HERO_DIE_FRAMES) - 1, elapsed // DIE_ANIM_INTERVAL)
            frame = HERO_DIE_FRAMES[self.die_frame_index]
            if self.facing == 1:
                frame = pygame.transform.flip(frame, True, False)
        elif not self.on_ground:
            frame = HERO_JUMP_LEFT if self.facing == -1 else HERO_JUMP_RIGHT
        elif self.vel.x != 0:
            if now - self.last_anim_tick > RUN_ANIM_INTERVAL:
                self.last_anim_tick = now
                self.run_frame_index = (self.run_frame_index + 1) % len(HERO_RUN_LEFT)
            frame = (HERO_RUN_LEFT if self.facing == -1 else HERO_RUN_RIGHT)[self.run_frame_index]
        else:
            frame = HERO_IDLE_LEFT if self.facing == -1 else HERO_IDLE_RIGHT

        self.image = frame

    def try_jump(self):
        self.jump_buffer_time = _now_ms()

    def try_dash(self):
        """Shift key. No-op if Dash hasn't been bought or the cooldown hasn't elapsed --
        callers don't need to check either condition themselves."""
        if upgrades["dash"] <= 0:
            return
        now = _now_ms()
        if now < self.dash_ready_time:
            return
        self.dashing_until = now + DASH_DURATION_MS
        # Lock in direction now, at the moment of activation -- update() must use this,
        # not the live self.facing, or holding the opposite key mid-dash reverses it.
        self.dash_facing = self.facing
        cooldown = max(DASH_COOLDOWN_MIN_MS, DASH_COOLDOWN_MS - DASH_COOLDOWN_PER_LEVEL * upgrades["dash_cooldown"])
        self.dash_ready_time = now + cooldown
        snd_dash.play()
        spawn_particles(self.rect.centerx, self.rect.centery, WHITE, count=8, speed=4)

    def kill_player(self):
        global stat_deaths
        if self.state != "dead":
            self.state = "dead"
            self.death_time = _now_ms()
            self.vel = pygame.Vector2(0, 0)
            stat_deaths += 1
            mark_dirty()
            snd_death.play()
            camera.shake(8, 200)

    def take_hit(self):
        """Spike/enemy damage goes through here instead of straight to kill_player() so
        Shield, Vitality, and Second Wind can absorb it in order: shield charges go first
        (they recharge every life/level so they're the "cheap" layer), then extra hearts,
        then a one-time Second Wind save, then death. Falling into the void bypasses this
        entirely -- these upgrades protect against hazards/enemies, not bottomless pits."""
        now = _now_ms()
        if now < self.invincible_until:
            return
        if self.shield_charges > 0:
            self.shield_charges -= 1
            self.invincible_until = now + HIT_INVINCIBILITY_MS
            snd_shield.play()
            spawn_particles(self.rect.centerx, self.rect.centery, (120, 200, 255), count=10)
        elif self.hearts > 0:
            self.hearts -= 1
            self.invincible_until = now + HIT_INVINCIBILITY_MS
            snd_shield.play()
            spawn_particles(self.rect.centerx, self.rect.centery, (255, 90, 90), count=10)
        elif upgrades["second_wind"] > 0 and not self.second_wind_used:
            # Deliberately placed here rather than in kill_player() so it only rescues a
            # hazard/enemy hit, never a fall into a bottomless pit -- see kill_player()'s
            # only other call site, matching every other survival upgrade's scope.
            self.second_wind_used = True
            self.invincible_until = now + HIT_INVINCIBILITY_MS
            snd_shield.play()
            spawn_particles(self.rect.centerx, self.rect.centery, (255, 215, 90), count=14, speed=4)
            popups.append(Popup(self.rect.centerx, self.rect.top, "Second Wind!", (255, 215, 90)))
            mark_dirty()
        else:
            self.kill_player()

    def respawn(self):
        self.rect.topleft = self.checkpoint_pos or self.start_pos
        # Hard teleport, not incremental movement -- reset the render-interpolation
        # snapshot (see _snapshot_render_positions()) too, or draw_playing_frame() would
        # blend from the pre-death position across to this one and visibly slide the
        # player across the level for a frame instead of a clean cut.
        self._prev_topleft = self.rect.topleft
        self.pos_x = float(self.rect.x)
        self.vel = pygame.Vector2(0, 0)
        self.state = "alive"
        self.air_jumps_remaining = 1 + upgrades["extra_jump"]
        self.shield_max = upgrades["shield"]
        self.shield_charges = self.shield_max
        self.hearts_max = upgrades["vitality"]
        self.hearts = self.hearts_max
        # A brief grace period, not 0 -- respawning right next to a hazard or an
        # already-alert enemy (a ChargerEnemy in particular, since it locks on and
        # closes distance fast) must never be able to chain straight into another death
        # before the player has even had a chance to react and move away.
        self.invincible_until = _now_ms() + RESPAWN_INVINCIBILITY_MS
        self.glide_frames_left = _effective_glide_max()
        self.dash_ready_time = 0
        self.dashing_until = 0
        self.stomp_combo = 0
        self.second_wind_used = False
        self._ice_vel = 0.0

    def _resolve(self, platforms, axis):
        for p in platforms:
            if self.rect.colliderect(p.rect):
                if axis == "x":
                    if self.vel.x > 0:
                        self.rect.right = p.rect.left
                    elif self.vel.x < 0:
                        self.rect.left = p.rect.right
                else:
                    if self.vel.y > 0:
                        self.rect.bottom = p.rect.top
                        # keep a tiny resting downward velocity (not exactly 0) so the
                        # rect always overlaps the ground by 1px next frame -- otherwise
                        # int(vel.y) truncates small gravity steps to 0, the rect never
                        # moves, colliderect sees two merely-touching rects as not
                        # colliding, and on_ground flickers True/False every frame.
                        self.vel.y = 1
                        self.on_ground = True
                    elif self.vel.y < 0:
                        self.rect.top = p.rect.bottom
                        self.vel.y = 0

    def _resolve_one_way(self, one_way_platforms, prev_bottom):
        """Only collide with a one-way platform if the player was already at or above
        its top surface before this frame's move AND is now moving downward -- i.e.
        landing on top. Approaching from below or the side never collides at all."""
        if self.vel.y <= 0:
            return
        for p in one_way_platforms:
            if prev_bottom <= p.rect.top and self.rect.colliderect(p.rect):
                self.rect.bottom = p.rect.top
                self.vel.y = 1
                self.on_ground = True

    def _handle_stomp(self, e, enemies):
        """Resolve a successful stomp on enemy `e`: kill it outright, or -- for a
        ShieldedEnemy (or anything else with hits_remaining > 1) -- crack its shield and
        pay a smaller consolation reward instead of killing it. A SplitterEnemy (or
        anything else exposing spawn_shards()) bursts into fragments added to `enemies`
        right before it dies, and `reward_scale` (defaulting to 1.0) lets a kill pay out
        more or less than a full STOMP_REWARD -- both SplitterEnemy and its ShardEnemy
        fragments use this to keep the total payout across a whole encounter balanced.
        Combo Momentum then scales the reward further based on self.stomp_combo, the
        number of consecutive stomps landed without touching ground in between. Bounces
        the player and updates score/wallet/stats. Called from update()'s enemy-collision
        loop once it's already determined this collision was a stomp, not a side hit."""
        global score, wallet, stat_coins, stat_stomps, stat_max_combo
        hits_remaining = getattr(e, "hits_remaining", 1)
        if hits_remaining > 1:
            e.hits_remaining = hits_remaining - 1
            reward = max(1, _effective_stomp_reward() // 4)
            snd_shield.play()
        else:
            spawn_shards = getattr(e, "spawn_shards", None)
            if spawn_shards is not None:
                for shard in spawn_shards():
                    enemies.add(shard)
            reward = max(1, round(_effective_stomp_reward() * getattr(e, "reward_scale", 1.0)))
            e.kill()
            snd_stomp.play()
        self.vel.y = JUMP_STRENGTH * 0.6
        self.stomp_combo += 1
        combo_bonus = min(COMBO_MOMENTUM_CAP,
                           COMBO_MOMENTUM_PER_LEVEL * upgrades["combo_momentum"] * (self.stomp_combo - 1))
        reward = max(1, round(reward * (1 + combo_bonus)))
        stat_max_combo = max(stat_max_combo, self.stomp_combo)
        score += _effective_score_gain(reward)
        wallet += reward
        stat_coins += reward
        stat_stomps += 1
        mark_dirty()
        spawn_particles(e.rect.centerx, e.rect.centery, getattr(e, "color", ENEMY_COLOR))
        popups.append(Popup(e.rect.centerx, e.rect.top, f"+{reward}"))
        camera.shake(3, 90)

    def update(self, platforms, spikes, enemies, coins, one_way_platforms=(), checkpoints=(), wind_zones=()):
        global score, wallet, stat_coins  # stat_stomps is now updated inside _handle_stomp()
        if self.state == "dead":
            self._update_animation()
            if _now_ms() - self.death_time > 600:
                self.respawn()
            return

        move_speed = (MOVE_SPEED + SPEED_PER_LEVEL * upgrades["speed"]) * (1 + _trail_bonus_value("speed"))
        jump_strength = (JUMP_STRENGTH - JUMP_PER_LEVEL * upgrades["jump"]) * (1 + _trail_bonus_value("jump"))
        effective_coyote = COYOTE_TIME + COYOTE_BONUS_PER_LEVEL * upgrades["coyote"]
        effective_buffer = JUMP_BUFFER + JUMP_BUFFER_BONUS_PER_LEVEL * upgrades["jump_buffer"]

        keys = pygame.key.get_pressed()
        # pygame.key.get_pressed() only reflects real keyboard state -- it has no idea
        # about the touch overlay's held buttons, so those are checked separately
        # everywhere a *continuous* key hold matters (move/jump-hold/glide below). Edge-
        # triggered actions (Jump's buffer, Dash) don't need this: their touch buttons
        # call player.try_jump()/try_dash() directly -- see TouchControls.
        touch_keys = touch_controls.held_keys()
        self.vel.x = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a] or pygame.K_LEFT in touch_keys:
            self.vel.x = -move_speed
            self.facing = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d] or pygame.K_RIGHT in touch_keys:
            self.vel.x = move_speed
            self.facing = 1

        now = _now_ms()
        if now < self.dashing_until:
            # Dash overrides normal horizontal input for its short duration, using the
            # direction locked in by try_dash() (self.dash_facing) rather than the live
            # self.facing -- otherwise holding the opposite arrow key mid-dash would
            # reverse it instantly. Vertical movement (gravity/jumping) is untouched, so
            # dashing never interrupts a jump. The Lightning trail's bonus amplifies dash
            # speed (its "acceleration" flavor) the same way Purple Magic amplifies the
            # Magnet upgrade -- a trail sweetener on top of an owned mechanic, not a
            # standalone one, since this game has no separate acceleration/momentum model.
            self.vel.x = DASH_SPEED * (1 + _trail_bonus_value("dash_speed")) * self.dash_facing
        else:
            # Icy floor: standing on an IceFloor tile blends horizontal velocity toward
            # the target speed (ICE_ACCEL per frame) instead of snapping to it instantly,
            # so changing direction -- or stopping -- takes a beat, a genuine "slippery"
            # feel. isinstance() against the same `platforms` group every other collision
            # check already uses, so this needed no new sprite group threaded through
            # build_level()/load_level(). Dash (above) always overrides it outright, and
            # stepping off the ice resyncs _ice_vel immediately so there's no leftover
            # "stickiness" once you're back on normal ground.
            #
            # A plain self.rect.colliderect(p.rect) would almost never be true here: by
            # the end of the previous frame, _resolve() already snapped rect.bottom
            # exactly flush with the platform's top (touching, not overlapping -- pygame
            # Rect collision needs actual overlap). So this uses the same small
            # just-below-the-feet probe patrol_at_ledge() uses for enemies, instead of
            # the full rects.
            foot_probe = pygame.Rect(self.rect.centerx - 2, self.rect.bottom, 4, 3)
            on_ice = self.on_ground and any(isinstance(p, IceFloor) and foot_probe.colliderect(p.rect)
                                             for p in platforms)
            if on_ice:
                self._ice_vel += (self.vel.x - self._ice_vel) * ICE_ACCEL
                self.vel.x = self._ice_vel
            else:
                self._ice_vel = self.vel.x

        can_coyote = now - self.last_ground_time <= effective_coyote
        buffered = now - self.jump_buffer_time <= effective_buffer
        if buffered and (self.on_ground or can_coyote):
            self.vel.y = jump_strength
            self.on_ground = False
            self.jump_buffer_time = -99999
            self.last_ground_time = -99999
            self.air_jumps_remaining = 1 + upgrades["extra_jump"]
            snd_jump.play()
        elif buffered and self.air_jumps_remaining > 0:
            self.vel.y = jump_strength * 0.85
            self.jump_buffer_time = -99999
            self.air_jumps_remaining -= 1
            snd_double_jump.play()
            spawn_particles(self.rect.centerx, self.rect.bottom, WHITE, count=6, speed=2)

        if (not (keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w] or pygame.K_SPACE in touch_keys)
                and self.vel.y < -6):
            self.vel.y = -6

        self.vel.y = min(self.vel.y + GRAVITY, MAX_FALL_SPEED)

        if (upgrades["glide"] > 0 and not self.on_ground and self.vel.y > 0
                and (keys[pygame.K_DOWN] or pygame.K_DOWN in touch_keys) and self.glide_frames_left > 0):
            self.vel.y = min(self.vel.y, GLIDE_FALL_SPEED)
            self.glide_frames_left -= 1

        for wz in wind_zones:
            if self.rect.colliderect(wz.rect):
                # WIND_ZONE_PUSH is deliberately small relative to move_speed -- holding
                # the opposing direction always beats it, so it's a nudge, not a shove.
                self.vel.x += wz.push
                break

        # Horizontal position accumulates in a float (self.pos_x) instead of truncating
        # self.rect.x by int(vel.x) every single frame -- with the original whole-number
        # MOVE_SPEED this behaves identically to before, but it's what makes the Speed
        # Boots upgrade's fractional per-level bonus (SPEED_PER_LEVEL=0.4) actually show
        # up frame-to-frame instead of some purchased levels being silently truncated away.
        self.pos_x += self.vel.x
        self.rect.x = round(self.pos_x)
        pre_resolve_x = self.rect.x
        self._resolve(platforms, "x")
        if self.rect.x != pre_resolve_x:
            # A wall collision snapped rect.x to an integer boundary -- resync so next
            # frame's accumulation starts from the wall, not a stale pre-collision value.
            # Only do this when a collision actually happened: resyncing unconditionally
            # every frame would round pos_x's fractional part away every single frame,
            # silently defeating the whole point of accumulating it as a float.
            self.pos_x = self.rect.x

        was_on_ground = self.on_ground
        falling_speed = self.vel.y
        prev_bottom = self.rect.bottom
        self.rect.y += int(self.vel.y)
        self.on_ground = False
        self._resolve(platforms, "y")
        self._resolve_one_way(one_way_platforms, prev_bottom)

        if self.on_ground:
            self.last_ground_time = now
            self.glide_frames_left = _effective_glide_max()
            self.stomp_combo = 0  # touching ground always ends an aerial stomp chain
            if not was_on_ground and falling_speed > 3:
                snd_land.play()
                if falling_speed > 12:
                    camera.shake(4, 120)

        if any(self.rect.colliderect(s.hitbox) for s in spikes):
            self.take_hit()

        for shot in list(projectiles):
            if self.rect.colliderect(shot.rect):
                shot.kill()
                self.take_hit()
                break

        for e in list(enemies):
            if self.rect.colliderect(e.rect):
                # Stomp detection uses where the player's feet were *before* this frame's
                # move (like the one-way-platform landing check above), not an arbitrary
                # overlap-distance threshold -- a side collision leaves prev_bottom well
                # below the enemy's top, so it can never be misread as a stomp no matter
                # how the two rects happen to overlap this particular frame.
                if falling_speed > 0 and prev_bottom <= e.rect.top + STOMP_TOLERANCE:
                    self._handle_stomp(e, enemies)
                else:
                    self.take_hit()

        magnet_level = upgrades["magnet"]
        if magnet_level > 0:
            # Purple Magic's coin_radius bonus amplifies an existing Coin Magnet range
            # rather than granting a pull from nothing -- it's a trail sweetener, not a
            # substitute for the upgrade.
            magnet_range = MAGNET_RANGES[magnet_level] * (1 + _trail_bonus_value("coin_radius"))
            for c in coins:
                dx = self.rect.centerx - c.base_rect.centerx
                dy = self.rect.centery - c.base_rect.centery
                dist = max(1.0, math.hypot(dx, dy))
                if dist < magnet_range:
                    c.base_rect.x += int(MAGNET_PULL_SPEED * dx / dist)
                    c.base_rect.y += int(MAGNET_PULL_SPEED * dy / dist)

        for c in list(coins):
            if self.rect.colliderect(c.rect):
                c.kill()
                snd_coin.play()
                coin_value = _effective_coin_value(c.value)
                score += _effective_score_gain(coin_value)
                wallet += coin_value
                stat_coins += coin_value
                mark_dirty()
                spawn_particles(c.rect.centerx, c.rect.centery, getattr(c, "color", COIN_COLOR), count=6)
                popups.append(Popup(c.rect.centerx, c.rect.top, f"+{coin_value}", COIN_COLOR))

        if upgrades["checkpoint"] > 0:
            for cp in checkpoints:
                if self.rect.colliderect(cp.trigger_rect):
                    self.checkpoint_pos = cp.rect.topleft
                    if not cp.activated:
                        # One-time polish the first time this particular flag becomes the
                        # active respawn point -- harmless to call again since activate()
                        # itself is idempotent, but gating the fanfare here too keeps it
                        # from re-firing every single frame the player stands in the column.
                        cp.activate()
                        snd_checkpoint.play()
                        spawn_particles(cp.rect.centerx, cp.rect.centery, CHECKPOINT_COLOR, count=10)
                        popups.append(Popup(cp.rect.centerx, cp.rect.top, "Checkpoint!", CHECKPOINT_COLOR))

        if self.rect.top > LEVEL_HEIGHT + 300:
            self.kill_player()

        self._update_animation()


class Camera:
    def __init__(self, level_width):
        self.x = 0.0
        self.level_width = level_width
        self.shake_until = 0
        self.shake_mag = 0
        self.shake_ox = 0
        self.shake_oy = 0

    def shake(self, magnitude, duration_ms):
        if not screen_shake_enabled:
            return
        self.shake_mag = magnitude
        self.shake_until = _now_ms() + duration_ms

    def update(self, target_rect):
        desired = target_rect.centerx - SCREEN_W / 2
        desired = max(0, min(desired, self.level_width - SCREEN_W))
        self.x += (desired - self.x) * 0.15
        if _now_ms() < self.shake_until:
            self.shake_ox = random.randint(-self.shake_mag, self.shake_mag)
            self.shake_oy = random.randint(-self.shake_mag, self.shake_mag)
        else:
            self.shake_ox = 0
            self.shake_oy = 0

    def apply(self, rect):
        return rect.move(-int(self.x) + self.shake_ox, self.shake_oy)


class Background:
    def __init__(self, level_width, palette=None):
        palette = palette or BIOME_PALETTES[0]
        self.hill_color = palette["hill"]
        self.cloud_color = palette["cloud"]
        rng = random.Random(1)
        self.hills = [(rng.randint(0, level_width), rng.randint(80, 140)) for _ in range(10)]
        self.clouds = [(rng.randint(0, level_width), rng.randint(20, 120), rng.randint(30, 60)) for _ in range(14)]
        self.sky = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            t = y / SCREEN_H
            color = tuple(int(a + (b - a) * t) for a, b in zip(palette["sky_top"], palette["sky_bottom"]))
            pygame.draw.line(self.sky, color, (0, y), (SCREEN_W, y))

    def draw(self, surf, cam_x):
        surf.blit(self.sky, (0, 0))
        for hx, hr in self.hills:
            sx = hx - cam_x * 0.3
            if -hr < sx < SCREEN_W + hr:
                pygame.draw.circle(surf, self.hill_color, (int(sx), SCREEN_H - 20), hr)
        for cx, cy, cr in self.clouds:
            sx = cx - cam_x * 0.15
            if -cr < sx < SCREEN_W + cr:
                pygame.draw.circle(surf, self.cloud_color, (int(sx), cy), cr)
                pygame.draw.circle(surf, self.cloud_color, (int(sx + cr * 0.7), cy + 6), int(cr * 0.7))


# ---------------- start-screen menu (neon toggle switch + slider) ----------------
class Toggle:
    def __init__(self, x, y, label, value=True):
        self.rect = pygame.Rect(x, y, 64, 30)
        self.label = label
        self.value = value

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.value = not self.value

    def draw(self, surf, font):
        on_color = (0, 255, 200)
        off_color = (70, 70, 85)
        pygame.draw.rect(surf, on_color if self.value else off_color, self.rect, border_radius=self.rect.height // 2)
        pygame.draw.rect(surf, (210, 255, 245), self.rect, 2, border_radius=self.rect.height // 2)
        knob_x = self.rect.right - 17 if self.value else self.rect.left + 17
        pygame.draw.circle(surf, (15, 10, 30), (knob_x, self.rect.centery), 11)
        pygame.draw.circle(surf, WHITE, (knob_x, self.rect.centery), 11, 2)
        state_text = "ON" if self.value else "OFF"
        label_surf = font.render(f"{self.label}: {state_text}", True, (0, 255, 220))
        surf.blit(label_surf, label_surf.get_rect(midbottom=(self.rect.centerx, self.rect.y - 8)))


class Slider:
    def __init__(self, x, y, w, label, value=0.5):
        self.rect = pygame.Rect(x, y, w, 10)
        self.label = label
        self.value = max(0.0, min(1.0, value))
        self.dragging = False

    def _handle_x(self):
        return self.rect.x + int(self.value * self.rect.w)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit_zone = self.rect.inflate(20, 30)
            if hit_zone.collidepoint(event.pos):
                self.dragging = True
                self._set_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_mouse(event.pos[0])

    def _set_from_mouse(self, mouse_x):
        self.value = max(0.0, min(1.0, (mouse_x - self.rect.x) / self.rect.w))

    def draw(self, surf, font):
        pygame.draw.rect(surf, (70, 70, 85), self.rect, border_radius=5)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, int(self.value * self.rect.w), self.rect.h)
        if fill_rect.width > 0:
            pygame.draw.rect(surf, (255, 0, 170), fill_rect, border_radius=5)
        hx = self._handle_x()
        pygame.draw.circle(surf, (15, 10, 30), (hx, self.rect.centery), 10)
        pygame.draw.circle(surf, (255, 60, 190), (hx, self.rect.centery), 10, 2)
        label_surf = font.render(f"{self.label}: {int(self.value * 100)}%", True, (255, 90, 210))
        surf.blit(label_surf, label_surf.get_rect(midbottom=(self.rect.centerx, self.rect.y - 8)))


class FpsCapControl:
    """The Settings screen's FPS Cap row: -/+ buttons step a remembered numeric value by
    FPS_CAP_STEP, clicking the number opens type-to-enter (any positive integer, not just
    a preset), and UNLIMITED removes the cap entirely -- see FPS_CAP_* near SIM_HZ.
    `value` (what the rest of the game reads) is 0 for Unlimited, else a clamped int.
    Stepping/typing while Unlimited resumes from the last numeric value instead of some
    arbitrary point, so flipping back out of Unlimited never surprises the player."""

    def __init__(self, x, y, value=FPS_CAP_DEFAULT):
        self.value = FPS_CAP_UNLIMITED if value <= 0 else _clamp(value, FPS_CAP_MIN, FPS_CAP_MAX)
        self._last_numeric = self.value or FPS_CAP_DEFAULT
        self.editing = False
        self.edit_text = ""
        self.minus_rect = pygame.Rect(x, y, 30, 30)
        self.value_rect = pygame.Rect(x + 36, y, 100, 30)
        self.plus_rect = pygame.Rect(x + 142, y, 30, 30)
        self.unlimited_rect = pygame.Rect(x + 182, y, 110, 30)

    def _set_numeric(self, new_value):
        self._last_numeric = _clamp(new_value, FPS_CAP_MIN, FPS_CAP_MAX)
        self.value = self._last_numeric

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.minus_rect.collidepoint(event.pos):
                self.editing = False
                self._set_numeric(self._last_numeric - FPS_CAP_STEP)
            elif self.plus_rect.collidepoint(event.pos):
                self.editing = False
                self._set_numeric(self._last_numeric + FPS_CAP_STEP)
            elif self.unlimited_rect.collidepoint(event.pos):
                self.editing = False
                self.value = FPS_CAP_UNLIMITED
            elif self.value_rect.collidepoint(event.pos):
                self.editing = True
                self.edit_text = ""
        elif event.type == pygame.KEYDOWN and self.editing:
            if event.key == pygame.K_RETURN:
                if self.edit_text:
                    self._set_numeric(int(self.edit_text))
                self.editing = False
                self.edit_text = ""
            elif event.key == pygame.K_ESCAPE:
                self.editing = False
                self.edit_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.edit_text = self.edit_text[:-1]
            elif event.unicode.isdigit() and len(self.edit_text) < 4:
                self.edit_text += event.unicode

    def draw(self, surf, font):
        label = f"FPS Cap: {'Unlimited' if self.value == 0 else self.value}"
        label_surf = font.render(label, True, (120, 220, 255))
        surf.blit(label_surf, label_surf.get_rect(midbottom=(
            (self.minus_rect.x + self.unlimited_rect.right) // 2, self.minus_rect.y - 8)))

        def button(rect, text, active=False):
            pygame.draw.rect(surf, (0, 200, 255) if active else (60, 60, 78), rect, border_radius=6)
            pygame.draw.rect(surf, (170, 235, 255), rect, 2, border_radius=6)
            t = font.render(text, True, (15, 10, 30) if active else (220, 240, 255))
            surf.blit(t, t.get_rect(center=rect.center))

        button(self.minus_rect, "-")
        button(self.plus_rect, "+")
        button(self.unlimited_rect, "UNLIMITED", active=self.value == 0)

        pygame.draw.rect(surf, (30, 30, 42), self.value_rect, border_radius=6)
        border = (255, 230, 90) if self.editing else (170, 235, 255)
        pygame.draw.rect(surf, border, self.value_rect, 2, border_radius=6)
        if self.editing:
            shown = self.edit_text + "_" if self.edit_text else "_"
        else:
            shown = "Unlimited" if self.value == 0 else str(self.value)
        val_surf = font.render(shown, True, (255, 255, 255))
        surf.blit(val_surf, val_surf.get_rect(center=self.value_rect.center))


# ---------------- touch controls ----------------
# Virtual on-screen controls for touchscreens: Android (via python-for-android/buildozer)
# and "hybrid" devices that have both a touchscreen and a keyboard/mouse (a Windows
# tablet, a Steam Deck in touchscreen mode, a touchscreen laptop). Movement is binary --
# Player.update() only ever sets vel.x to -move_speed/+move_speed/0, never anything
# analog -- so plain press-buttons are the right fit here, not a virtual joystick.
#
# Desktop players who never touch the screen never see any of this: the overlay only
# becomes visible once a real pygame.FINGERDOWN/UP/MOTION event has actually been seen
# (or immediately, on Android, which has no keyboard to fall back to), and it fades back
# out after a few seconds of touch inactivity so it doesn't clutter a keyboard/mouse
# session on a hybrid device. Button hit-testing stays live even while fully faded out,
# so the very first tap after hiding both registers as input AND fades the overlay back
# in -- see TouchControls.on_finger_down().
def _is_android():
    """True only under python-for-android/buildozer -- the only realistic way this pygame
    game reaches an Android device. Checked once (TouchControls.__init__); a process
    never stops being Android mid-session."""
    return (
        sys.platform == "android"
        or hasattr(sys, "getandroidapilevel")
        or "ANDROID_ARGUMENT" in os.environ
        or "ANDROID_PRIVATE" in os.environ
    )


def _finger_event_logical_pos(event):
    """FINGERDOWN/UP/MOTION report x/y as 0..1 fractions of the real OS window, unlike
    mouse events (which pygame.SCALED already remaps for us) -- this replicates SCALED's
    own letterbox math (uniform scale-to-fit, centered) to recover logical 800x440
    coordinates from them."""
    win_w, win_h = pygame.display.get_window_size()
    scale = min(win_w / SCREEN_W, win_h / SCREEN_H) or 1.0
    draw_w, draw_h = SCREEN_W * scale, SCREEN_H * scale
    off_x, off_y = (win_w - draw_w) / 2, (win_h - draw_h) / 2
    return ((event.x * win_w - off_x) / scale, (event.y * win_h - off_y) / scale)


def _finger_event_logical_delta(event):
    """FINGERMOTION's dx/dy are normalized deltas (fractions of the real window size),
    using the same normalization as x/y above -- only the uniform scale factor applies to
    a delta, not the letterbox offset (which cancels out between two points)."""
    win_w, win_h = pygame.display.get_window_size()
    scale = min(win_w / SCREEN_W, win_h / SCREEN_H) or 1.0
    return (event.dx * win_w / scale, event.dy * win_h / scale)


TOUCH_SAFE_MARGIN = 6          # logical px kept clear of every screen edge -- a manual
                                # stand-in for an OS safe-area/notch inset, since pygame/SDL
                                # expose no such API in the general case
TOUCH_INACTIVITY_MS = 3500     # hybrid devices: start fading out this long after the last touch
TOUCH_FADE_MS = 220             # fade in/out duration
TOUCH_ANDROID_MIN_ALPHA = 0.35  # Android never fades below this -- it has no keyboard fallback
TOUCH_SIZE_MIN, TOUCH_SIZE_MAX = 0.8, 1.2  # Settings' Touch Size slider maps 0..1 onto this


def _draw_touch_icon(surf, kind, cx, cy, r, color):
    """Small hand-drawn icons for the touch overlay -- plain pygame.draw shapes, matching
    every other in-game icon (spikes, the checkpoint flag, coins, ...) instead of depending
    on whatever system font happens to be installed actually having e.g. a down-triangle
    glyph."""
    s = r * 0.5
    if kind == "left":
        pygame.draw.polygon(surf, color, [(cx + s * 0.5, cy - s), (cx - s * 0.7, cy), (cx + s * 0.5, cy + s)])
    elif kind == "right":
        pygame.draw.polygon(surf, color, [(cx - s * 0.5, cy - s), (cx + s * 0.7, cy), (cx - s * 0.5, cy + s)])
    elif kind == "up":
        pygame.draw.polygon(surf, color, [(cx, cy - s), (cx - s, cy + s * 0.7), (cx + s, cy + s * 0.7)])
    elif kind == "down":
        pygame.draw.polygon(surf, color, [(cx, cy + s), (cx - s, cy - s * 0.7), (cx + s, cy - s * 0.7)])
    elif kind == "dash":
        for dx in (-s * 0.55, s * 0.35):
            pygame.draw.polygon(surf, color, [(cx + dx - s * 0.3, cy - s * 0.6), (cx + dx + s * 0.4, cy),
                                               (cx + dx - s * 0.3, cy + s * 0.6)])
    elif kind == "pause":
        bar_w, bar_h, gap = max(2, round(s * 0.35)), round(s * 1.4), max(2, round(s * 0.3))
        pygame.draw.rect(surf, color, (cx - gap - bar_w, cy - bar_h // 2, bar_w, bar_h), border_radius=1)
        pygame.draw.rect(surf, color, (cx + gap, cy - bar_h // 2, bar_w, bar_h), border_radius=1)
    elif kind == "pause_toggle":
        # The one touch button that means two different things depending on state: a
        # Pause icon while playing, a Play/Resume triangle once paused -- see
        # TouchControls._buttons_for_state(), which is what keeps it on screen (and at
        # the same spot) across that transition instead of disappearing along with every
        # other gameplay button the moment the world freezes.
        _draw_touch_icon(surf, "right" if state == "paused" else "pause", cx, cy, r, color)


def _toggle_pause():
    """The Esc key and the touch overlay's Pause button both go through this, so the two
    input methods can never end up with a different idea of what pausing does."""
    global state
    if state == "playing":
        state = "paused"
    elif state == "paused":
        state = "playing"


class TouchButton:
    """One press-region of the touch overlay. Position is defined as an offset from
    whichever screen corner it belongs to (`home_side`), not an absolute point -- that's
    what lets TouchControls scale the whole cluster (Settings' Touch Size) or mirror it to
    the opposite side (Settings' Swap Sides) just by changing how the offset is resolved,
    in center_radius(), instead of storing a separate layout per setting combination.

    `keys` are pygame key constants this button acts as "held" for while any finger is on
    it (consumed by Player.update() for continuous actions -- move/glide/the jump-hold
    that controls variable jump height). `on_press` (if given) fires once per finger that
    lands on it, for edge-triggered actions (Jump/Dash/Pause) that go through the exact
    same functions/state transitions a keyboard press already triggers in handle_events(),
    instead of duplicating that logic here."""

    def __init__(self, home_side, dx, dy, radius, icon, keys=(), on_press=None,
                 color=(0, 255, 200), is_active=None):
        self.home_side = home_side  # "left", "right", or "top"
        self.dx, self.dy = dx, dy   # magnitude inward-from-edge / upward-from-bottom (or
                                     # downward-from-top for "top"), always >= 0
        self.radius = radius
        self.icon = icon
        self.keys = keys
        self.on_press = on_press
        self.color = color
        self.is_active = is_active or (lambda: True)
        self.fingers = set()  # finger_ids currently pressing this button

    @property
    def held(self):
        return bool(self.fingers)

    def center_radius(self, swap_sides, scale, margin):
        side = self.home_side
        if swap_sides and side in ("left", "right"):
            side = "right" if side == "left" else "left"
        if side == "left":
            cx, cy = margin + self.dx * scale, SCREEN_H - margin - self.dy * scale
        elif side == "right":
            cx, cy = SCREEN_W - margin - self.dx * scale, SCREEN_H - margin - self.dy * scale
        else:  # "top" -- fixed in place; Swap Sides only affects the move/action clusters
            cx, cy = SCREEN_W / 2 + self.dx * scale, margin + self.dy * scale
        return (cx, cy), self.radius * scale

    def contains(self, pos, swap_sides, scale, margin):
        (cx, cy), r = self.center_radius(swap_sides, scale, margin)
        return (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 <= r * r

    def draw(self, surf, swap_sides, scale, margin):
        (cx, cy), r = self.center_radius(swap_sides, scale, margin)
        cx, cy, r = round(cx), round(cy), round(r)
        pressed = self.held
        pygame.draw.circle(surf, (18, 16, 26, 165), (cx, cy), r)
        ring_color = tuple(min(255, c + 60) for c in self.color) if pressed else self.color
        pygame.draw.circle(surf, (*ring_color, 235), (cx, cy), r, 0 if pressed else 3)
        icon_color = (15, 12, 22) if pressed else ring_color
        _draw_touch_icon(surf, self.icon, cx, cy, r, icon_color)


class TouchControls:
    """Owns every piece of the touch-overlay feature: platform/touch detection, the fade
    timer, per-button finger tracking, and the user-facing Settings (opacity/size/side).
    One instance (`touch_controls`, created in init_game_state()) is read by
    Player.update() (held_keys()), fed events by handle_events() (on_finger_*()), advanced
    once a frame by game_loop() (update()), and drawn by draw_playing_frame() (draw())."""

    def __init__(self):
        self.is_android = _is_android()
        self.ever_touched = self.is_android  # Android shows its controls immediately
        self.last_touch_ms = _now_ms() if self.is_android else -10 ** 9
        self._alpha = 1.0 if self.is_android else 0.0
        self.opacity = 0.8       # Settings' Touch Opacity slider, 0..1
        self.size = 1.0          # Settings' Touch Size slider, mapped into TOUCH_SIZE_MIN..MAX
        self.swap_sides = False  # Settings' Swap Sides toggle

        self.btn_left = TouchButton("left", 56, 70, 36, "left", keys=(pygame.K_LEFT,),
                                     color=(0, 200, 255))
        self.btn_right = TouchButton("left", 138, 70, 36, "right", keys=(pygame.K_RIGHT,),
                                      color=(0, 200, 255))
        self.btn_jump = TouchButton("right", 56, 70, 42, "up", keys=(pygame.K_SPACE,),
                                     on_press=lambda: player.try_jump(), color=(0, 255, 200))
        self.btn_dash = TouchButton("right", 140, 104, 28, "dash",
                                     on_press=lambda: player.try_dash(), color=(120, 220, 255),
                                     is_active=lambda: upgrades["dash"] > 0)
        self.btn_glide = TouchButton("right", 140, 36, 26, "down", keys=(pygame.K_DOWN,),
                                      color=(170, 220, 140), is_active=lambda: upgrades["glide"] > 0)
        # Bigger than every other button (a mis-tap here is far more disruptive than one
        # on Left/Right) and it's the one button that stays reachable through "paused" too
        # -- see _buttons_for_state() and _draw_touch_icon()'s "pause_toggle" case. dy/
        # radius are tuned to clear the "PAUSED" title (title_font, centered at y=70,
        # its rendered rect top lands at y=48) with a few px to spare.
        self.btn_pause = TouchButton("top", 0, 16, 20, "pause_toggle", on_press=_toggle_pause,
                                      color=(255, 210, 90))
        self.buttons = [self.btn_left, self.btn_right, self.btn_jump, self.btn_dash,
                         self.btn_glide, self.btn_pause]

    def _active_buttons(self):
        return [b for b in self.buttons if b.is_active()]

    def _buttons_for_state(self):
        """Which buttons actually exist right now. Movement/action buttons only make
        sense while the world is live ("playing"); once paused, everything except the
        Pause/Resume toggle disappears (the real pause menu takes over from there, and
        it's just as reachable from a touch as from a mouse -- see handle_events()).
        Anywhere else (menus, settings, help, ...) there's no touch-overlay button at
        all."""
        if state == "playing":
            return self._active_buttons()
        if state == "paused":
            return [self.btn_pause]
        return []

    def held_keys(self):
        return {k for b in self._buttons_for_state() if b.held for k in b.keys}

    def on_finger_down(self, event):
        self.ever_touched = True
        self.last_touch_ms = _now_ms()
        pos = _finger_event_logical_pos(event)
        for b in self._buttons_for_state():
            if b.contains(pos, self.swap_sides, self.size, TOUCH_SAFE_MARGIN):
                b.fingers.add(event.finger_id)
                if b.on_press:
                    b.on_press()

    def on_finger_up(self, event):
        self.last_touch_ms = _now_ms()
        for b in self.buttons:
            b.fingers.discard(event.finger_id)

    def on_finger_motion(self, event):
        self.last_touch_ms = _now_ms()

    def update(self, dt_ms):
        """Advance the fade timer -- called once per rendered frame regardless of game
        state, so a touch anywhere (even in a menu) starts the fade-in, and a hybrid
        device keeps counting down toward hiding again even on frames where the overlay
        itself isn't drawn."""
        if not self.ever_touched:
            self._alpha = 0.0
            return
        floor = TOUCH_ANDROID_MIN_ALPHA if self.is_android else 0.0
        idle_ms = _now_ms() - self.last_touch_ms
        target = 1.0 if idle_ms < TOUCH_INACTIVITY_MS else floor
        step = dt_ms / TOUCH_FADE_MS
        if self._alpha < target:
            self._alpha = min(target, self._alpha + step)
        else:
            self._alpha = max(target, self._alpha - step)

    def effective_alpha(self):
        return self._alpha * self.opacity

    def draw(self, surf):
        buttons = self._buttons_for_state()
        eff = self.effective_alpha()
        if not buttons or eff <= 0.01:
            return
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for b in buttons:
            b.draw(overlay, self.swap_sides, self.size, TOUCH_SAFE_MARGIN)
        overlay.set_alpha(round(255 * eff))
        surf.blit(overlay, (0, 0))


def build_menu_background():
    # simple flat dark gradient -- no grid/sun clutter, just a calm neon-toned backdrop
    surf = pygame.Surface((SCREEN_W, SCREEN_H))
    top, bottom = (12, 10, 24), (26, 16, 38)
    for y in range(SCREEN_H):
        t = y / SCREEN_H
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (SCREEN_W, y))
    return surf


def draw_glow_text(surf, text, font, center, color):
    """A single soft halo behind the crisp text -- simpler than a multi-directional glow."""
    glow = font.render(text, True, color)
    glow = pygame.transform.smoothscale(glow, (int(glow.get_width() * 1.08), int(glow.get_height() * 1.2)))
    glow.set_alpha(90)
    surf.blit(glow, glow.get_rect(center=center))
    main = font.render(text, True, color)
    surf.blit(main, main.get_rect(center=center))


def draw_menu_button(rect, label, hovered, primary=False, font_obj=None, sub_label=None):
    """Shared button chrome for every menu screen (main menu, pause menu, trail menu,
    help menu) so navigation looks and behaves the same everywhere instead of each screen
    inventing its own button style. `primary` renders a bold filled action (e.g. Play,
    Resume); everything else renders as a quieter outlined secondary action."""
    font_obj = font_obj or small_font
    base = (0, 195, 155) if primary else (40, 58, 58)
    hover = (0, 255, 200) if primary else (58, 84, 82)
    bg = hover if hovered else base
    pygame.draw.rect(screen, bg, rect, border_radius=10)
    border = (210, 255, 245) if primary else (140, 195, 190)
    pygame.draw.rect(screen, border, rect, 2, border_radius=10)
    text_color = (8, 20, 18) if primary else WHITE
    if sub_label:
        label_surf = font_obj.render(label, True, text_color)
        sub_surf = small_font.render(sub_label, True, text_color if primary else (170, 200, 198))
        total_h = label_surf.get_height() + sub_surf.get_height()
        top = rect.centery - total_h // 2
        screen.blit(label_surf, label_surf.get_rect(midtop=(rect.centerx, top)))
        screen.blit(sub_surf, sub_surf.get_rect(midtop=(rect.centerx, top + label_surf.get_height())))
    else:
        label_surf = font_obj.render(label, True, text_color)
        screen.blit(label_surf, label_surf.get_rect(center=rect.center))


GOAL_TRIGGER_WIDTH_TILES = 3  # finish trigger column is this many tiles wide, centered on the flag


def build_level(level_map, enemy_speed=2, palette=None, rare_coin_chance=0.0, rng=None):
    """`rare_coin_chance`/`rng` control how many plain coin tiles spawn as a RareCoin
    instead (see the Lucky Charm upgrade). Defaulting rare_coin_chance to 0.0 keeps this
    function fully deterministic given the same level_map when called without them (as
    every existing test does), and load_level() seeds its own rng deterministically from
    level_num so the *real* in-game call sites stay reproducible too."""
    palette = palette or BIOME_PALETTES[0]
    rng = rng or random.Random()
    platforms = pygame.sprite.Group()
    one_way_platforms = pygame.sprite.Group()
    spikes = pygame.sprite.Group()
    enemies = pygame.sprite.Group()
    coins = pygame.sprite.Group()
    checkpoints = pygame.sprite.Group()
    wind_zones = pygame.sprite.Group()
    goal_rect = None
    player_start = (0, 0)

    for row, line in enumerate(level_map):
        for col, ch in enumerate(line):
            x, y = col * TS, row * TS
            if ch == "G":
                platforms.add(Platform(x, y, TS, TS, palette["ground"], palette["ground_top"]))
            elif ch == "B":
                platforms.add(CrumblingPlatform(x, y, TS, TS, palette["ground"], palette["ground_top"]))
            elif ch == "I":
                platforms.add(IceFloor(x, y, TS, TS))
            elif ch == "=":
                one_way_platforms.add(OneWayPlatform(x, y, TS, palette["ground_top"]))
            elif ch == "^":
                extra_h = SPIKE_TALL_EXTRA_HEIGHT if random.random() < SPIKE_TALL_CHANCE else 0
                spikes.add(Spike(x, y, TS, TS, extra_height=extra_h))
            elif ch == "~":
                spikes.add(HazardPool(x, y, TS, TS))
            elif ch == "M":
                spikes.add(MovingSpike(x, y, TS, TS))
            elif ch == "Z":
                spikes.add(TimedSpike(x, y, TS, TS))
            elif ch == "V":
                spikes.add(GasVent(x, y, TS, TS))
            elif ch == "E":
                enemies.add(Enemy(x, y, enemy_speed))
            elif ch == "F":
                enemies.add(FlyingEnemy(x, y - TS, enemy_speed))
            elif ch == "J":
                enemies.add(JumperEnemy(x, y, enemy_speed))
            elif ch == "C":
                enemies.add(ChargerEnemy(x, y, enemy_speed))
            elif ch == "S":
                enemies.add(ShieldedEnemy(x, y, enemy_speed))
            elif ch == "T":
                enemies.add(SplitterEnemy(x, y, enemy_speed))
            elif ch == "D":
                enemies.add(DiveEnemy(x, y - TS, enemy_speed))
            elif ch == "H":
                enemies.add(HomingFlyerEnemy(x, y - TS, enemy_speed))
            elif ch == "U":
                enemies.add(TurretEnemy(x, y, enemy_speed))
            elif ch == "L":
                enemies.add(SpitterEnemy(x, y, enemy_speed))
            elif ch == ">":
                wind_zones.add(WindZone(x, y, 1))
            elif ch == "<":
                wind_zones.add(WindZone(x, y, -1))
            elif ch == "c":
                coins.add(RareCoin(x, y) if rng.random() < rare_coin_chance else Coin(x, y))
            elif ch == "K":
                checkpoints.add(Checkpoint(x, y))
            elif ch == "X":
                goal_rect = pygame.Rect(x, y, TS, TS)
            elif ch == "P":
                player_start = (x, y)

    # Finish trigger: a tall column around the flag, not just the flag's own tile, so
    # jumping over or above it still completes the level -- the exact miss the flag-only
    # hitbox used to allow. Boundary walls just past each edge of the map (most importantly
    # just past the goal) stop the player from ever running off the generated level, which
    # is what used to happen if you cleared the goal tile without touching it.
    goal_trigger_rect = None
    cols = len(level_map[0]) if level_map else 0
    if goal_rect is not None:
        trigger_w = TS * GOAL_TRIGGER_WIDTH_TILES
        goal_trigger_rect = pygame.Rect(goal_rect.centerx - trigger_w // 2, 0, trigger_w, LEVEL_HEIGHT)
    if cols:
        platforms.add(InvisibleWall(-TS, 0, TS, LEVEL_HEIGHT))
        platforms.add(InvisibleWall(cols * TS, 0, TS, LEVEL_HEIGHT))

    return (platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect, player_start,
            checkpoints, wind_zones)


def load_level(level_num, rare_coin_chance=0.0):
    level_map, enemy_speed = generate_level(level_num)
    palette = get_biome(level_num)
    rng = random.Random(50000 + level_num)
    (platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect,
     player_start, checkpoints, wind_zones) = build_level(
        level_map, enemy_speed, palette, rare_coin_chance=rare_coin_chance, rng=rng
    )
    level_width = len(level_map[0]) * TS
    return (platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect,
            player_start, checkpoints, wind_zones, level_width, palette)


# ---------------- save file (level, score, and settings all together) ----------------
SAVE_DEFAULTS = {
    "save_version": SAVE_VERSION,
    "level": 1, "score": 0, "coins": 0,
    "screen_shake": 1, "music_volume": 50, "sfx_volume": 70,
    "stat_coins": 0, "stat_stomps": 0, "stat_deaths": 0, "stat_max_combo": 0,
    "equipped_trail_id": 0,  # 0 = no trail equipped, else a 1-based index into TRAILS
    "trail_visible": 1,
    "fps_cap": FPS_CAP_DEFAULT,  # 0 = Unlimited, else a capped positive integer -- see Settings
    "touch_opacity": 80, "touch_size": 50, "touch_swap_sides": 0,
    "touch_ever_used": 0,  # sticky flag: once a real touch is seen, Settings keeps showing
                           # the touch rows even after restarting with no keyboard/mouse used
}
SAVE_DEFAULTS.update({f"upg_{key}": 0 for key in UPGRADE_DEFAULTS})
SAVE_DEFAULTS.update({f"trail_owned_{key}": 0 for key in TRAIL_DEFAULTS})


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _clamp_save_data(data):
    """Force every field into its valid range, no matter what was actually read from
    disk. A hand-edited or corrupted save.txt can hand us anything -- negative levels,
    a music volume of 9000, an upgrade level past its max (which would otherwise index
    UPGRADES[...]['costs'] with a huge or negative number) -- this is the single place
    that guarantees the rest of the game only ever sees sane values."""
    data = dict(data)
    data["save_version"] = _clamp(data.get("save_version", SAVE_VERSION), 1, SAVE_VERSION)
    data["level"] = max(1, data.get("level", 1))
    data["score"] = max(0, data.get("score", 0))
    data["coins"] = max(0, data.get("coins", 0))
    data["screen_shake"] = 1 if data.get("screen_shake", 1) else 0
    data["music_volume"] = _clamp(data.get("music_volume", 50), 0, 100)
    data["sfx_volume"] = _clamp(data.get("sfx_volume", 70), 0, 100)
    data["stat_coins"] = max(0, data.get("stat_coins", 0))
    data["stat_stomps"] = max(0, data.get("stat_stomps", 0))
    data["stat_deaths"] = max(0, data.get("stat_deaths", 0))
    data["stat_max_combo"] = max(0, data.get("stat_max_combo", 0))
    for key, max_level in UPGRADE_MAX_LEVELS.items():
        field = f"upg_{key}"
        data[field] = _clamp(data.get(field, 0), 0, max_level)
    for key in TRAIL_DEFAULTS:
        field = f"trail_owned_{key}"
        data[field] = 1 if data.get(field, 0) else 0
    data["equipped_trail_id"] = _clamp(data.get("equipped_trail_id", 0), 0, len(TRAILS))
    if data["equipped_trail_id"]:
        equipped_key = TRAILS[data["equipped_trail_id"] - 1]["key"]
        if not data.get(f"trail_owned_{equipped_key}", 0):
            data["equipped_trail_id"] = 0  # can't have an unowned trail equipped
    data["trail_visible"] = 1 if data.get("trail_visible", 1) else 0
    fps_cap_value = data.get("fps_cap", FPS_CAP_DEFAULT)
    data["fps_cap"] = FPS_CAP_UNLIMITED if fps_cap_value <= 0 else _clamp(fps_cap_value, FPS_CAP_MIN, FPS_CAP_MAX)
    data["touch_opacity"] = _clamp(data.get("touch_opacity", 80), 0, 100)
    data["touch_size"] = _clamp(data.get("touch_size", 50), 0, 100)
    data["touch_swap_sides"] = 1 if data.get("touch_swap_sides", 0) else 0
    data["touch_ever_used"] = 1 if data.get("touch_ever_used", 0) else 0
    return data


def load_save(path=None):
    """Read a save file into a dict of int fields, tolerating a missing file, unreadable
    file, or individual corrupted/garbage lines -- always returns a complete, validated
    dict (every SAVE_DEFAULTS key present, every value in its safe range). Defaults to
    the currently-configured SAVE_FILE; choose_save_file() passes an explicit `path` to
    preview *other* save slots without switching the game to them."""
    if path is None:
        path = SAVE_FILE
    data = dict(SAVE_DEFAULTS)
    if not os.path.exists(path):
        write_save(data, path)
        return data
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return _clamp_save_data(data)
    # parse each line independently so one corrupted/truncated line doesn't throw away
    # every other value that parsed fine
    saw_save_version = False
    saw_stat_coins = False
    for line in lines:
        key, _, value = line.strip().partition("=")
        if key in data:
            try:
                data[key] = int(value)
                if key == "save_version":
                    saw_save_version = True
                elif key == "stat_coins":
                    saw_stat_coins = True
            except ValueError:
                pass
    data = _clamp_save_data(data)
    # equipped_trail_id is a *position* in TRAILS, and that table was reordered in
    # save_version 5 (see SAVE_VERSION's comment) -- a save from before that point has a
    # position that now points at a different trail, so it's safest to just unequip
    # rather than silently equip the wrong one. Ownership (trail_owned_*, keyed by name)
    # is completely unaffected and doesn't need any migration.
    if not saw_save_version or data["save_version"] < 5:
        data["equipped_trail_id"] = 0
    if not saw_save_version:
        # Claude's first stat build stored stat_coins as coin pickup count. The menu
        # now shows full lifetime currency earned, so old saves are promoted once.
        if saw_stat_coins:
            data["stat_coins"] = max(data["stat_coins"] * COIN_VALUE, data["score"])
        else:
            data["stat_coins"] = data["score"]
        data["save_version"] = SAVE_VERSION
    return _clamp_save_data(data)


def write_save(data, path=None):
    # write to a temp file and rename over the real one -- os.replace is atomic, so a
    # crash or power loss mid-write can never leave the save file half-written/corrupted
    if path is None:
        path = SAVE_FILE
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    os.replace(tmp_path, path)


def save_game():
    """Write the full current game state to disk immediately. Called at natural
    checkpoints (level complete, shop purchase, pausing/quitting) rather than on every
    single coin pickup -- see mark_dirty()/maybe_autosave() for the safety net that
    covers everything in between without hammering the disk every frame."""
    global _dirty
    data = {
        "save_version": SAVE_VERSION,
        "level": level_num,
        "score": score,
        "coins": wallet,
        "screen_shake": int(screen_shake_enabled),
        "music_volume": music_volume,
        "sfx_volume": sfx_volume,
        "stat_coins": stat_coins,
        "stat_stomps": stat_stomps,
        "stat_deaths": stat_deaths,
        "stat_max_combo": stat_max_combo,
        "equipped_trail_id": TRAIL_KEY_TO_ID.get(equipped_trail_key, 0) if equipped_trail_key else 0,
        "trail_visible": int(trail_visible),
        "fps_cap": fps_cap,
        "touch_opacity": round(touch_controls.opacity * 100),
        "touch_size": round((touch_controls.size - TOUCH_SIZE_MIN) / (TOUCH_SIZE_MAX - TOUCH_SIZE_MIN) * 100),
        "touch_swap_sides": int(touch_controls.swap_sides),
        "touch_ever_used": int(touch_controls.ever_touched),
    }
    data.update({f"upg_{key}": lvl for key, lvl in upgrades.items()})
    data.update({f"trail_owned_{key}": 1 if key in owned_trails else 0 for key in TRAIL_DEFAULTS})
    write_save(_clamp_save_data(data))
    _dirty = False


_dirty = False
_last_autosave = 0


# ---------------- multiple save files ----------------
def _is_save_filename(name):
    """True for "save.txt" or "save<digits>.txt" specifically (not e.g.
    "saved_settings.txt" or "savegame.txt.bak") -- only genuine save slots should ever
    show up in the picker below."""
    if not (name.startswith("save") and name.endswith(".txt")):
        return False
    middle = name[len("save"):-len(".txt")]
    return middle == "" or middle.isdigit()


def _save_slot_sort_key(filename):
    digits = filename[len("save"):-len(".txt")]
    return int(digits) if digits else 0  # "save.txt" (no digits) always sorts first


def _discover_save_files():
    """Every save*.txt file sitting next to the currently-configured SAVE_FILE (same
    directory) -- lets one shared build/install support independent save slots with zero
    configuration: drop another save1.txt, save2.txt, ... next to save.txt and
    choose_save_file() offers a picker automatically. Returns a sorted list of full paths;
    empty or single-item lists (the overwhelmingly common case, including every brand new
    install with no save yet) mean there's nothing to choose between."""
    save_dir = os.path.dirname(SAVE_FILE) or "."
    if not os.path.isdir(save_dir):
        return []
    names = sorted((n for n in os.listdir(save_dir) if _is_save_filename(n)), key=_save_slot_sort_key)
    return [os.path.join(save_dir, n) for n in names]


def _backup_dir():
    """Where backup_save() (and nothing else) writes to -- a subfolder next to the real
    save file, deliberately never scanned by _discover_save_files(). An earlier version
    of this feature wrote save<N>.txt directly next to save.txt, which meant every backup
    immediately became a second selectable slot in choose_save_file()'s "multiple save
    files found" picker at the next launch -- confusing at best (why is the game asking
    which save to play?), and at worst an easy way to accidentally launch into a stale
    backup instead of your real progress. Backups now live somewhere that picker never
    looks, so making one can never change which save the game actually loads."""
    d = os.path.join(os.path.dirname(SAVE_FILE) or ".", "backups")
    os.makedirs(d, exist_ok=True)
    return d


def backup_save():
    """Duplicates the current save into backups/save_backup_<timestamp>.txt. This is
    export_save()'s fallback when a native file dialog isn't available (see
    _tk_file_dialog()) -- there's no tkinter on Android, where this in-place backup is
    what "export" means instead (see the Help entry for how to bring it back: drop the
    file into the save folder, as save.txt or any save<N>.txt, before launching).
    Returns the new path, or None if the copy failed (e.g. a read-only install
    location)."""
    save_game()  # flush the very latest state first, not a stale on-disk copy
    dest = os.path.join(_backup_dir(), f"save_backup_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        shutil.copy2(SAVE_FILE, dest)
        return dest
    except OSError:
        return None


class _FileDialogUnavailable(Exception):
    """tkinter isn't installed/importable -- expected on Android, and not guaranteed on
    every desktop Python install either."""


def _tk_file_dialog(mode, **kwargs):
    """Runs a native OS Save-As/Open dialog via tkinter -- pygame has no file dialog of
    its own, and this is the only way to let the player pick an arbitrary export/import
    location instead of a fixed in-folder slot. A hidden, throwaway Tk root drives the
    (modal, blocking) dialog and is destroyed immediately after; the pygame window stays
    open underneath the whole time, same as any native file picker layered on top of a
    game. Returns the chosen path, or None if the player cancelled."""
    try:
        import tkinter
        from tkinter import filedialog
    except Exception as e:
        raise _FileDialogUnavailable from e
    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        fn = filedialog.asksaveasfilename if mode == "save" else filedialog.askopenfilename
        path = fn(**kwargs)
    finally:
        root.destroy()
    return path or None


def export_save():
    """Lets the player export their save to any location/filename they choose via a
    native file dialog, falling back to backup_save() if tkinter isn't available.
    Returns a short message for the main menu's toast (see handle_events())."""
    save_game()
    try:
        dest = _tk_file_dialog(
            "save", defaultextension=".txt", filetypes=[("Save files", "*.txt"), ("All files", "*.*")],
            initialfile="platformer_save.txt", title="Export Save As",
        )
    except _FileDialogUnavailable:
        path = backup_save()
        return (f"No file dialog available -- saved a backup at backups/{os.path.basename(path)}"
                if path else "Export failed")
    if dest is None:
        return "Export cancelled"
    try:
        shutil.copy2(SAVE_FILE, dest)
        return f"Exported save to {dest}"
    except OSError:
        return "Export failed -- couldn't write to that location"


def import_save():
    """Lets the player import a save from any file they choose via a native file dialog,
    overwriting the current save and reloading the whole game state from it. load_save()
    tolerates a garbage/non-save file already (see its docstring), so picking the wrong
    file just lands on mostly-default values instead of crashing. Falls back to a short
    explanatory message if tkinter isn't available -- there's no tkinter on Android,
    where dropping the file into the save folder before launching is how importing works
    instead (see the Help entry)."""
    global SAVE_FILE
    try:
        src = _tk_file_dialog(
            "open", filetypes=[("Save files", "*.txt"), ("All files", "*.*")], title="Import Save",
        )
    except _FileDialogUnavailable:
        return "No file dialog available -- see Help for how to import a save manually"
    if src is None:
        return "Import cancelled"
    try:
        shutil.copy2(src, SAVE_FILE)
    except OSError:
        return "Import failed -- couldn't read that file"
    init_game_state()
    return f"Imported save from {src}"


def _save_picker_layout(candidates):
    """Fixed (path, preview_data, hitbox_rect) rows for the save picker -- computed once
    (each candidate file is read via load_save() for a level/score/coins preview) instead
    of every single frame."""
    rows = []
    y = 150
    for path in candidates:
        data = load_save(path)
        rect = pygame.Rect(0, 0, 560, 54)
        rect.center = (SCREEN_W // 2, y)
        rows.append((path, data, rect))
        y += 66
    return rows


def _draw_save_picker(rows, mouse_pos):
    screen.fill((12, 10, 24))
    title_surf = pygame.font.SysFont(None, 44).render("MULTIPLE SAVE FILES FOUND", True, (0, 255, 220))
    screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_W // 2, 60)))
    sub_font = pygame.font.SysFont(None, 22)
    sub_surf = sub_font.render("Click a save file to play it", True, (170, 170, 190))
    screen.blit(sub_surf, sub_surf.get_rect(center=(SCREEN_W // 2, 100)))

    row_font = pygame.font.SysFont(None, 26)
    for path, data, rect in rows:
        hovered = rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (58, 84, 82) if hovered else (40, 58, 58), rect, border_radius=8)
        pygame.draw.rect(screen, (140, 195, 190), rect, 2, border_radius=8)
        name_surf = row_font.render(os.path.basename(path), True, WHITE)
        screen.blit(name_surf, (rect.x + 14, rect.y + 6))
        info_surf = sub_font.render(
            f"Level {data['level']}   Score {data['score']}   Coins {data['coins']}",
            True, (170, 200, 198),
        )
        screen.blit(info_surf, (rect.x + 14, rect.y + 30))
    pygame.display.flip()


def choose_save_file():
    """If more than one save*.txt file sits next to the game, shows a simple picker so
    the player can choose which one to play, and points the global SAVE_FILE at it for
    the rest of the session -- e.g. so a developer's own save never gets loaded on a
    fresh player's machine just because a stray save2.txt ended up next to save.txt, or
    so multiple people can keep separate progress from one shared install. With zero or
    one candidate (every brand new install, and every existing install before this
    feature) this returns immediately without drawing anything -- completely unaffected.
    Runs its own tiny event loop since it happens before init_game_state() has set up the
    full menu/game state machine."""
    global SAVE_FILE
    candidates = _discover_save_files()
    if len(candidates) <= 1:
        return

    rows = _save_picker_layout(candidates)
    picker_clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for path, _data, rect in rows:
                    if rect.collidepoint(event.pos):
                        SAVE_FILE = path
                        return
        _draw_save_picker(rows, pygame.mouse.get_pos())
        picker_clock.tick(30)


def mark_dirty():
    """Flag that score/coins/stats have changed since the last save. Doesn't touch disk
    itself -- maybe_autosave() decides when to actually flush, so rapid-fire coin pickups
    (e.g. a magnet dragging in several at once) cost one write eventually, not five."""
    global _dirty
    _dirty = True


def maybe_autosave():
    global _last_autosave
    now = _now_ms()
    if _dirty and now - _last_autosave >= AUTOSAVE_INTERVAL_MS:
        save_game()
        _last_autosave = now


# ---------------- shop ----------------
def upgrade_cost(key):
    u = next(u for u in UPGRADES if u["key"] == key)
    level = upgrades[key]
    if level >= len(u["costs"]):
        return None
    base_cost = u["costs"][level]
    discount = SHOP_DISCOUNT_PER_LEVEL * upgrades.get("shop_discount", 0)
    return max(1, round(base_cost * (1 - discount)))


def buy_upgrade(key):
    global wallet
    cost = upgrade_cost(key)
    if cost is None or wallet < cost:
        return False
    wallet -= cost
    upgrades[key] += 1
    if key == "shield":
        player.shield_max = upgrades["shield"]
        player.shield_charges = min(player.shield_charges + 1, player.shield_max)
    elif key == "vitality":
        player.hearts_max = upgrades["vitality"]
        player.hearts = min(player.hearts + 1, player.hearts_max)
    save_game()
    snd_purchase.play()
    return True


def trail_price(key):
    discount = SHOP_DISCOUNT_PER_LEVEL * upgrades.get("shop_discount", 0)
    return max(1, round(TRAILS_BY_KEY[key]["price"] * (1 - discount)))


def buy_trail(key):
    """One-time purchase -- trails aren't leveled, so unlike buy_upgrade() this doesn't
    get more expensive on a repeat call; it just fails once already owned."""
    global wallet
    if key in owned_trails:
        return False
    cost = trail_price(key)
    if wallet < cost:
        return False
    wallet -= cost
    owned_trails.add(key)
    save_game()
    snd_purchase.play()
    return True


RECENT_TRAILS_MAX = 3  # how many quick-swap slots (keys 1/2/3 on the Trail Menu) to remember


def equip_trail(key):
    """Pass None to unequip entirely -- no trail rendered, no trail bonus active."""
    global equipped_trail_key, recent_trails
    if key is not None and key not in owned_trails:
        return False
    equipped_trail_key = key
    if key is not None:
        # Most-recently-equipped first; session-only (not saved), purely a convenience
        # for the Trail Menu's 1/2/3 quick-swap so switching between a few favorites
        # doesn't need re-browsing the full paginated list every time.
        recent_trails = [key] + [k for k in recent_trails if k != key]
        recent_trails = recent_trails[:RECENT_TRAILS_MAX]
    save_game()
    return True


def _quick_swap_candidates():
    return [k for k in recent_trails if k != equipped_trail_key and k in owned_trails]


def set_trail_visible(value):
    """Independent of which trail is equipped (or whether one is equipped at all): only
    controls whether the equipped trail's visual effect is drawn. Its gameplay bonus, if
    it has one, keeps working either way -- see _trail_bonus_value()."""
    global trail_visible
    trail_visible = bool(value)
    save_game()


# Shop layout: one tab per category (see SHOP_CATEGORIES/SHOP_PAGES). Each tab shows up to
# 6 rows in the same 6 fixed screen slots -- simpler and more testable than free scrolling
# (no partial-row clipping/scroll-offset math), and keeps click targets stable regardless
# of what's showing. A category with more than 6 items (Trails has 10) paginates *within*
# itself via shop_subpage, using the small Up/Down chevrons or the Up/Down keys -- category
# switching (Left/Right or A/D) always resets shop_subpage back to 0.
SHOP_ROW_Y0 = 146
SHOP_ROW_H = 42
SHOP_ROWS_PER_PAGE = 6
SHOP_INFO_LINE_H = 26  # line height for the plain-text Statistics/Coming Soon tabs (not
                       # item rows -- tighter than SHOP_ROW_H so the longest of the two,
                       # Statistics' 9 lines, still clears the back-button prompt below it
SHOP_ROW_SLOTS = [pygame.Rect(SCREEN_W // 2 - 340, SHOP_ROW_Y0 + i * SHOP_ROW_H, 680, 36) for i in range(SHOP_ROWS_PER_PAGE)]
SHOP_PAGE_NAV_Y = 127
SHOP_PREV_RECT = pygame.Rect(SCREEN_W // 2 - 260, SHOP_PAGE_NAV_Y - 14, 50, 28)
SHOP_NEXT_RECT = pygame.Rect(SCREEN_W // 2 + 210, SHOP_PAGE_NAV_Y - 14, 50, 28)
SHOP_SUBPAGE_UP_RECT = pygame.Rect(SCREEN_W // 2 + 350, SHOP_ROW_Y0, 26, 26)
SHOP_SUBPAGE_DOWN_RECT = pygame.Rect(SCREEN_W // 2 + 350, SHOP_ROW_Y0 + SHOP_ROW_H * (SHOP_ROWS_PER_PAGE - 1) + 10, 26, 26)


def _shop_category_items(category):
    """The full (unpaginated) list of item keys for a purchasable-item category. Empty
    for "Statistics"/"Coming Soon", which render custom content instead of item rows."""
    return SHOP_PAGES.get(category, [])


def _shop_subpage_count(category):
    total = len(_shop_category_items(category))
    return max(1, (total + SHOP_ROWS_PER_PAGE - 1) // SHOP_ROWS_PER_PAGE)


def _shop_visible_items(category, subpage):
    items = _shop_category_items(category)
    start = subpage * SHOP_ROWS_PER_PAGE
    return items[start:start + SHOP_ROWS_PER_PAGE]


def _trail_subpage_count():
    return max(1, (len(TRAILS) + SHOP_ROWS_PER_PAGE - 1) // SHOP_ROWS_PER_PAGE)


def _replay_choices():
    """Every level you've actually completed: 1 .. level_num-1. level_num itself is the
    level currently in progress, not yet beaten, so it's excluded. Nothing about a
    replayable level is stored anywhere -- generate_level() regenerates it identically
    from just the level number, so this list is just a range, not a lookup into saved
    data (see start_replay())."""
    return list(range(1, level_num))


def _replay_subpage_count():
    return max(1, (len(_replay_choices()) + SHOP_ROWS_PER_PAGE - 1) // SHOP_ROWS_PER_PAGE)


# ---------------- procedural sound (no asset files needed) ----------------
class _NoSound:
    def play(self, *a, **k):
        pass

    def set_volume(self, *a, **k):
        pass


def const(freq):
    return lambda t: freq


def make_sound(duration, freq_func, waveform="square", volume=0.35):
    if not SOUND_ENABLED:
        return _NoSound()
    n = int(SAMPLE_RATE * duration)
    buf = array.array("h", [0] * (n * 2))
    fade = max(1, int(n * 0.08))
    for i in range(n):
        t = i / SAMPLE_RATE
        phase = 2 * math.pi * freq_func(t) * t
        if waveform == "square":
            val = 1.0 if math.sin(phase) >= 0 else -1.0
        elif waveform == "noise":
            val = random.uniform(-1, 1)
        else:
            val = math.sin(phase)
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = (n - i) / fade
        sample = int(max(-1.0, min(1.0, val)) * volume * env * 32767)
        buf[i * 2] = sample
        buf[i * 2 + 1] = sample
    return pygame.mixer.Sound(buffer=buf)


def concat_sounds(sounds):
    if not SOUND_ENABLED:
        return _NoSound()
    raw = b"".join(s.get_raw() for s in sounds)
    return pygame.mixer.Sound(buffer=raw)


def apply_sfx_volume(value_0_1):
    for s in SFX_SOUNDS:
        s.set_volume(value_0_1)


def set_muted(value):
    global muted
    muted = value
    if muted:
        music.set_volume(0)
        apply_sfx_volume(0)
    else:
        music.set_volume(music_volume / 100.0)
        apply_sfx_volume(sfx_volume / 100.0)


def sync_settings_from_widgets():
    """Copy the menu widgets' current values into the real settings globals and apply
    them immediately. Called both when starting a run AND when quitting from the menu,
    so a slider drag or toggle flip is never silently lost just because you closed the
    game instead of pressing Play."""
    global screen_shake_enabled, music_volume, sfx_volume, fps_cap
    screen_shake_enabled = toggle_shake.value
    music_volume = int(slider_volume.value * 100)
    sfx_volume = int(slider_sfx.value * 100)
    fps_cap = fps_cap_control.value
    music.set_volume(music_volume / 100.0)
    apply_sfx_volume(sfx_volume / 100.0)
    if touch_controls.ever_touched:
        touch_controls.opacity = slider_touch_opacity.value
        touch_controls.size = TOUCH_SIZE_MIN + slider_touch_size.value * (TOUCH_SIZE_MAX - TOUCH_SIZE_MIN)
        touch_controls.swap_sides = toggle_touch_swap.value


def _sync_settings_widgets_from_globals():
    """The opposite direction of sync_settings_from_widgets() -- pulls Fullscreen/Show
    FPS/Mute's current values into their Settings widgets. Needed because those three
    (unlike Screen Shake, Music, SFX, FPS Cap) can also change from a keyboard shortcut
    (F11/F3/M) or the pause menu's quick Mute toggle while the Settings screen isn't even
    open, which would otherwise leave a widget showing a stale value next time it's
    opened. Called at every "open Settings" site, right before state flips to
    "paused_settings"."""
    toggle_fullscreen.value = fullscreen
    toggle_show_fps.value = show_fps
    toggle_mute.value = muted


def start_game():
    global state
    sync_settings_from_widgets()
    save_game()
    state = resume_state


def _load_level_into_world(lvl_num):
    """Populate every level-scoped global (platforms, enemies, camera, background, ...)
    for `lvl_num`, threading the current Lucky Charm rare-coin chance through. Shared by
    init_game_state(), advance_level(), and the replay system (start_replay()/
    end_replay()) so "build a fresh world for a level number" exists in exactly one place
    instead of being copy-pasted at every call site. Does NOT touch level_num, the
    player, or save.txt -- callers decide what those should be."""
    global platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect
    global player_start, checkpoints, wind_zones, level_width, level_palette, camera, background, projectiles
    rare_chance = LUCKY_CHARM_CHANCE_PER_LEVEL * upgrades["lucky_charm"] + _trail_bonus_value("rare_coin_chance")
    (platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect, player_start,
     checkpoints, wind_zones, level_width, level_palette) = load_level(lvl_num, rare_coin_chance=rare_chance)
    camera = Camera(level_width)
    background = Background(level_width, level_palette)
    projectiles = pygame.sprite.Group()  # turret shots never carry over between levels/replays


def advance_level():
    global level_num
    level_num += 1
    save_game()
    _load_level_into_world(level_num)
    player.start_pos = player_start
    player.checkpoint_pos = None  # a new level always starts you fresh, even with the upgrade
    player.respawn()


def start_replay(chosen_level):
    """Load a previously-completed level for a free-play replay without touching real
    progress. Nothing about the replayed level is stored -- generate_level()/build_level()
    are already fully deterministic from just the level number (same seeded RNG every
    time), so "replay level N" is simply "load level N again," the same call
    advance_level() already makes for real progression. Ending the replay (see
    end_replay()) reloads the player's actual current level fresh, so nothing about the
    replay leaks into their real save."""
    global is_replay, replay_level, state
    _load_level_into_world(chosen_level)
    player.start_pos = player_start
    player.checkpoint_pos = None
    player.respawn()
    is_replay = True
    replay_level = chosen_level
    state = "playing"


def end_replay():
    """Return from a replay to the player's real current level (level_num), which was
    left untouched in save.txt the whole time -- see start_replay()."""
    global is_replay
    is_replay = False
    _load_level_into_world(level_num)
    player.start_pos = player_start
    player.checkpoint_pos = None
    player.respawn()


def quit_game():
    """Centralized quit path (window-close button AND Esc-from-menu both use this) so
    settings and progress are always saved on the way out, not just on some paths.
    sync_settings_from_widgets() reads whatever the menu widgets are currently set to --
    that's safe to call unconditionally (not just when state == "menu") because the
    widgets are frozen at their last-set values the moment you leave the menu (via
    start_game() or by opening the shop) and nothing else changes those globals in the
    meantime, so this can never overwrite a newer value with a stale one. Guarding it to
    "menu" only was the bug: opening the shop after changing a slider, then closing the
    window from there, used to save the pre-change value."""
    global running
    sync_settings_from_widgets()
    save_game()
    running = False


ENEMY_RADAR_RANGE = 500   # world px; enemies farther than this never get an indicator
ENEMY_RADAR_MARGIN = 24   # px in from the screen edge the indicator arrows sit


def _effective_radar_range():
    """Danger Sense level 1 uses the base range; level 2 ("Keen Senses") extends it a
    lot further -- a function (not a flat constant) since draw_enemy_radar() is the only
    caller and this keeps the level-scaling logic in one place next to it."""
    if upgrades["enemy_radar"] <= 1:
        return ENEMY_RADAR_RANGE
    return ENEMY_RADAR_RANGE + ENEMY_RADAR_RANGE_BONUS_PER_LEVEL * (upgrades["enemy_radar"] - 1)


def _draw_radar_arrow(surf, camera, target_rect, color):
    """Shared geometry for a single edge-of-screen arrow pointing toward an off-screen
    `target_rect` -- the player is always at screen center by construction (the camera
    follows them), so no player position is needed here. Factored out of
    draw_enemy_radar() so Hazard Sense (which points at hazards instead of enemies)
    reuses the exact same drawing code rather than a second copy of the same trigonometry."""
    cx, cy = SCREEN_W / 2, SCREEN_H / 2
    ex, ey = camera.apply(target_rect).center
    vx, vy = ex - cx, ey - cy
    vlen = max(1.0, math.hypot(vx, vy))
    vx, vy = vx / vlen, vy / vlen
    scale = min(
        (cx - ENEMY_RADAR_MARGIN) / abs(vx) if vx else 1e9,
        (cy - ENEMY_RADAR_MARGIN) / abs(vy) if vy else 1e9,
    )
    px, py = cx + vx * scale, cy + vy * scale
    angle = math.atan2(vy, vx)
    tip = (px + math.cos(angle) * 8, py + math.sin(angle) * 8)
    left = (px + math.cos(angle + 2.5) * 7, py + math.sin(angle + 2.5) * 7)
    right = (px + math.cos(angle - 2.5) * 7, py + math.sin(angle - 2.5) * 7)
    pygame.draw.polygon(surf, color, [tip, left, right])


def _radar_visible_targets(rects_and_colors, player_rect, camera, max_range):
    """Filters (rect, color) pairs down to the ones worth drawing an arrow for: within
    max_range of the player and currently off-screen. Shared by the enemy and hazard
    passes in draw_enemy_radar() so both apply the exact same range/visibility rule."""
    for rect, color in rects_and_colors:
        dx = rect.centerx - player_rect.centerx
        dy = rect.centery - player_rect.centery
        dist = math.hypot(dx, dy)
        if dist > max_range or dist < 1:
            continue
        ex, ey = camera.apply(rect).center
        if 0 <= ex <= SCREEN_W and 0 <= ey <= SCREEN_H:
            continue  # already visible on screen, no indicator needed
        yield rect, color


def draw_enemy_radar(surf, enemies, player_rect, camera, spikes=(), platforms=()):
    """Danger Sense upgrade: a small colored arrow at the screen edge pointing toward any
    nearby enemy that's currently off-screen. Hazard Sense (a Mastery-tier upgrade that
    requires Danger Sense to mean anything -- note the shared early-out below) extends the
    same arrows to nearby off-screen GasVents and currently-shaking CrumblingPlatforms, so
    that new hazard variety benefits from the existing indicator system instead of needing
    one of its own. Purely informational -- reads camera and sprite positions, doesn't
    touch collision or game state at all."""
    if upgrades["enemy_radar"] <= 0:
        return
    targets = [(e.rect, getattr(e, "color", ENEMY_COLOR)) for e in enemies]
    if upgrades["hazard_sense"] > 0:
        for s in spikes:
            if isinstance(s, GasVent):
                targets.append((s.rect, GAS_VENT_ACTIVE_COLOR))
        for p in platforms:
            if isinstance(p, CrumblingPlatform) and p.state != "solid":
                targets.append((p.rect, CRUMBLE_WARNING_COLOR))
    radar_range = _effective_radar_range()
    for rect, color in _radar_visible_targets(targets, player_rect, camera, radar_range):
        _draw_radar_arrow(surf, camera, rect, color)


def draw_fps_overlay(top_right_y=10):
    """Draws the FPS counter in the top-right corner if enabled, and returns the y
    position the next thing stacked under it should start at (so it doesn't overlap
    whatever else is already drawn there, e.g. the Level readout during gameplay)."""
    if not show_fps:
        return top_right_y
    fps_surf = small_font.render(f"FPS: {clock.get_fps():.0f}", True, (0, 255, 120))
    draw_hud_text(screen, fps_surf, (SCREEN_W - fps_surf.get_width() - 10, top_right_y))
    return top_right_y + fps_surf.get_height() + 6


# ---------------- init: pygame/display ----------------
def init_pygame():
    """Everything that touches the OS window/audio device. Safe to call exactly once
    per process, and never runs just from `import main`."""
    global screen, clock, SOUND_ENABLED, fullscreen

    # On Windows, an app that isn't marked DPI-aware gets its window bitmap-scaled by
    # the OS when display scaling is >100% (very common, e.g. 125%/150% on laptops).
    # That silently shifts where the OS reports mouse clicks relative to what's
    # actually drawn -- big click targets mostly still work by luck, but a slider that
    # needs a precise position along a track becomes unusable. Has to happen before any
    # window is created.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
    except pygame.error:
        pass
    pygame.init()

    SOUND_ENABLED = pygame.mixer.get_init() is not None

    fullscreen = False
    # pygame.SCALED keeps all drawing code working in the fixed 800x440 logical
    # resolution and lets pygame handle scaling that up (with letterboxing) to the
    # real window/display size.
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED)
    pygame.display.set_caption("Platformer")
    clock = pygame.time.Clock()


# ---------------- init: sound objects ----------------
def init_sounds():
    global snd_jump, snd_land, snd_stomp, snd_death, snd_win, snd_double_jump
    global snd_coin, snd_shield, snd_purchase, snd_dash, snd_checkpoint, SFX_SOUNDS, music

    snd_jump = make_sound(0.15, lambda t: 400 + t * 1400, "square", 0.25)
    snd_land = make_sound(0.08, lambda t: 180 - t * 300, "square", 0.2)
    snd_stomp = make_sound(0.12, lambda t: 300 - t * 800, "square", 0.3)
    snd_death = make_sound(0.5, lambda t: 500 - t * 480, "square", 0.3)
    snd_win = concat_sounds([make_sound(0.12, const(f), "square", 0.3) for f in (523, 659, 784, 1047)])
    snd_double_jump = make_sound(0.12, lambda t: 700 + t * 900, "square", 0.22)
    snd_coin = make_sound(0.1, lambda t: 900 + t * 500, "square", 0.25)
    snd_shield = concat_sounds([make_sound(0.06, const(f), "square", 0.22) for f in (700, 500)])
    snd_purchase = concat_sounds([make_sound(0.08, const(f), "square", 0.25) for f in (600, 900, 1200)])
    snd_dash = make_sound(0.1, lambda t: 900 - t * 2000, "square", 0.25)
    snd_checkpoint = concat_sounds([make_sound(0.09, const(f), "square", 0.22) for f in (523, 784, 1047)])

    # every one-shot sound effect (not the looping background music) -- the volume baked
    # into each sound above is just its relative mix level; set_volume() applies an extra
    # multiplier on top of that at playback time, which is what the SFX slider controls.
    SFX_SOUNDS = [snd_jump, snd_land, snd_stomp, snd_death, snd_win, snd_double_jump, snd_coin, snd_shield,
                  snd_purchase, snd_dash, snd_checkpoint]

    bass_notes = (130.81, 130.81, 164.81, 146.83)
    music = concat_sounds([make_sound(0.3, const(f), "square", 0.12) for f in bass_notes])
    if SOUND_ENABLED:
        music.play(loops=-1)


# ---------------- init: hero sprite sheet ----------------
def init_sprites():
    """Sheet layout: rows are actions (IDLE/WALK/RUN/ATTACK/HURT/JUMP/DIE). Despite having
    7-8 columns, this is NOT an 8-directional sheet -- almost every column already faces
    left (they're genuine animation-cycle frames), with one stray back-facing pose mixed
    into each row. There is no clean mirrored right-facing art anywhere, so instead of
    hunting for a "right" column we use only the confirmed left-facing frames and mirror
    them with pygame.transform.flip() to get right-facing versions."""
    global _hero_sheet, HERO_SCALE
    global HERO_IDLE_LEFT, HERO_IDLE_RIGHT, HERO_RUN_LEFT, HERO_RUN_RIGHT
    global HERO_JUMP_LEFT, HERO_JUMP_RIGHT, HERO_DIE_FRAMES
    global RUN_ANIM_INTERVAL, DIE_ANIM_INTERVAL

    HERO_SCALE = 0.45
    _hero_sheet = pygame.image.load(str(HERO_SPRITESHEET_PATH)).convert_alpha()

    def cut(x0, y0, x1, y1):
        rect = pygame.Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        surf = _hero_sheet.subsurface(rect).copy()
        w, h = surf.get_size()
        return pygame.transform.smoothscale(surf, (max(1, int(w * HERO_SCALE)), max(1, int(h * HERO_SCALE))))

    def flip(surf):
        return pygame.transform.flip(surf, True, False)

    idle_y = (34, 158)
    run_y = (360, 486)
    jump_y = (844, 995)
    die_y = (1033, 1151)

    HERO_IDLE_LEFT = cut(274, idle_y[0], 347, idle_y[1])  # facing the camera, standing still
    HERO_IDLE_RIGHT = flip(HERO_IDLE_LEFT)

    # All 8 RUN columns, skipping column index 5 which is a running-away-from-camera
    # (back-of-head) pose rather than a left-facing stride -- mixing it into the cycle
    # is what caused the "freezes for a frame" glitch.
    HERO_RUN_LEFT = [
        cut(145, run_y[0], 242, run_y[1]),
        cut(269, run_y[0], 365, run_y[1]),
        cut(391, run_y[0], 488, run_y[1]),
        cut(511, run_y[0], 609, run_y[1]),
        cut(633, run_y[0], 730, run_y[1]),
        cut(892, run_y[0], 989, run_y[1]),
        cut(1016, run_y[0], 1113, run_y[1]),
    ]
    HERO_RUN_RIGHT = [flip(f) for f in HERO_RUN_LEFT]

    HERO_JUMP_LEFT = cut(390, jump_y[0], 455, jump_y[1])  # clean left profile
    HERO_JUMP_RIGHT = flip(HERO_JUMP_LEFT)

    HERO_DIE_FRAMES = [
        cut(150, die_y[0], 226, die_y[1]),
        cut(273, die_y[0], 362, die_y[1]),
        cut(380, die_y[0], 488, die_y[1]),
        cut(503, die_y[0], 610, die_y[1]),
        cut(628, die_y[0], 738, die_y[1]),
        cut(756, die_y[0], 860, die_y[1]),
        cut(878, die_y[0], 1002, die_y[1]),
        cut(1042, die_y[0], 1175, die_y[1]),
    ]

    RUN_ANIM_INTERVAL = 90   # ms between run-cycle frames
    DIE_ANIM_INTERVAL = 75   # ms per die frame (8 frames over the 600ms death delay)


# ---------------- init: game/session state ----------------
def init_game_state():
    global _save_data, level_num, score, wallet, upgrades, screen_shake_enabled
    global music_volume, sfx_volume, stat_coins, stat_stomps, stat_deaths, stat_max_combo
    global platforms, one_way_platforms, spikes, enemies, coins, goal_rect, goal_trigger_rect
    global player_start, checkpoints
    global level_width, level_palette, camera, background, player
    global font, small_font, title_font, MENU_BG
    global toggle_shake, slider_volume, slider_sfx, fps_cap, fps_cap_control
    global toggle_fullscreen, toggle_show_fps, toggle_mute
    global touch_controls, slider_touch_opacity, slider_touch_size, toggle_touch_swap
    global PLAY_RECT, SHOP_ENTRY_RECT, TRAIL_ENTRY_RECT, HELP_ENTRY_RECT, SHOP_BACK_PROMPT, SHOP_BACK_RECT
    global TRAIL_UNEQUIP_RECT, TRAIL_VISIBILITY_RECT
    global PAUSE_RESUME_RECT, PAUSE_HELP_RECT, PAUSE_SETTINGS_RECT, PAUSE_MAIN_MENU_RECT, PAUSE_QUIT_RECT
    global PAUSE_SHAKE_TOGGLE_RECT, PAUSE_MUTE_TOGGLE_RECT
    global running, state, resume_state, transition_start, muted, show_fps, shop_page, shop_subpage
    global particles, popups, _dirty, _last_autosave, menu_toast_text, menu_toast_until
    global owned_trails, equipped_trail_key, trail_visible, _last_trail_particle_time, trail_page, recent_trails
    global paused_from, help_scroll, is_replay, replay_level, replay_page
    global REPLAY_ENTRY_RECT, SETTINGS_ENTRY_RECT, QUIT_ENTRY_RECT, EXPORT_SAVE_RECT, IMPORT_SAVE_RECT

    _save_data = load_save()
    level_num = _save_data["level"]
    score = _save_data["score"]
    wallet = _save_data["coins"]  # spendable currency (separate from the permanent Score)
    upgrades = {key: _save_data[f"upg_{key}"] for key in UPGRADE_DEFAULTS}
    screen_shake_enabled = bool(_save_data["screen_shake"])
    music_volume = _save_data["music_volume"]
    sfx_volume = _save_data["sfx_volume"]
    stat_coins = _save_data["stat_coins"]
    stat_stomps = _save_data["stat_stomps"]
    stat_deaths = _save_data["stat_deaths"]
    stat_max_combo = _save_data["stat_max_combo"]
    fps_cap = _save_data["fps_cap"]
    music.set_volume(music_volume / 100.0)
    apply_sfx_volume(sfx_volume / 100.0)

    owned_trails = {key for key in TRAIL_DEFAULTS if _save_data[f"trail_owned_{key}"]}
    equipped_id = _save_data["equipped_trail_id"]
    equipped_trail_key = TRAILS[equipped_id - 1]["key"] if equipped_id else None
    trail_visible = bool(_save_data["trail_visible"])
    _last_trail_particle_time = 0
    recent_trails = [equipped_trail_key] if equipped_trail_key else []

    _load_level_into_world(level_num)
    player = Player(*player_start)
    is_replay = False
    replay_level = None
    replay_page = 0

    touch_controls = TouchControls()
    touch_controls.ever_touched = touch_controls.ever_touched or bool(_save_data["touch_ever_used"])
    touch_controls.opacity = _save_data["touch_opacity"] / 100.0
    touch_controls.size = TOUCH_SIZE_MIN + (_save_data["touch_size"] / 100.0) * (TOUCH_SIZE_MAX - TOUCH_SIZE_MIN)
    touch_controls.swap_sides = bool(_save_data["touch_swap_sides"])

    font = pygame.font.SysFont(None, 44)
    small_font = pygame.font.SysFont(None, 22)
    title_font = pygame.font.SysFont(None, 64)

    MENU_BG = build_menu_background()
    toggle_shake = Toggle(SCREEN_W // 2 - 32, 110, "Screen Shake", screen_shake_enabled)
    slider_volume = Slider(SCREEN_W // 2 - 100, 175, 200, "Music Volume", music_volume / 100.0)
    slider_sfx = Slider(SCREEN_W // 2 - 100, 235, 200, "Sound Effects", sfx_volume / 100.0)
    fps_cap_control = FpsCapControl(SCREEN_W // 2 - 146, 300, fps_cap)
    # Fullscreen/FPS-counter/Mute previously only had keyboard shortcuts (F11/F3/M) --
    # tucked into the LEFT margin beside the three centered rows above (each narrower than
    # the 800px-wide screen) so a touch or mouse click reaches every one of them too,
    # without a new row or disturbing the existing layout.
    # show_fps/muted are set below (they're session-only, never persisted) -- both
    # always start False, same as the literals used here.
    toggle_fullscreen = Toggle(46, 175, "Fullscreen", fullscreen)
    toggle_show_fps = Toggle(46, 235, "Show FPS", False)
    toggle_mute = Toggle(46, 300, "Mute", False)
    # Touch-specific rows are tucked into the RIGHT margin instead, and only ever show up
    # for players Settings' touch rows are actually relevant to (see
    # touch_controls.ever_touched) without disturbing the existing layout for anyone else.
    slider_touch_opacity = Slider(610, 175, 170, "Touch Opacity", touch_controls.opacity)
    slider_touch_size = Slider(610, 235, 170, "Touch Size", _save_data["touch_size"] / 100.0)
    toggle_touch_swap = Toggle(668, 300, "Swap Sides", touch_controls.swap_sides)
    PLAY_RECT = pygame.Rect(0, 0, 240, 54)
    PLAY_RECT.center = (SCREEN_W // 2, 278)
    _menu_btn_w, _menu_btn_h, _menu_btn_gap = 130, 40, 14
    _menu_row_start_x = SCREEN_W // 2 - (_menu_btn_w * 5 + _menu_btn_gap * 4) // 2
    SHOP_ENTRY_RECT = pygame.Rect(_menu_row_start_x, 320, _menu_btn_w, _menu_btn_h)
    TRAIL_ENTRY_RECT = pygame.Rect(_menu_row_start_x + (_menu_btn_w + _menu_btn_gap), 320, _menu_btn_w, _menu_btn_h)
    HELP_ENTRY_RECT = pygame.Rect(_menu_row_start_x + 2 * (_menu_btn_w + _menu_btn_gap), 320, _menu_btn_w, _menu_btn_h)
    REPLAY_ENTRY_RECT = pygame.Rect(_menu_row_start_x + 3 * (_menu_btn_w + _menu_btn_gap), 320, _menu_btn_w, _menu_btn_h)
    SETTINGS_ENTRY_RECT = pygame.Rect(_menu_row_start_x + 4 * (_menu_btn_w + _menu_btn_gap), 320, _menu_btn_w, _menu_btn_h)
    # Off in its own corner rather than a 6th slot in the row above: on a touch-only
    # device (no Esc key, and often no window-chrome close button either -- Android in
    # particular) that row is otherwise the only way in or out of the game, and Quit
    # doesn't belong crammed in alongside Shop/Trails/Help/Replay/Settings.
    QUIT_ENTRY_RECT = pygame.Rect(SCREEN_W - 98, 48, 84, 30)
    # Mirrors QUIT_ENTRY_RECT on the opposite corner -- see export_save()/import_save()
    # in handle_events() for what these actually do (a native file dialog, with a
    # fallback if tkinter isn't available).
    EXPORT_SAVE_RECT = pygame.Rect(14, 48, 100, 30)
    IMPORT_SAVE_RECT = pygame.Rect(14, 82, 100, 30)
    SHOP_BACK_PROMPT = "PRESS ESC OR CLICK TO GO BACK"
    SHOP_BACK_RECT = small_font.render(SHOP_BACK_PROMPT, True, WHITE).get_rect(center=(SCREEN_W // 2, 415))
    TRAIL_UNEQUIP_RECT = pygame.Rect(SCREEN_W // 2 - 260, 103, 220, 32)
    TRAIL_VISIBILITY_RECT = pygame.Rect(SCREEN_W // 2 + 40, 103, 220, 32)

    PAUSE_RESUME_RECT = pygame.Rect(0, 0, 280, 50)
    PAUSE_RESUME_RECT.center = (SCREEN_W // 2, 150)
    PAUSE_SHAKE_TOGGLE_RECT = pygame.Rect(0, 0, 64, 50)
    PAUSE_SHAKE_TOGGLE_RECT.center = (SCREEN_W // 2 - 200, 150)
    PAUSE_MUTE_TOGGLE_RECT = pygame.Rect(0, 0, 64, 50)
    PAUSE_MUTE_TOGGLE_RECT.center = (SCREEN_W // 2 + 200, 150)
    PAUSE_HELP_RECT = pygame.Rect(0, 0, 280, 44)
    PAUSE_HELP_RECT.center = (SCREEN_W // 2, 215)
    PAUSE_SETTINGS_RECT = pygame.Rect(0, 0, 280, 44)
    PAUSE_SETTINGS_RECT.center = (SCREEN_W // 2, 269)
    PAUSE_MAIN_MENU_RECT = pygame.Rect(0, 0, 280, 44)
    PAUSE_MAIN_MENU_RECT.center = (SCREEN_W // 2, 323)
    PAUSE_QUIT_RECT = pygame.Rect(0, 0, 280, 44)
    PAUSE_QUIT_RECT.center = (SCREEN_W // 2, 377)

    running = True
    state = "menu"  # see game_loop()/handle_events() for the full list of valid states
    resume_state = "playing"  # state to return to when leaving the menu (set fresh on first launch)
    transition_start = 0
    muted = False
    show_fps = False
    shop_page = 0
    shop_subpage = 0
    trail_page = 0
    paused_from = "playing"  # state Resume returns to (always "playing", kept explicit for clarity)
    help_scroll = 0
    particles = []
    popups = []
    _dirty = False
    _last_autosave = _now_ms()
    menu_toast_text = ""
    menu_toast_until = 0

# ---------------- game loop ----------------
def handle_events():
    global running, fullscreen, screen, show_fps, state, resume_state, shop_page, shop_subpage
    global trail_page, help_scroll, paused_from, replay_page, screen_shake_enabled, replay_search_text
    global menu_toast_text, menu_toast_until

    for event in pygame.event.get():
        # Every menu/button/slider/toggle in the game is written in terms of
        # MOUSEBUTTONDOWN/UP/MOTION -- that's left entirely alone here, so a touch is
        # picked up through SDL's own, platform-tested touch-to-mouse event synthesis
        # (on by default; see https://wiki.libsdl.org/SDL2/SDL_HINT_TOUCH_MOUSE_EVENTS)
        # exactly the same way a real mouse click would be. The one thing that mechanism
        # can't do is multitouch (it's a single synthesized pointer), which is what the
        # touch overlay's own buttons (Left/Right/Jump/...) need -- those are driven
        # directly from the real FINGERDOWN/UP/MOTION events instead, via TouchControls.
        if event.type == pygame.FINGERDOWN:
            touch_controls.on_finger_down(event)
        elif event.type == pygame.FINGERUP:
            touch_controls.on_finger_up(event)
        elif event.type == pygame.FINGERMOTION:
            touch_controls.on_finger_motion(event)
            if state == "help":
                # The help screen has no slider to drag -- a vertical swipe scrolls its
                # content directly, the one screen-specific touch gesture in the game.
                _, dy_px = _finger_event_logical_delta(event)
                help_scroll = max(0, min(_help_max_scroll(), round(help_scroll - dy_px)))

        if event.type == pygame.QUIT:
            quit_game()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if state in ("shop", "trail_menu", "replay_picker"):
                    state = "menu"
                elif state == "help":
                    state = paused_from
                elif state == "paused_settings":
                    sync_settings_from_widgets()
                    save_game()
                    state = paused_from  # "menu" or "paused", whichever opened Settings
                elif state == "menu":
                    quit_game()
                elif state in ("playing", "paused"):
                    _toggle_pause()
                else:
                    resume_state = state
                    save_game()
                    set_muted(False)
                    state = "menu"
            if event.key == pygame.K_F11:
                fullscreen = not fullscreen
                flags = pygame.SCALED | (pygame.FULLSCREEN if fullscreen else 0)
                screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
            if event.key == pygame.K_F3:
                show_fps = not show_fps
            if state == "menu":
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    start_game()
                if event.key == pygame.K_s:
                    state = "shop"
                if event.key == pygame.K_t:
                    state = "trail_menu"
                if event.key == pygame.K_h:
                    paused_from = "menu"
                    state = "help"
                if event.key == pygame.K_r:
                    replay_search_text = ""
                    state = "replay_picker"
                if event.key == pygame.K_o:
                    paused_from = "menu"
                    _sync_settings_widgets_from_globals()
                    state = "paused_settings"
            elif state == "replay_picker":
                if event.key == pygame.K_UP:
                    replay_page = max(0, replay_page - 1)
                elif event.key == pygame.K_DOWN:
                    replay_page = min(_replay_subpage_count() - 1, replay_page + 1)
                elif event.key == pygame.K_BACKSPACE:
                    replay_search_text = replay_search_text[:-1]
                elif event.key == pygame.K_RETURN:
                    if replay_search_text.isdigit() and int(replay_search_text) in _replay_choices():
                        start_replay(int(replay_search_text))
                    replay_search_text = ""
                elif event.unicode.isdigit() and len(replay_search_text) < 5:
                    replay_search_text += event.unicode
            elif state == "shop":
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    shop_page = (shop_page - 1) % len(SHOP_CATEGORIES)
                    shop_subpage = 0
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    shop_page = (shop_page + 1) % len(SHOP_CATEGORIES)
                    shop_subpage = 0
                elif event.key == pygame.K_UP:
                    shop_subpage = max(0, shop_subpage - 1)
                elif event.key == pygame.K_DOWN:
                    category = SHOP_CATEGORIES[shop_page]
                    shop_subpage = min(_shop_subpage_count(category) - 1, shop_subpage + 1)
            elif state == "trail_menu":
                if event.key == pygame.K_UP:
                    trail_page = max(0, trail_page - 1)
                elif event.key == pygame.K_DOWN:
                    trail_page = min(_trail_subpage_count() - 1, trail_page + 1)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    idx = (pygame.K_1, pygame.K_2, pygame.K_3).index(event.key)
                    candidates = _quick_swap_candidates()
                    if idx < len(candidates):
                        equip_trail(candidates[idx])
            elif state == "help":
                if event.key == pygame.K_UP:
                    help_scroll = max(0, help_scroll - HELP_SCROLL_STEP)
                elif event.key == pygame.K_DOWN:
                    help_scroll = min(_help_max_scroll(), help_scroll + HELP_SCROLL_STEP)
            elif state == "paused":
                if event.key == pygame.K_r:
                    state = "playing"
            elif state == "playing":
                if event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                    player.try_jump()
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    player.try_dash()
                if event.key == pygame.K_m:
                    set_muted(not muted)

        elif event.type == pygame.MOUSEWHEEL and state == "help":
            help_scroll = max(0, min(_help_max_scroll(), help_scroll - event.y * HELP_SCROLL_STEP))

        if state == "menu":
            toggle_shake.handle_event(event)
            slider_volume.handle_event(event)
            sfx_was_dragging = slider_sfx.dragging
            slider_sfx.handle_event(event)
            if sfx_was_dragging and event.type == pygame.MOUSEBUTTONUP:
                snd_coin.play()  # quick preview so you can hear the level you just set
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if PLAY_RECT.collidepoint(event.pos):
                    start_game()
                elif SHOP_ENTRY_RECT.collidepoint(event.pos):
                    state = "shop"
                elif TRAIL_ENTRY_RECT.collidepoint(event.pos):
                    state = "trail_menu"
                elif HELP_ENTRY_RECT.collidepoint(event.pos):
                    paused_from = "menu"
                    state = "help"
                elif REPLAY_ENTRY_RECT.collidepoint(event.pos):
                    state = "replay_picker"
                elif SETTINGS_ENTRY_RECT.collidepoint(event.pos):
                    paused_from = "menu"
                    _sync_settings_widgets_from_globals()
                    state = "paused_settings"
                elif QUIT_ENTRY_RECT.collidepoint(event.pos):
                    quit_game()
                elif EXPORT_SAVE_RECT.collidepoint(event.pos):
                    menu_toast_text = export_save()
                    menu_toast_until = _now_ms() + 4000
                elif IMPORT_SAVE_RECT.collidepoint(event.pos):
                    menu_toast_text = import_save()
                    menu_toast_until = _now_ms() + 4000
        elif state == "shop":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                category = SHOP_CATEGORIES[shop_page]
                if SHOP_BACK_RECT.collidepoint(event.pos):
                    state = "menu"
                elif SHOP_PREV_RECT.collidepoint(event.pos):
                    shop_page = (shop_page - 1) % len(SHOP_CATEGORIES)
                    shop_subpage = 0
                elif SHOP_NEXT_RECT.collidepoint(event.pos):
                    shop_page = (shop_page + 1) % len(SHOP_CATEGORIES)
                    shop_subpage = 0
                elif SHOP_SUBPAGE_UP_RECT.collidepoint(event.pos):
                    shop_subpage = max(0, shop_subpage - 1)
                elif SHOP_SUBPAGE_DOWN_RECT.collidepoint(event.pos):
                    shop_subpage = min(_shop_subpage_count(category) - 1, shop_subpage + 1)
                elif category in UPGRADE_CATEGORIES:
                    for i, up_key in enumerate(_shop_visible_items(category, shop_subpage)):
                        if SHOP_ROW_SLOTS[i].collidepoint(event.pos):
                            buy_upgrade(up_key)
                            break
                elif category == "Trails":
                    for i, trail_key in enumerate(_shop_visible_items(category, shop_subpage)):
                        if SHOP_ROW_SLOTS[i].collidepoint(event.pos):
                            buy_trail(trail_key)
                            break
        elif state == "trail_menu":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if SHOP_BACK_RECT.collidepoint(event.pos):
                    state = "menu"
                elif TRAIL_UNEQUIP_RECT.collidepoint(event.pos):
                    equip_trail(None)
                elif TRAIL_VISIBILITY_RECT.collidepoint(event.pos):
                    set_trail_visible(not trail_visible)
                elif SHOP_SUBPAGE_UP_RECT.collidepoint(event.pos):
                    trail_page = max(0, trail_page - 1)
                elif SHOP_SUBPAGE_DOWN_RECT.collidepoint(event.pos):
                    trail_page = min(_trail_subpage_count() - 1, trail_page + 1)
                else:
                    start = trail_page * SHOP_ROWS_PER_PAGE
                    for i, t in enumerate(TRAILS[start:start + SHOP_ROWS_PER_PAGE]):
                        if SHOP_ROW_SLOTS[i].collidepoint(event.pos):
                            if t["key"] in owned_trails:
                                equip_trail(None if equipped_trail_key == t["key"] else t["key"])
                            break
        elif state == "replay_picker":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if SHOP_BACK_RECT.collidepoint(event.pos):
                    state = "menu"
                elif SHOP_SUBPAGE_UP_RECT.collidepoint(event.pos):
                    replay_page = max(0, replay_page - 1)
                elif SHOP_SUBPAGE_DOWN_RECT.collidepoint(event.pos):
                    replay_page = min(_replay_subpage_count() - 1, replay_page + 1)
                else:
                    choices = _replay_choices()
                    start = replay_page * SHOP_ROWS_PER_PAGE
                    for i, lvl in enumerate(choices[start:start + SHOP_ROWS_PER_PAGE]):
                        if SHOP_ROW_SLOTS[i].collidepoint(event.pos):
                            start_replay(lvl)
                            break
        elif state == "help":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if SHOP_BACK_RECT.collidepoint(event.pos):
                    state = paused_from
        elif state == "paused":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if PAUSE_RESUME_RECT.collidepoint(event.pos):
                    state = "playing"
                elif PAUSE_SHAKE_TOGGLE_RECT.collidepoint(event.pos):
                    screen_shake_enabled = not screen_shake_enabled
                    toggle_shake.value = screen_shake_enabled  # keep the Settings widget in sync
                    save_game()
                elif PAUSE_MUTE_TOGGLE_RECT.collidepoint(event.pos):
                    set_muted(not muted)
                    toggle_mute.value = muted  # keep the Settings widget in sync
                elif PAUSE_HELP_RECT.collidepoint(event.pos):
                    paused_from = "paused"
                    state = "help"
                elif PAUSE_SETTINGS_RECT.collidepoint(event.pos):
                    paused_from = "paused"
                    _sync_settings_widgets_from_globals()
                    state = "paused_settings"
                elif PAUSE_MAIN_MENU_RECT.collidepoint(event.pos):
                    if is_replay:
                        end_replay()
                    resume_state = "playing"
                    save_game()
                    state = "menu"
                elif PAUSE_QUIT_RECT.collidepoint(event.pos):
                    quit_game()
        elif state == "paused_settings":
            toggle_shake.handle_event(event)
            slider_volume.handle_event(event)
            sfx_was_dragging = slider_sfx.dragging
            slider_sfx.handle_event(event)
            fps_cap_control.handle_event(event)
            toggle_fullscreen.handle_event(event)
            toggle_show_fps.handle_event(event)
            toggle_mute.handle_event(event)
            if touch_controls.ever_touched:
                slider_touch_opacity.handle_event(event)
                slider_touch_size.handle_event(event)
                toggle_touch_swap.handle_event(event)
            if sfx_was_dragging and event.type == pygame.MOUSEBUTTONUP:
                snd_coin.play()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if SHOP_BACK_RECT.collidepoint(event.pos):
                    sync_settings_from_widgets()
                    save_game()
                    state = paused_from  # "menu" or "paused", whichever opened Settings


def draw_menu():
    music.set_volume(slider_volume.value)
    apply_sfx_volume(slider_sfx.value)
    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "PLATFORMER", title_font, (SCREEN_W // 2, 46), (0, 255, 220))

    toggle_shake.draw(screen, small_font)
    slider_volume.draw(screen, small_font)
    slider_sfx.draw(screen, small_font)

    mouse_pos = pygame.mouse.get_pos()
    draw_menu_button(PLAY_RECT, "PLAY", PLAY_RECT.collidepoint(mouse_pos), primary=True, font_obj=font,
                      sub_label="Space / Click")
    draw_menu_button(SHOP_ENTRY_RECT, "SHOP", SHOP_ENTRY_RECT.collidepoint(mouse_pos), sub_label="S")
    draw_menu_button(TRAIL_ENTRY_RECT, "TRAILS", TRAIL_ENTRY_RECT.collidepoint(mouse_pos), sub_label="T")
    draw_menu_button(HELP_ENTRY_RECT, "HELP", HELP_ENTRY_RECT.collidepoint(mouse_pos), sub_label="H")
    draw_menu_button(REPLAY_ENTRY_RECT, "REPLAY", REPLAY_ENTRY_RECT.collidepoint(mouse_pos), sub_label="R")
    draw_menu_button(SETTINGS_ENTRY_RECT, "SETTINGS", SETTINGS_ENTRY_RECT.collidepoint(mouse_pos), sub_label="O")
    draw_menu_button(QUIT_ENTRY_RECT, "QUIT", QUIT_ENTRY_RECT.collidepoint(mouse_pos))
    draw_menu_button(EXPORT_SAVE_RECT, "EXPORT", EXPORT_SAVE_RECT.collidepoint(mouse_pos))
    draw_menu_button(IMPORT_SAVE_RECT, "IMPORT", IMPORT_SAVE_RECT.collidepoint(mouse_pos))

    # A live Export/Import confirmation or error briefly takes over this line instead of
    # adding new geometry that would have to squeeze into an already-tight 440px-tall
    # screen -- see export_save()/import_save()'s call sites in handle_events().
    # render_text_fit() rather than a plain render(): a chosen export/import path can
    # easily be wider than the screen (render_text_fit shrinks it to fit instead of
    # letting it run off both edges).
    if _now_ms() < menu_toast_until:
        stats_surf = render_text_fit(small_font, menu_toast_text, (255, 230, 120), SCREEN_W - 40)
    else:
        stats_surf = small_font.render(
            f"Lifetime: {stat_coins:,} coins earned | {stat_stomps:,} stomps | {stat_deaths:,} deaths",
            True,
            (140, 140, 160),
        )
    screen.blit(stats_surf, stats_surf.get_rect(center=(SCREEN_W // 2, 386)))

    hint = small_font.render(f"Coins: {wallet}   F11: Fullscreen   F3: FPS   Esc: Quit", True, (150, 150, 170))
    screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H - 18)))
    draw_fps_overlay()

    pygame.display.flip()


def _draw_shop_row(row_rect, mouse_pos, name_line, desc, status_text, status_color, affordable, resolved):
    """Shared row chrome for both upgrade rows and trail rows -- `resolved` means "nothing
    more to do here" (an upgrade at max level, or a trail already owned), which dims the
    row the same way in both cases.

    The status label is drawn first so its left edge is known, and the name/desc lines
    are then measured against that (via render_text_fit()) instead of just being blit
    directly -- otherwise a description a little too long for its row runs straight into
    the status text (or past the row's right edge) instead of being caught."""
    hovered = row_rect.collidepoint(mouse_pos)
    bg_color = (35, 45, 40) if resolved else ((55, 80, 75) if hovered else (40, 60, 55))
    pygame.draw.rect(screen, bg_color, row_rect, border_radius=8)
    border_color = (0, 255, 200) if affordable else (80, 80, 95)
    pygame.draw.rect(screen, border_color, row_rect, 2, border_radius=8)

    status_surf = small_font.render(status_text, True, status_color)
    status_rect = status_surf.get_rect(midright=(row_rect.right - 12, row_rect.centery))
    screen.blit(status_surf, status_rect)

    text_max_w = max(20, status_rect.left - 8 - (row_rect.x + 12))
    name_surf = render_text_fit(small_font, name_line, WHITE, text_max_w)
    screen.blit(name_surf, (row_rect.x + 12, row_rect.y + 3))
    desc_surf = render_text_fit(small_font, desc, (170, 170, 190), text_max_w)
    screen.blit(desc_surf, (row_rect.x + 12, row_rect.y + 19))


def draw_shop():
    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "SHOP", title_font, (SCREEN_W // 2, 37), (255, 90, 210))
    wallet_surf = font.render(f"Coins: {wallet}", True, (255, 223, 60))
    screen.blit(wallet_surf, wallet_surf.get_rect(center=(SCREEN_W // 2, 92)))

    category = SHOP_CATEGORIES[shop_page]
    mouse_pos = pygame.mouse.get_pos()

    nav_hover_color = (0, 255, 200)
    pygame.draw.rect(screen, (40, 60, 55), SHOP_PREV_RECT, border_radius=6)
    pygame.draw.rect(screen, nav_hover_color, SHOP_PREV_RECT, 2, border_radius=6)
    prev_surf = small_font.render("<", True, WHITE)
    screen.blit(prev_surf, prev_surf.get_rect(center=SHOP_PREV_RECT.center))
    pygame.draw.rect(screen, (40, 60, 55), SHOP_NEXT_RECT, border_radius=6)
    pygame.draw.rect(screen, nav_hover_color, SHOP_NEXT_RECT, 2, border_radius=6)
    next_surf = small_font.render(">", True, WHITE)
    screen.blit(next_surf, next_surf.get_rect(center=SHOP_NEXT_RECT.center))

    page_surf = small_font.render(
        f"{category}  ({shop_page + 1}/{len(SHOP_CATEGORIES)})  -- A/D or Left/Right to change tab",
        True, (255, 200, 240),
    )
    screen.blit(page_surf, page_surf.get_rect(center=(SCREEN_W // 2, SHOP_PAGE_NAV_Y)))

    if category == "Statistics":
        lines = [
            f"Current level: {level_num}",
            f"Score: {score:,}",
            f"Coins in wallet: {wallet:,}",
            f"Lifetime coins earned: {stat_coins:,}",
            f"Enemies stomped: {stat_stomps:,}",
            f"Deaths: {stat_deaths:,}",
            f"Best stomp combo: {stat_max_combo:,}",
            f"Upgrades owned: {sum(1 for lvl in upgrades.values() if lvl > 0)}/{len(UPGRADES)}",
            f"Trails owned: {len(owned_trails)}/{len(TRAILS)}",
        ]
        for i, line in enumerate(lines):
            surf = small_font.render(line, True, WHITE if i == 0 else (200, 200, 220))
            screen.blit(surf, (SCREEN_W // 2 - 300, SHOP_ROW_Y0 + i * SHOP_INFO_LINE_H))
    elif category == "Coming Soon":
        lines = [
            "Latest update -- everything below is now live:",
            "- Mastery-tier upgrades: Second Wind, Combo Momentum, Hazard Sense",
            "- 6 new trails with fresh particle styles (square/ring/comet)",
            "- New enemies: Splitter, Homing Flyer, Spitter",
            "- New hazards: Crumbling Floor, Gas Vent, Icy Floor",
            "- Multiple checkpoints per level, 3 new biomes, a sky staircase platform pattern",
            "More updates may still be on the way -- thanks for playing!",
        ]
        for i, line in enumerate(lines):
            surf = small_font.render(line, True, (255, 200, 240) if i == 0 else (170, 170, 190))
            screen.blit(surf, (SCREEN_W // 2 - 300, SHOP_ROW_Y0 + i * SHOP_INFO_LINE_H))
    else:
        visible = _shop_visible_items(category, shop_subpage)
        subpages = _shop_subpage_count(category)
        for i, key in enumerate(visible):
            row_rect = SHOP_ROW_SLOTS[i]
            if category == "Trails":
                t = TRAILS_BY_KEY[key]
                owned = key in owned_trails
                cost = trail_price(key)
                affordable = (not owned) and wallet >= cost
                status_text = "OWNED" if owned else f"{cost} coins"
                status_color = (120, 255, 160) if owned else ((255, 223, 60) if affordable else (150, 90, 90))
                name_line = t["name"] + ("  (equipped)" if equipped_trail_key == key else "")
                _draw_shop_row(row_rect, mouse_pos, name_line, t["desc"], status_text, status_color, affordable, owned)
            else:
                u = next(u for u in UPGRADES if u["key"] == key)
                level = upgrades[key]
                cost = upgrade_cost(key)
                affordable = cost is not None and wallet >= cost
                maxed = cost is None
                status_text = "MAXED" if maxed else f"{cost} coins"
                status_color = (120, 255, 160) if maxed else ((255, 223, 60) if affordable else (150, 90, 90))
                name_line = f"{u['name']}  ({level}/{len(u['costs'])})"
                _draw_shop_row(row_rect, mouse_pos, name_line, u["desc"], status_text, status_color, affordable, maxed)

        if subpages > 1:
            up_color = nav_hover_color if shop_subpage > 0 else (80, 80, 95)
            down_color = nav_hover_color if shop_subpage < subpages - 1 else (80, 80, 95)
            pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_UP_RECT, border_radius=6)
            pygame.draw.rect(screen, up_color, SHOP_SUBPAGE_UP_RECT, 2, border_radius=6)
            up_surf = small_font.render("^", True, WHITE)
            screen.blit(up_surf, up_surf.get_rect(center=SHOP_SUBPAGE_UP_RECT.center))
            pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_DOWN_RECT, border_radius=6)
            pygame.draw.rect(screen, down_color, SHOP_SUBPAGE_DOWN_RECT, 2, border_radius=6)
            down_surf = small_font.render("v", True, WHITE)
            screen.blit(down_surf, down_surf.get_rect(center=SHOP_SUBPAGE_DOWN_RECT.center))
            sub_surf = small_font.render(f"{shop_subpage + 1}/{subpages}", True, (200, 200, 220))
            screen.blit(sub_surf, sub_surf.get_rect(
                midtop=(SHOP_SUBPAGE_UP_RECT.centerx, SHOP_SUBPAGE_DOWN_RECT.top - 40)))

    back_surf = small_font.render(SHOP_BACK_PROMPT, True, (200, 200, 220))
    screen.blit(back_surf, SHOP_BACK_RECT)
    draw_fps_overlay()

    pygame.display.flip()


def _draw_trail_swatch(center, trail):
    """A small static preview of a trail's colors -- 3 dots for a multi-color trail, or a
    hue-cycled ring for the "rainbow" style. Cheap, always-on preview shown right in the
    row instead of needing a separate hover/preview mode."""
    if trail["style"] == "rainbow":
        now = pygame.time.get_ticks()
        for i in range(3):
            hue = ((now / 3000) + i / 3) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            cx = center[0] + (i - 1) * 10
            pygame.draw.circle(screen, (int(r * 255), int(g * 255), int(b * 255)), (cx, center[1]), 5)
    else:
        colors = trail["colors"]
        for i in range(3):
            cx = center[0] + (i - 1) * 10
            pygame.draw.circle(screen, colors[i % len(colors)], (cx, center[1]), 5)


def draw_trail_menu():
    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "TRAILS", title_font, (SCREEN_W // 2, 37), (180, 120, 255))
    wallet_surf = small_font.render(f"Coins: {wallet}", True, (255, 223, 60))
    screen.blit(wallet_surf, wallet_surf.get_rect(center=(SCREEN_W // 2, 84)))

    mouse_pos = pygame.mouse.get_pos()

    unequip_hover = TRAIL_UNEQUIP_RECT.collidepoint(mouse_pos)
    draw_menu_button(TRAIL_UNEQUIP_RECT, "No Trail (Unequip)", unequip_hover,
                      primary=(equipped_trail_key is None))
    vis_label = "Visibility: Visible" if trail_visible else "Visibility: Invisible"
    draw_menu_button(TRAIL_VISIBILITY_RECT, vis_label, TRAIL_VISIBILITY_RECT.collidepoint(mouse_pos))

    start = trail_page * SHOP_ROWS_PER_PAGE
    visible = TRAILS[start:start + SHOP_ROWS_PER_PAGE]
    for i, t in enumerate(visible):
        row_rect = SHOP_ROW_SLOTS[i]
        owned = t["key"] in owned_trails
        equipped = equipped_trail_key == t["key"]
        hovered = row_rect.collidepoint(mouse_pos)
        bg_color = (45, 60, 80) if equipped else ((55, 80, 75) if hovered and owned else (40, 60, 55))
        pygame.draw.rect(screen, bg_color, row_rect, border_radius=8)
        border_color = (0, 255, 200) if equipped else ((150, 150, 190) if owned else (80, 80, 95))
        pygame.draw.rect(screen, border_color, row_rect, 2, border_radius=8)

        _draw_trail_swatch((row_rect.x + 24, row_rect.centery), t)

        if owned:
            status_text, status_color = ("EQUIPPED" if equipped else "Click to equip"), (120, 255, 160)
        else:
            status_text, status_color = f"{trail_price(t['key'])} coins (buy in Shop)", (150, 150, 170)
        status_surf = small_font.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(midright=(row_rect.right - 12, row_rect.centery))
        screen.blit(status_surf, status_rect)

        # Measured against the status label's actual position (via render_text_fit()),
        # same as the Shop's own rows -- see _draw_shop_row()'s docstring for why a long
        # name/description can't just be blit directly without risking running into it.
        text_max_w = max(20, status_rect.left - 8 - (row_rect.x + 46))
        name_line = t["name"] + ("  (equipped)" if equipped else "")
        name_surf = render_text_fit(small_font, name_line, WHITE, text_max_w)
        screen.blit(name_surf, (row_rect.x + 46, row_rect.y + 3))
        desc_surf = render_text_fit(small_font, t["desc"], (170, 170, 190), text_max_w)
        screen.blit(desc_surf, (row_rect.x + 46, row_rect.y + 19))

    subpages = _trail_subpage_count()
    if subpages > 1:
        nav_color = (0, 255, 200)
        up_color = nav_color if trail_page > 0 else (80, 80, 95)
        down_color = nav_color if trail_page < subpages - 1 else (80, 80, 95)
        pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_UP_RECT, border_radius=6)
        pygame.draw.rect(screen, up_color, SHOP_SUBPAGE_UP_RECT, 2, border_radius=6)
        screen.blit(small_font.render("^", True, WHITE), small_font.render("^", True, WHITE).get_rect(center=SHOP_SUBPAGE_UP_RECT.center))
        pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_DOWN_RECT, border_radius=6)
        pygame.draw.rect(screen, down_color, SHOP_SUBPAGE_DOWN_RECT, 2, border_radius=6)
        screen.blit(small_font.render("v", True, WHITE), small_font.render("v", True, WHITE).get_rect(center=SHOP_SUBPAGE_DOWN_RECT.center))
        sub_surf = small_font.render(f"{trail_page + 1}/{subpages}", True, (200, 200, 220))
        screen.blit(sub_surf, sub_surf.get_rect(midtop=(SHOP_SUBPAGE_UP_RECT.centerx, SHOP_SUBPAGE_DOWN_RECT.top - 40)))

    candidates = _quick_swap_candidates()
    if candidates:
        names = ", ".join(f"[{i + 1}] {TRAILS_BY_KEY[key]['name']}" for i, key in enumerate(candidates))
        quick_swap_surf = small_font.render(f"Quick-swap: {names}", True, (200, 200, 220))
        screen.blit(quick_swap_surf, quick_swap_surf.get_rect(center=(SCREEN_W // 2, 403)))

    back_surf = small_font.render(SHOP_BACK_PROMPT, True, (200, 200, 220))
    screen.blit(back_surf, SHOP_BACK_RECT)
    draw_fps_overlay()

    pygame.display.flip()


# ---------------- help / guide ----------------
HELP_SCROLL_STEP = 40
HELP_CONTENT = [
    ("Controls", [
        "Move: Left/Right arrows or A/D.",
        "Jump: Space, Up, or W. Hold longer for a higher jump.",
        "Dash: Shift (once you've bought the Dash upgrade).",
        "Glide: hold Down while falling (once you've bought Feather Fall).",
        "Mute: M.  FPS counter: F3.  Fullscreen: F11.  Pause/back: Esc.",
        "Fullscreen, FPS counter, and Mute also have a toggle on the Settings screen, so",
        "they're reachable with a mouse or a touch, not just those keyboard shortcuts.",
    ]),
    ("Touch Controls", [
        "On a touchscreen -- Android, or a touch-capable device like a Windows tablet or",
        "a Steam Deck in touch mode -- on-screen buttons for Left/Right, Jump, Dash and",
        "Glide (once owned), and Pause appear the first time you touch the screen, fade",
        "out after a few seconds of inactivity, and fade back in the moment you touch",
        "again. On Android they never fully disappear, since there's no keyboard to fall",
        "back to. The Pause button doubles as Resume once the game is actually paused.",
        "Every menu, button, and slider in the game -- Play, Shop, Settings, Pause menu,",
        "Help's scrollbar (or just swipe), all of it -- responds to a tap the same way",
        "it responds to a mouse click, so no separate touch navigation exists or is",
        "needed. Once the game has seen a touch, three extra Settings rows appear to",
        "customize the on-screen buttons: Touch Opacity, Touch Size, and Swap Sides",
        "(mirrors Left/Right/Jump/Dash/Glide to the opposite side of the screen).",
    ]),
    ("Movement", [
        "You get a short grace window to jump just after walking off a ledge (coyote",
        "time), and a short window where a jump press just before landing still counts",
        "(jump buffer). The Ledge Grace and Quick Reflex upgrades stretch both.",
        "Extra Air Jump upgrades let you jump again while airborne (double, then triple).",
    ]),
    ("Coins & Score", [
        "Coins are your spendable currency -- picking one up is worth 10, stomping an",
        "enemy is worth 50. Score is a separate, permanent running total that only goes",
        "up. Cyan gem-shaped coins are rare coins worth 50 (need the Lucky Charm trail",
        "upgrade to appear at all).",
    ]),
    ("The Shop", [
        "Spend coins on permanent upgrades, organized into Movement, Survival, Economy,",
        "and Mastery tabs, plus a Trails tab for cosmetic purchases. Use A/D or",
        "Left/Right to switch tabs; Up/Down (or the small arrows) page through a tab",
        "with more than 6 items. Every upgrade has a hard maximum level and a rising price.",
    ]),
    ("Trails", [
        "Trails are visual effects you own permanently once bought in the Shop's Trails",
        "tab, listed cheapest to priciest. Every trail carries a small gameplay bonus --",
        "stated in its description -- that gets stronger the further down the list (and",
        "further up the price) you go, so it's a real progression, not just a look.",
        "The bonus keeps working even while Visibility is set to Invisible in the",
        "dedicated Trail Menu (main menu, T) -- only which trail is equipped matters,",
        "not whether you can see it. The Trail Menu also lets you unequip back to none.",
    ]),
    ("Checkpoints", [
        "Starting on level 5, longer levels place checkpoint flags partway across --",
        "just one on shorter levels, more on longer ones (up to 3), spaced evenly. If",
        "you own the Checkpoint Flag upgrade, walking through a flag's column --",
        "touching it, jumping over it, or passing above it all count -- makes it your",
        "respawn point for the rest of that level. A flag flashes gold with a chime the",
        "moment it becomes your active checkpoint.",
    ]),
    ("Hazards & Enemies", [
        "Beyond patrollers, fliers, jumpers, chargers, shielded enemies, divers, and",
        "turrets: Splitter enemies burst into two fast shards when stomped, Homing",
        "Flyers slowly drift toward your x position, and Spitters lob an arcing shot",
        "you have to judge the landing spot of. Crumbling floor tiles collapse a moment",
        "after you stand on them and reform later; icy floor patches make your footing",
        "slide instead of stopping instantly; gas vents cycle a tall dangerous column",
        "you have to jump above rather than just sidestep.",
    ]),
    ("Procedural Generation & Difficulty", [
        "Every level is generated from a seed based on its level number, so the same",
        "level number always looks the same. Difficulty ramps up smoothly: early levels",
        "keep pits narrow and hazards sparse, the mid game introduces tighter timing,",
        "hazard combinations, and new terrain (crumbling floor, ice), and the late game",
        "raises enemy variety, hazard density, and level width further. Seven rotating",
        "biomes (day, sunset, night, dawn, cavern, volcanic, crystal) keep the scenery",
        "varied. Every level is designed to be beatable without any upgrades at all.",
    ]),
    ("Upgrades", [
        "Upgrades are permanent, leveled improvements bought with coins -- movement",
        "(speed, jump, dash, coyote/buffer timing), survival (shield, extra hearts, air",
        "jumps, glide, checkpoints, enemy radar), economy (magnet, coin/stomp bonuses,",
        "luck, discounts, score multiplier), and Mastery: Second Wind (a one-time save",
        "per life), Combo Momentum (consecutive stomps grow in reward), and Hazard",
        "Sense (radar arrows also point at nearby gas vents/crumbling floor). None are",
        "required to beat any level; they make repeat runs faster and safer.",
    ]),
    ("Winning & Losing", [
        "Reach the goal flag's column to complete a level -- jumping over it still",
        "counts, you don't have to land exactly on it. Touching a hazard or enemy from",
        "the side is usually fatal, unless a Shield charge or Extra Heart absorbs it;",
        "falling into a pit is always instant, with no exceptions. Dying respawns you",
        "at your checkpoint (or the level start) without losing coins, score, or",
        "upgrades.",
    ]),
    ("Saving", [
        "Progress saves automatically: periodically while playing, and always when you",
        "finish a level, make a purchase, pause, or quit. Nothing is lost by closing",
        "the game normally.",
        "",
        f"The save file lives at: {os.path.dirname(SAVE_FILE) or '.'}",
        "",
        "EXPORT (top-left of the main menu) opens a native Save-As dialog so you can",
        "save a copy anywhere you like -- a USB drive, cloud folder, another device,",
        "wherever. IMPORT opens the matching Open dialog to load any save file back in,",
        "replacing your current progress. Neither is available on Android (there's no",
        "such dialog there) -- EXPORT instead drops a timestamped copy in a \"backups\"",
        "folder next to the save file, and IMPORTing means placing a file in the folder",
        "above (as save.txt) yourself before launching. If more than one save.txt/",
        "save<N>.txt sits directly in that folder (not in \"backups\"), a picker appears",
        "at startup letting you choose which one to play.",
    ]),
    ("Pause Menu", [
        "Esc during a run opens the Pause menu, which fully freezes gameplay. From",
        "there: Resume, open Help, adjust Settings, return to the Main Menu, or Quit.",
    ]),
    ("Settings & Statistics", [
        "Screen Shake, Music Volume, and Sound Effects volume live on the main menu;",
        "the full Settings screen (same controls, plus FPS Cap, Fullscreen, Show FPS,",
        "and Mute) opens from the main menu's SETTINGS button (O) or Pause -> Settings",
        "during a run. FPS Cap picks any frame rate, or Unlimited -- it only changes",
        "smoothness, never game speed. If the game has ever seen a touch, three more",
        "rows appear for customizing the on-screen touch buttons (see Touch Controls).",
        "The Shop's Statistics tab shows your current level, score, lifetime coins",
        "earned, stomps, deaths, best stomp combo, and how many upgrades/trails you own.",
    ]),
]


def draw_replay_picker():
    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "REPLAY A LEVEL", title_font, (SCREEN_W // 2, 37), (255, 190, 90))

    choices = _replay_choices()
    if not choices:
        msg = small_font.render("Complete level 1 to unlock replays.", True, (200, 200, 220))
        screen.blit(msg, msg.get_rect(center=(SCREEN_W // 2, 220)))
    else:
        mouse_pos = pygame.mouse.get_pos()
        subpages = _replay_subpage_count()
        info_surf = small_font.render(
            f"{len(choices)} level{'s' if len(choices) != 1 else ''} completed -- pick one to replay "
            f"(coins/score still count; your real progress is untouched)",
            True, (200, 200, 220),
        )
        screen.blit(info_surf, info_surf.get_rect(center=(SCREEN_W // 2, 84)))

        start = replay_page * SHOP_ROWS_PER_PAGE
        for i, lvl in enumerate(choices[start:start + SHOP_ROWS_PER_PAGE]):
            row_rect = SHOP_ROW_SLOTS[i]
            hovered = row_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, (55, 80, 75) if hovered else (40, 60, 55), row_rect, border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), row_rect, 2, border_radius=8)
            name_surf = small_font.render(f"Level {lvl}", True, WHITE)
            screen.blit(name_surf, (row_rect.x + 12, row_rect.y + 3))
            hint_surf = small_font.render("Click to replay", True, (170, 170, 190))
            screen.blit(hint_surf, (row_rect.x + 12, row_rect.y + 19))

        if subpages > 1:
            nav_color = (0, 255, 200)
            up_color = nav_color if replay_page > 0 else (80, 80, 95)
            down_color = nav_color if replay_page < subpages - 1 else (80, 80, 95)
            pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_UP_RECT, border_radius=6)
            pygame.draw.rect(screen, up_color, SHOP_SUBPAGE_UP_RECT, 2, border_radius=6)
            screen.blit(small_font.render("^", True, WHITE), small_font.render("^", True, WHITE).get_rect(center=SHOP_SUBPAGE_UP_RECT.center))
            pygame.draw.rect(screen, (40, 60, 55), SHOP_SUBPAGE_DOWN_RECT, border_radius=6)
            pygame.draw.rect(screen, down_color, SHOP_SUBPAGE_DOWN_RECT, 2, border_radius=6)
            screen.blit(small_font.render("v", True, WHITE), small_font.render("v", True, WHITE).get_rect(center=SHOP_SUBPAGE_DOWN_RECT.center))
            sub_surf = small_font.render(f"{replay_page + 1}/{subpages}", True, (200, 200, 220))
            screen.blit(sub_surf, sub_surf.get_rect(midtop=(SHOP_SUBPAGE_UP_RECT.centerx, SHOP_SUBPAGE_DOWN_RECT.top - 40)))

    back_surf = small_font.render(SHOP_BACK_PROMPT, True, (200, 200, 220))
    screen.blit(back_surf, SHOP_BACK_RECT)
    draw_fps_overlay()

    pygame.display.flip()


def _help_content_height():
    line_h = 20
    section_gap = 16
    total = 0
    for _title, lines in HELP_CONTENT:
        total += line_h + len(lines) * line_h + section_gap
    return total


def _help_max_scroll():
    viewport_h = 300
    return max(0, _help_content_height() - viewport_h)


def draw_help():
    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "HELP / GUIDE", title_font, (SCREEN_W // 2, 40), (0, 255, 220))

    viewport = pygame.Rect(SCREEN_W // 2 - 340, 80, 680, 300)
    pygame.draw.rect(screen, (18, 15, 28), viewport, border_radius=8)
    pygame.draw.rect(screen, (60, 55, 80), viewport, 1, border_radius=8)

    content = pygame.Surface((viewport.w, _help_content_height()), pygame.SRCALPHA)
    y = 0
    line_h = 20
    section_gap = 16
    for title, lines in HELP_CONTENT:
        title_surf = small_font.render(title, True, (0, 255, 200))
        content.blit(title_surf, (10, y))
        y += line_h
        for line in lines:
            line_surf = small_font.render(line, True, (210, 210, 225))
            content.blit(line_surf, (18, y))
            y += line_h
        y += section_gap

    screen.set_clip(viewport)
    screen.blit(content, (viewport.x, viewport.y - help_scroll))
    screen.set_clip(None)

    max_scroll = _help_max_scroll()
    if max_scroll > 0:
        track = pygame.Rect(viewport.right + 8, viewport.y, 6, viewport.h)
        pygame.draw.rect(screen, (40, 35, 55), track, border_radius=3)
        thumb_h = max(24, int(viewport.h * viewport.h / _help_content_height()))
        thumb_y = viewport.y + int((viewport.h - thumb_h) * (help_scroll / max_scroll))
        pygame.draw.rect(screen, (0, 255, 200), (track.x, thumb_y, track.w, thumb_h), border_radius=3)
        hint_text = "Up/Down, mouse wheel, or swipe to scroll" if touch_controls.ever_touched \
            else "Up/Down or mouse wheel to scroll"
        hint = small_font.render(hint_text, True, (150, 150, 170))
        screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, 393)))

    back_surf = small_font.render(SHOP_BACK_PROMPT, True, (200, 200, 220))
    screen.blit(back_surf, SHOP_BACK_RECT)
    draw_fps_overlay()

    pygame.display.flip()


# ---------------- pause menu ----------------
def draw_paused():
    """Draws the last real gameplay frame (already on screen from before Esc was
    pressed) dimmed under a modal button stack. Deliberately does NOT call
    update_and_draw_playing() or advance any game state -- see game_loop(), which skips
    the gameplay update entirely while state == "paused", so the world truly freezes."""
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((10, 8, 16, 180))
    screen.blit(overlay, (0, 0))

    draw_glow_text(screen, "PAUSED", title_font, (SCREEN_W // 2, 70), (0, 255, 220))

    mouse_pos = pygame.mouse.get_pos()
    draw_menu_button(PAUSE_SHAKE_TOGGLE_RECT, "Shake", PAUSE_SHAKE_TOGGLE_RECT.collidepoint(mouse_pos),
                      sub_label="ON" if screen_shake_enabled else "OFF")
    draw_menu_button(PAUSE_RESUME_RECT, "RESUME", PAUSE_RESUME_RECT.collidepoint(mouse_pos),
                      primary=True, font_obj=font, sub_label="Esc / R")
    draw_menu_button(PAUSE_MUTE_TOGGLE_RECT, "Mute", PAUSE_MUTE_TOGGLE_RECT.collidepoint(mouse_pos),
                      sub_label="ON" if muted else "OFF")
    draw_menu_button(PAUSE_HELP_RECT, "Help", PAUSE_HELP_RECT.collidepoint(mouse_pos))
    draw_menu_button(PAUSE_SETTINGS_RECT, "Settings", PAUSE_SETTINGS_RECT.collidepoint(mouse_pos))
    draw_menu_button(PAUSE_MAIN_MENU_RECT, "Return to Main Menu", PAUSE_MAIN_MENU_RECT.collidepoint(mouse_pos))
    draw_menu_button(PAUSE_QUIT_RECT, "Quit Game", PAUSE_QUIT_RECT.collidepoint(mouse_pos))

    hint = small_font.render(f"Level {level_num}  |  Score: {score}  |  Coins: {wallet}", True, (150, 150, 170))
    screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, 412)))

    touch_controls.draw(screen)  # just the Pause/Resume toggle -- see _buttons_for_state()

    pygame.display.flip()


def draw_paused_settings():
    global fps_cap, fullscreen, screen, show_fps
    # Applied straight from the live widget every frame (same pattern draw_menu() uses
    # for music/SFX volume) so dragging/typing a new FPS Cap takes effect on the very
    # next rendered frame -- no separate "Apply" step, no restart needed.
    fps_cap = fps_cap_control.value
    show_fps = toggle_show_fps.value
    if toggle_mute.value != muted:
        set_muted(toggle_mute.value)
    if toggle_fullscreen.value != fullscreen:
        # Same mode switch F11 does (see handle_events()) -- just reachable by touch/
        # mouse now instead of only a keyboard shortcut.
        fullscreen = toggle_fullscreen.value
        flags = pygame.SCALED | (pygame.FULLSCREEN if fullscreen else 0)
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)

    screen.blit(MENU_BG, (0, 0))
    draw_glow_text(screen, "SETTINGS", title_font, (SCREEN_W // 2, 60), (0, 255, 220))

    toggle_shake.draw(screen, small_font)
    slider_volume.draw(screen, small_font)
    slider_sfx.draw(screen, small_font)
    fps_cap_control.draw(screen, small_font)
    toggle_fullscreen.draw(screen, small_font)
    toggle_show_fps.draw(screen, small_font)
    toggle_mute.draw(screen, small_font)
    if touch_controls.ever_touched:
        touch_controls.opacity = slider_touch_opacity.value
        touch_controls.size = TOUCH_SIZE_MIN + slider_touch_size.value * (TOUCH_SIZE_MAX - TOUCH_SIZE_MIN)
        touch_controls.swap_sides = toggle_touch_swap.value
        slider_touch_opacity.draw(screen, small_font)
        slider_touch_size.draw(screen, small_font)
        toggle_touch_swap.draw(screen, small_font)

    back_surf = small_font.render("PRESS ESC OR CLICK TO GO BACK", True, (200, 200, 220))
    screen.blit(back_surf, SHOP_BACK_RECT)
    draw_fps_overlay()

    pygame.display.flip()


def _snapshot_render_positions():
    """Records where every rendered sprite (plus the player and camera) is *about* to
    move from, right before a fixed tick runs. draw_playing_frame() blends between this
    and the post-tick position it ends up at, so a rendered frame that lands between two
    ticks (any time the FPS cap is above SIM_HZ) shows smooth in-between motion instead
    of holding the same simulated position until the next tick lands. Purely a rendering
    concern -- stored as an extra attribute on each sprite, never read by any gameplay
    code, and never influences collision/physics.

    Covers every group drawn via camera.apply() in draw_playing_frame() -- static groups
    (checkpoints, wind_zones) are harmless no-ops here, and it's one less thing to
    remember to touch if that ever changes."""
    for group in (platforms, wind_zones, one_way_platforms, checkpoints, spikes, coins, enemies, projectiles):
        for spr in group:
            spr._prev_topleft = spr.rect.topleft
    player._prev_topleft = player.rect.topleft
    camera._prev_x = camera.x


def _lerp(a, b, t):
    return a + (b - a) * t


def _interp_rect(spr, alpha):
    """spr.rect blended `alpha` (0..1) of the way from its pre-tick position (see
    _snapshot_render_positions()) to its current, already-simulated position. Falls back
    to the exact current rect for alpha>=1 (the common case: a rendered frame that lands
    exactly on a tick boundary, or any direct call with no interpolation requested -- see
    update_and_draw_playing()) and for a sprite with no snapshot yet (spawned this tick,
    e.g. a SplitterEnemy's shards or a freshly fired projectile -- it simply appears at
    its true spawn position instead of sliding in from a stale/missing one)."""
    if alpha >= 1.0:
        return spr.rect
    prev = getattr(spr, "_prev_topleft", None)
    if prev is None:
        return spr.rect
    cx, cy = spr.rect.topleft
    r = spr.rect.copy()
    r.topleft = (round(_lerp(prev[0], cx, alpha)), round(_lerp(prev[1], cy, alpha)))
    return r


def _update_playing_tick():
    """Exactly one fixed simulation tick (1/SIM_HZ of game time) for the "playing"/
    "level_complete" states -- gameplay logic, autosave, and particle/popup upkeep. Called
    once per tick by run_fixed_updates() (zero or more times per rendered frame -- see the
    fixed-timestep comment near SIM_HZ) and exactly once by update_and_draw_playing() for
    direct/standalone callers (every existing test)."""
    global state, transition_start, particles, popups

    if state == "playing":
        player.update(platforms, spikes, enemies, coins, one_way_platforms, checkpoints, wind_zones)
        enemies.update(platforms, player.rect)
        coins.update()
        spikes.update()  # no-op for plain Spike/HazardPool; moves MovingSpike/TimedSpike/GasVent
        # no-op for Platform/OneWayPlatform/InvisibleWall/IceFloor (pygame.sprite.Sprite's
        # default update() ignores whatever args it's given); only CrumblingPlatform
        # actually reads player.rect here, to run its stand-on-it/shake/collapse cycle.
        platforms.update(player.rect)
        projectiles.update(platforms)
        if player.vel.x != 0 or not player.on_ground:
            maybe_spawn_trail_particles(player.rect, player.state)
        if goal_trigger_rect and player.rect.colliderect(goal_trigger_rect):
            state = "level_complete"
            transition_start = _now_ms()
            snd_win.play()
    elif state == "level_complete":
        if _now_ms() - transition_start > LEVEL_TRANSITION_MS:
            if is_replay:
                end_replay()  # restores the player's real current level; never touches save progress
                state = "replay_picker"
            else:
                advance_level()
                state = "playing"

    maybe_autosave()

    particles = [p for p in particles if p.alive()]
    for p in particles:
        p.update()
    popups = [p for p in popups if p.alive()]

    camera.update(player.rect)


def draw_playing_frame(alpha=1.0):
    """Renders the current "playing"/"level_complete" state. `alpha` (0..1, from
    run_fixed_updates()) is how far the leftover accumulator already is toward a next,
    not-yet-simulated tick -- see _interp_rect(). alpha=1.0 (the default, used by every
    direct/standalone caller) means "exactly the latest simulated position," identical to
    this function's pre-interpolation behavior."""
    real_cam_x = camera.x
    if alpha < 1.0:
        # Temporarily override camera.x to the interpolated value for the duration of
        # this draw -- every draw call below (including Particle/Popup.draw(), which read
        # camera.x directly rather than going through camera.apply()) picks it up for
        # free, and it's restored in the `finally` below so next tick's camera.update()
        # still lerps from the true, non-interpolated position.
        camera.x = _lerp(getattr(camera, "_prev_x", real_cam_x), real_cam_x, alpha)
    try:
        background.draw(screen, camera.x)
        for spr in platforms:
            screen.blit(spr.image, camera.apply(_interp_rect(spr, alpha)))
        for spr in wind_zones:
            screen.blit(spr.image, camera.apply(_interp_rect(spr, alpha)))
        for spr in one_way_platforms:
            screen.blit(spr.image, camera.apply(_interp_rect(spr, alpha)))
        for spr in checkpoints:
            screen.blit(spr.image, camera.apply(_interp_rect(spr, alpha)))
        for spr in spikes:
            screen.blit(spr.image, camera.apply(_interp_rect(spr, alpha)))
        for c in coins:
            screen.blit(c.image, camera.apply(_interp_rect(c, alpha)))
        for e in enemies:
            screen.blit(e.image, camera.apply(_interp_rect(e, alpha)))
        for shot in projectiles:
            screen.blit(shot.image, camera.apply(_interp_rect(shot, alpha)))
        if goal_rect:
            r = camera.apply(goal_rect)
            pygame.draw.rect(screen, (150, 120, 60), (r.centerx - 3, r.y, 6, r.h))
            pygame.draw.polygon(
                screen, GOAL_COLOR,
                [(r.centerx + 3, r.y), (r.centerx + 19, r.y + 8), (r.centerx + 3, r.y + 16)],
            )

        now = _now_ms()
        player_render_rect = _interp_rect(player, alpha)
        player_visible = True
        if player.state == "dead":
            player_visible = (now // 100) % 2 == 0
        elif now < player.invincible_until:
            # quick flash while the post-shield-hit grace period is active, so it's obvious
            # you got hit (and briefly can't be hit again) instead of the damage being silent
            player_visible = (now // 80) % 2 == 0
        if player_visible:
            img_rect = player.image.get_rect(midbottom=player_render_rect.midbottom)
            screen.blit(player.image, camera.apply(img_rect))
        if player.state == "dead":
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((200, 30, 30, 80))
            screen.blit(overlay, (0, 0))

        for p in particles:
            p.draw(screen, camera)
        for p in popups:
            p.draw(screen, camera, small_font)
        draw_enemy_radar(screen, enemies, player.rect, camera, spikes, platforms)

        hud_y = 10
        score_surf = draw_hud_text(screen, f"Score: {score}", (10, hud_y))
        hud_y += score_surf.get_height() + 6
        coins_surf = draw_hud_text(screen, f"Coins: {wallet}", (10, hud_y), color=COIN_COLOR)
        hud_y += coins_surf.get_height() + 6
        if player.shield_max > 0:
            shield_surf = draw_hud_text(
                screen, f"Shield: {player.shield_charges}/{player.shield_max}", (10, hud_y), color=(120, 200, 255)
            )
            hud_y += shield_surf.get_height() + 6
        if player.hearts_max > 0:
            hearts_surf = draw_hud_text(
                screen, f"Hearts: {player.hearts}/{player.hearts_max}", (10, hud_y), color=(255, 110, 110)
            )
            hud_y += hearts_surf.get_height() + 6
        if upgrades["dash"] > 0:
            cooldown_left = max(0, player.dash_ready_time - now)
            dash_color = (120, 220, 255) if cooldown_left == 0 else (110, 110, 130)
            dash_text = "Dash: Ready" if cooldown_left == 0 else f"Dash: {cooldown_left / 1000:.1f}s"
            dash_surf = draw_hud_text(screen, dash_text, (10, hud_y), color=dash_color)
            hud_y += dash_surf.get_height() + 6
        if muted:
            draw_hud_text(screen, "MUTED", (10, hud_y), color=(255, 120, 120))
        top_right_y = draw_fps_overlay()
        level_label = f"REPLAY: Level {replay_level}" if is_replay else f"Level {level_num}"
        level_surf = small_font.render(level_label, True, (255, 190, 90) if is_replay else WHITE)
        draw_hud_text(screen, level_surf, (SCREEN_W - level_surf.get_width() - 10, top_right_y))
        if touch_controls.effective_alpha() < 0.05:
            # The touch overlay explains itself visually -- this legacy keyboard hint would
            # otherwise sit right underneath the touch buttons drawn below.
            hud_surf = small_font.render(
                "Move: Arrows/WASD  Jump: Space/Up  Shift: Dash  Down: Glide  M: Mute  F3: FPS  F11: Full  Esc: Menu",
                True, WHITE,
            )
            draw_hud_text(screen, hud_surf, (10, SCREEN_H - hud_surf.get_height() - 8))

        if state == "level_complete":
            text_surf = font.render(f"Level {level_num} complete!", True, WHITE)
            draw_hud_text(screen, text_surf, text_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)).topleft,
                           pad=10)

        touch_controls.draw(screen)  # no-op unless state == "playing" -- see _buttons_for_state()

        pygame.display.flip()
    finally:
        camera.x = real_cam_x


def update_and_draw_playing(alpha=1.0):
    """Runs exactly one fixed tick, then draws -- the pre-fixed-timestep behavior, kept
    as the direct/standalone entry point every existing test (and any other one-off
    caller) already relies on. game_loop() itself calls _update_playing_tick() (zero or
    more times, via run_fixed_updates()) and draw_playing_frame() separately instead of
    going through this, so a real session's tick rate stays decoupled from its render
    rate -- see the fixed-timestep comment near SIM_HZ."""
    _snapshot_render_positions()
    _update_playing_tick()
    draw_playing_frame(alpha)


def run_fixed_updates(frame_time_ms):
    """Advances the fixed-tick simulation to absorb `frame_time_ms` of newly-elapsed real
    time, running _update_playing_tick() zero or more times -- see the fixed-timestep
    architecture comment near SIM_HZ. Returns the render alpha (0..1) draw_playing_frame()
    should interpolate with: how far the leftover accumulator already is toward a next,
    not-yet-simulated tick.

    frame_time_ms is clamped to MAX_CATCHUP_TICKS worth of ticks before being added to
    the accumulator -- the classic fixed-timestep "spiral of death" guard, so a huge
    one-off stall (window drag, breakpoint, a slow asset load) can't make the game try to
    frantically replay minutes of missed time; any backlog beyond the cap is simply
    dropped, same as pausing briefly would look from the player's perspective.

    Stops early if a tick changes `state` away from "playing"/"level_complete" (e.g.
    reaching the goal, and then the level-complete banner expiring into the replay
    picker, both within the same rendered frame's catch-up burst) -- whatever's left in
    the accumulator simply carries over, harmless since it's always well under one
    rendered frame's worth of real time."""
    global _accumulator_ms, _tick_time_offset_ms

    _accumulator_ms = min(_accumulator_ms + frame_time_ms, MAX_CATCHUP_TICKS * SIM_DT_MS)
    ticks_run = 0
    # `- _TICK_EPSILON_MS`: repeatedly subtracting SIM_DT_MS (itself 1000/60, not exactly
    # representable in binary) accumulates float error, which can leave the accumulator a
    # few 1e-13ms *short* of a full tick that mathematically already completed --
    # dropping it entirely would silently lose ~1/60s of simulated time. The epsilon is
    # many orders of magnitude below any real timing granularity (clock.tick() reports
    # whole milliseconds), so it never causes an extra tick to run early.
    while _accumulator_ms >= SIM_DT_MS - _TICK_EPSILON_MS and state in ("playing", "level_complete"):
        if ticks_run > 0:
            # Space this tick's _now_ms() reads SIM_DT_MS apart from the previous one in
            # this same burst -- see _now_ms()'s docstring for why that matters.
            _tick_time_offset_ms += SIM_DT_MS
        _snapshot_render_positions()
        _update_playing_tick()
        _accumulator_ms -= SIM_DT_MS
        ticks_run += 1
    return _accumulator_ms / SIM_DT_MS


def game_loop(max_frames=None):
    """The main loop. With max_frames=None this runs until the player quits (normal
    play); with a number, it stops after that many frames regardless -- used by the
    --smoke-test CLI mode and by tests to run a bounded slice of real gameplay.

    Rendering happens once per iteration of this loop, paced by fps_cap (see Settings;
    0 means uncapped) via clock.tick(). Gameplay simulation is entirely decoupled from
    that -- see run_fixed_updates()/_update_playing_tick() -- so how fast this loop spins
    only ever changes visual smoothness and input latency, never game speed. "frame" below
    always means one iteration of this loop (one handle_events() + one render), the same
    thing max_frames counts and always has -- not one fixed simulation tick, which may
    happen zero, one, or several times within a given frame."""
    global running, _accumulator_ms, _tick_time_offset_ms

    # Fresh session: neither a stale reading from however long ago the caller last
    # touched `clock` (harmless either way -- run_fixed_updates() clamps it -- but no
    # reason to carry it forward) nor a leftover catch-up offset should bleed in.
    clock.tick()
    _accumulator_ms = 0.0
    _tick_time_offset_ms = 0.0

    frame_count = 0
    while running and (max_frames is None or frame_count < max_frames):
        frame_time_ms = clock.tick(fps_cap)
        frame_count += 1

        touch_controls.update(frame_time_ms)
        handle_events()

        if state == "menu":
            draw_menu()
            continue
        if state == "shop":
            draw_shop()
            continue
        if state == "trail_menu":
            draw_trail_menu()
            continue
        if state == "replay_picker":
            draw_replay_picker()
            continue
        if state == "help":
            draw_help()
            continue
        if state == "paused":
            draw_paused()  # deliberately no gameplay update -- the world is frozen
            continue
        if state == "paused_settings":
            draw_paused_settings()
            continue

        alpha = run_fixed_updates(frame_time_ms)
        if state in ("playing", "level_complete"):
            draw_playing_frame(alpha)
        # else: a tick this frame moved to some other state (see run_fixed_updates()'s
        # docstring) -- skip drawing this iteration; the next one's dispatch above
        # handles it, at most one frame (well under a millisecond at any real FPS cap)
        # later than it otherwise would have.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Initialize the game, run a few frames of real gameplay headlessly, then exit. For CI/testing.",
    )
    parser.add_argument(
        "--smoke-test-frames", type=int, default=90,
        help="Number of frames to simulate in --smoke-test mode (default: 90).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    init_pygame()
    init_sounds()
    init_sprites()
    if not args.smoke_test:
        # Skipped in --smoke-test mode -- it must stay deterministic/non-interactive for
        # CI regardless of how many save*.txt files happen to sit next to main.py (e.g.
        # a developer's own working copy while testing this very feature).
        choose_save_file()
    init_game_state()

    try:
        if args.smoke_test:
            start_game()  # jump straight into "playing" so the smoke test exercises real gameplay, not just the menu
            game_loop(max_frames=args.smoke_test_frames)
            print(f"Smoke test OK: ran {args.smoke_test_frames} frames without crashing "
                  f"(level {level_num}, score {score}, coins {wallet}).")
        else:
            game_loop()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
    if "__compiled__" in globals() and sys.platform == "win32" and "--smoke-test" not in sys.argv:
        # Keeps a double-clicked packaged .exe's console window open instead of
        # vanishing instantly, so a fatal-error message can actually be read -- that's
        # the ONLY scenario this is for (hence gating on __compiled__, the same Nuitka
        # marker _app_dir() checks). A plain `python main.py` run -- every dev/
        # interactive session, and any touchscreen device with no physical keyboard to
        # send Enter with -- would otherwise sit here forever after the game window has
        # already closed, looking exactly like the process hung.
        input("Press Enter to exit...")
