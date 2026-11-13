# Ocarina Dynamic FPS add-on

`Dynamic FPS by Dark (built-in)` is a game Add-On, enabled by default for Ocarina.
It targets title `0400486024DEC000`, module `main`, build ID
`E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348` only. Max Resolution remains a
separate, default-on Add-On. Either can be disabled independently in Add-Ons;
the existing disabled-add-on configuration persists that choice.

## Credit and source

Dark, an Eglantine developer, wrote the routine. This port preserves the call and
120-byte routine from Frizti's `oot-gather-fix` commit
`765659af4ccb7201d1145c57f3b45ce84653cea0`. That commit explicitly states that
Dark gave permission to bundle it. Frizti supplied the built-in integration.
CTGL's #299 changed one instruction: the delta cap is `CNTFRQ / 10` instead of
`CNTFRQ / 20`.

The routine's SHA-256 is
`0afb4af10a78bb49372e6807fc5b71d9fa99d828b7625f0fab7429fc66a4b1e6`
(`3c15dfb57f748328ad7d1c1675941dc2ace32a5e7b10907368ceac6a9a8c5ee1` before the
cap change).
The call's SHA-256 is
`e4d3f65ad1b568b557f02b9a087a9c304240ac70d6ef474dac83a1f97737515e`.
This port adds original-byte and memory-layout guards, and
transactional validation before writing either record.

## Verified original layout

An independently extracted original `main.text` has SHA-256
`f71eb7b75b11dcc8d955f3f6e987a098573288d9ce41aa22a96272541e858c5f`, matching
the text hash in its NSO header. Its instruction at module offset `0x6E6CC4`
is `28 18 20 1E`, or `fdiv s8, s1, s0`. The patch replaces it with
`bl 0x3C46200` and executes the original division first in the new routine.
The enclosing function saves and restores its return address. The continuation
does not consume the routine's clobbered scratch registers or flags before
intervening calls or redefinition.

The original text ends at `0x3C461D0`; read-only data starts at `0x3C47000`.
The routine occupies `0x3C46200` through `0x3C46277`, in executable alignment
padding, not original text. `AppLoader_NSO::LoadModule` zero-initializes the
module image before copying segments and rounds the text mapping to a page.
The loader checks that this space is still zero-filled executable padding
before installing the routine.

Data ends at `0x4D46B80` and BSS ends at `0x4EA0000`. The routine stores its
previous timestamp and smoothed delta at `0x4E9FE00` and `0x4E9FE08`. The
loader requires both words to fit BSS and remain zero-initialized. These
checks establish addressability and initialization, not exclusive ownership
of that storage throughout gameplay.

Any changed user-mod bytes in either patch record or the timer state take
precedence over the entire built-in patch. An incompatible build, original
instruction, layout, or truncated module leaves both patch records untouched.

## Timing limits

The routine reads `CNTPCT_EL0` and `CNTFRQ_EL0`. After its first call, it caps
the timestamp difference at `CNTFRQ / 10`, or 100 ms, then smooths it by adding
one eighth of the signed difference from the previous smoothed value.
It calculates `s2 = (smoothed_ticks / CNTFRQ) * s1`. It replaces the original
`s8 = s1 / s0` only when `s2 > 1.03 * s8`.
The original code selects `s1` as 30 or 60 and converts the integer field at
`x21 + 0x2C8` into `s0`, so the result is a frame-step scale, not seconds.

This is not a promise of correct game speed at every frame rate. It retains
the original step as a floor, caps long frames, and smooths changes. Actual
gameplay speed and the runtime value of that frame-rate field still require
live validation.

In Lichii, `CoreTiming::GetClockTicks` scales the counter for CPU Boost and
Overclock, while Dynarmic's advertised counter frequency stays unchanged.
Boost uses a 1.7 multiplier and Overclock uses 2. Enabling core-speed
synchronization can further divide the counter by the configured speed limit.
Those settings alter the elapsed seconds this routine calculates. CPU Clock
Normal and an unscaled counter are prerequisites for evaluating its timing;
the Add-On does not change global clock behavior or automatically override
the user's settings. Correct gameplay speed with the Add-On is unverified.

## FPS Unlock step counters

`FPS Unlock (built-in)` also corrects four counters through a shared fractional
step accumulator. `Dynamic FPS by Dark (built-in)` keeps its
original records and does not install these fixes. Addresses below are `main`
module offsets.

| Hook | System | Without the fix | Fix |
|---|---|---|---|
| `0xA68300` | Pickup window count-up | Truncating a step below one stalls the displayed count. | Advance by accumulated whole ticks. |
| `0x3293410` | Ocarina playback tick | Truncating a step below one stalls playback. | Advance by accumulated whole ticks. |
| `0x2F53268` | Step rate manager countdown | `max(round(step), 1)` per frame runs fast above 60 fps. | Decrement by accumulated whole ticks. |
| `0x53F914` | Ambient wave phase | Counting frames makes the phase run fast above 60 fps. | Accumulate the raw step instead of adding one. |

The helper takes the counter address in `x13` and step in `s0`, returning ticks
in `w13`. Each of 16 slots retains a fractional remainder. A new or evicted
slot returns `max(trunc(step), 1)`; reused addresses inherit the stored fraction.
More than 16 interleaved counters can therefore fall back to that minimum.

The stubs occupy `0x3C46900` (40 bytes), `0x3C46928` (40 bytes), `0x3C46950`
(48 bytes), and `0x3C46A00` (60 bytes); the 128-byte helper starts at
`0x3C46980`. All are guarded zero-filled executable padding. The hitstop hook
validates all 44 original bytes through `0x2F53294`, including the instructions
the stub skips. Any changed hook, cave, state byte, or user mod in a record
skips the entire FPS Unlock patch transactionally.

FPS Unlock reserves `0x1C0` bytes from `0x4E9FE00`. Its step state occupies
`0x4E9FEB0..0x4E9FFC0` (end exclusive): a replacement index at `+0`, then 16
slots of `{u64 counter address, f32 remainder, pad}` at `+0x10`. The range
`0x4E9FE10..0x4E9FEB0` stays free for the separate open water PR #1180 (state
through `+0x98`, cave `0x3C46300`) and sound PR #1289 (state through `+0xA8`,
caves `0x3C46280` and `0x3C46800`). Those patches are not included here.
Zero initialization and non-overlap do not establish exclusive BSS ownership
throughout gameplay.

## FPS Unlock ground velocity

The character controller compares a frame-rate value with `100.0` in three places; above it
(`b.gt` at `0x13A070`, `0x13D710` and `0x389AC2C`) it scales its ground's velocity by `0.1`
(`0x3E40FB4`). Over 100 fps a moving platform then slides away under the player.
`FPS Unlock (built-in)` replaces the three branches with `nop`. `Dynamic FPS by Dark (built-in)`
never exceeds 60 fps and leaves them alone. Measured on the Deku Tree B1 raft: the applied
velocity was 0.100 of the ground's at steps below 0.6 and 1.000 at steps of 0.627 and above.

## Tests

The `[patch_manager]` tests call production `PatchManager::PatchNSO` with
synthetic module images and real IPS/IPS32 user mods. They cover default-on and
disable behavior, title/module/build gates, original-byte and layout
guards, truncated inputs, all-record user-mod precedence, and Max Resolution.
The `[addon_config]` tests round-trip the production configuration reader and
writer, including legacy configurations and repeated enable/disable cycles.

Run them from a clang/Ninja test build with:

```sh
cmake --build build --target tests --parallel 4
build/bin/tests '[patch_manager],[addon_config]'
```
