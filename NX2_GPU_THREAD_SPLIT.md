# Splitting the GPU thread

Title 1 (Mario Kart World) is bound by one host thread, the GPU thread: ~100% busy while the
host GPU is 20-40% busy and the game's own CPU thread mostly polls the counter. Per second of
3D scenes (profiled, 2026-10-29): command processing and register writes ~335 ms, macros
~200 ms (1.3 million calls), draw setup ~345 ms (55,000 draws), compute ~70 ms. Speeding up the
pieces (NX2_NOTES rows 123-139) took frames from ~22-38 ms to ~18-26 ms; a locked 60 fps on the
heavy courses needs work on more than one core.

## Split point: method calls

- **Front thread** (the current GPU thread): takes the CPU's command queue, parses GPFIFO
  entries and command lists (DmaPusher), runs macros (compiled, macro_turing_jit.h), and keeps
  what macros read: a mirror of each engine's registers, its macro code and data RAM.
- **Back thread**: executes the resulting stream of method writes in order: Maxwell3D /
  KeplerCompute / Fermi2D / DMA engines' CallMethod and everything behind them (register
  state, dirty flags, constant buffer and inline uploads, draws, dispatches, queries, fences,
  the rasterizer and its caches). Only the back thread touches the rasterizer.
- The stream: an SPSC ring of {engine, method, argument} plus copied data for multi-word
  methods (command lists are often guest memory read in place, which the guest reuses).
  CPU-side commands (flush/invalidate requests, RunInOrder actions, presentation) are
  forwarded into the same ring so they keep their order.

## What the front needs without waiting

- Register values macros read (MME_READ): the front mirrors every register write it sends.
  Back-side changes to registers it must reproduce: constant buffer offset (+4 per
  const_buffer.buffer word), shadow_scratch[0] = 1 (falcon[4]), shadow RAM control (Track /
  Replay change the effective value), macro upload pointers (the front handles load_mme
  itself). Compute: native_registers (0xad is changed temporarily by the inline QMD path).
- Macro code uploads and binds (load_mme.*, compute 0x45-0x48): handled by the front.

## Where the front waits for the back (sync points)

- Macro DMA reads (MME_DMA_READ / _FIFOED, ~450 a second): read guest memory as of their
  position, so the back must have executed everything before them.
- Semaphore acquires that are not satisfied yet (~10,500 a second, cross-channel waits on QMD
  releases): the release is executed by the back, so the front waits until the semaphore
  holds (or parks the channel and runs another one, as the pending-channel code does now).
- Syncpoint waits, WaitForIdle-style methods, and anything else that reads guest memory or
  host state on the front.
- Expected cost: each wait drains the back only up to the dependency, then the back idles
  until the front sends the next method (a few microseconds), so even ~10,000 waits a second
  leave most of the overlap. Ceiling: front ~35-40% of the work, back ~60-65% -> ~1.4-1.6x.

## Stages

1. Instrument: count sync points per second and the front/back share precisely (zones).
2. Ring + back thread executing method writes; the front still does everything else.
   Correctness first: an EDEN_NX2_SPLIT=0 switch runs both on one thread.
3. Move macros to the front with the register mirrors; verify mirrors against the back's
   registers in a debug mode (compare on every MME_READ).
4. Sync points; crash and screenshot tests across courses; then title 4.

## Status (2026-10-29)

Stages 1-3 are in (babd395, `gpu_split.h`), off by default: `EDEN_NX2_SPLIT=1` runs engine methods
on the back, `EDEN_NX2_SPLIT=2` also runs the 3D and compute macros on the front against register
copies. Measured in title 1 (NX2_NOTES rows 142-144): ~44,000 draws a second on one thread, ~47,400
with stage 2, 50,000-66,500 with stage 3. The front waits ~4,000-5,000 times a second (semaphore
acquires, MME DMA reads); the back still spends ~200 ms a second outside its zones (ring dispatch,
method sink). Clean A/B (NX2_NOTES row 145): 1.46x throughput, frames 17% shorter; on by
default since. The back is ~92% busy, ~53% of it draw setup: the next step is there, not the ring.

## Fast draw ports on the #805 performance branch (2026-11-09)

The port of nx2 `1c25fbdfb4` (#921) makes full resolves and texture memo hits use
the same image-preparation predicate. Empty descriptor views backed by the null
image are excluded even when their view id is not the null view id. Delta draws
fall back when a sampled image needs an upload or alias synchronization, or when
the draw's serial, mappings, command buffer or render pass changed before the
caches were locked. Rescaling ends the chain too.

Only the uniform-keep and image-descriptor replay portions of nx2 `a4f6287327`
(#949) are ported: `6b38851501`, `ee98215f14`, and the CPU-write follow-up
`3e86407514`. The test-runtime support in `b2375cff7f` already exists on #805.
Its semaphore, waiter and acquire/drain replacements are not included; #805
retains its own implementations. The subsequent nx2 merge keeps the independent
scheduler-owner guard on buffer-cache writes from CPU/service threads, without
replacing #805's back-owned `Run` operations with nx2's `back_quiet` drain scheme.

Small streamed uniforms can reuse the previous allocation without rereading
guest memory when the stage, slot, device range and command-buffer tick still
match, and no overlapping inline data, notified CPU write, GPU write or buffer
deletion invalidated it. More than eight notified writes between draws disable
keeps for the next draw. Unsynchronized CPU writes unknown to the caches remain
an assumption to check on the game. Per-stage image descriptor entries and
rescaling words are replayed until the views are recaptured or the scale epoch
changes; buffer descriptors still bind on every draw.

Validate a fixed scene with these diagnostics separately (set each to `1`):

- `EDEN_NX2_TEXTURE_MEMO_VERIFY`: compare views, samplers and prepared image sets.
- `EDEN_NX2_FAST_DRAW_VERIFY`: full-resolve comparison; this deliberately bypasses
  delta draws, so it does not exercise uniform keeps or descriptor replay.
- `EDEN_NX2_UBO_KEEP_VERIFY`: reread would-be kept uniforms and report changed bytes.
- `EDEN_NX2_DELTA_DESCRIPTORS_VERIFY`: regenerate and compare image descriptors
  and rescaling words.
- `EDEN_NX2_GPU_PROFILE`: report keep/replay counts and draw fallback reasons;
  value `2` also retains #805's fine uniform-binding counters.

After building the test target, run `tests.exe "[buffer_ports]"`,
`tests.exe "[video_core][texture_cache]"`, and
`tests.exe "Uniform stream*"`. These ports were reviewed without building on
the porting machine; compilation, runtime verification and visual A/B testing
are still required. #949's reported Bananza gain (+8–9%, back thread about
56.0 to 52.2 ms/frame on a Ryzen 9 5900X) includes its acquire changes and does
not predict the gain of this partial port in Ocarina of Time.

## nx2 integration through 1a5e995a38

- #1028 does not replace #805's read downloads, write tracking, sampled method
  profiling or diagnostics. Its correctness fixes are retained: discard/DMA
  destinations first land overlapping read downloads, and older deferred query
  results cannot mark newer GPU writes clean. Synced #874 payload callbacks still
  do not advance `fenced_landed` or retire an older deferred report's tracking;
  `FencedPayloadLanding` remains authoritative.
- #979/#1014 retain every registered semaphore region and the direct-release
  shadow. Mirrored releases are emitted only when `ViewApplies()` is false
  (unsplit, back macros, or delayed/high-accuracy releases). The full #805 front
  view continues to decide from the original releases, never a second mirrored
  counter. Its unbounded, source-tracking waiter collection is retained.
- Shader cache v2 replaces the legacy Vulkan cache. #970's no-flush TIC reads
  apply to worker translation too, so no pool-wide flush is introduced before
  launching workers. As with #970, pending GPU writes to descriptors remain a
  runtime validation risk. AMD shader exceptions, the image recycler and memory
  usage caching are unchanged.
- #965/#1039 add stutter sites, GPU waits by zone, queue/present waits, contention
  and opt-in live-read comparisons around the existing split. Method profiling
  still takes the fast path and samples timing, rather than timing every write.
- #1056's macro-on-back port arrived during integration. Keep #805's full effect
  analysis and back-macro machinery, but take the review fixes: recheck command
  pages even when the writer wait did not block, and remember the actual last
  writer at each host sync rather than waiting through unrelated intervening work.

In-game validation should cover low/default and high/delayed fence modes,
cross-channel semaphore waits, concurrent CPU read-back, cold/migrated/warm v2
caches, and the fast-draw checks above. The integration build disables
`BUILD_TESTING`; the updated C++ regression sources must be run separately in a
test-enabled build.
