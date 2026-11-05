# Qwen3.5-4B in 3D

A 3D walkthrough of one real forward pass of Qwen3.5-4B, in the spirit of
[llm-viz](https://bbycroft.net/llm). Tailnet URL: https://jimmy-dgx-spark.tailbac7f9.ts.net:8444

This started as a fork of Brendan Bycroft's [llm-viz](https://github.com/bbycroft/llm-viz) (MIT, see
`LICENSE`). The WebGL renderer in `src/llm/render` and the camera are his, with shader changes. The layout,
data pipeline and walkthrough are new, because the model is different: 24 of its 32 layers use Gated DeltaNet,
which has no attention matrix.

## How it fits together

- `tracer/trace.py` runs the model and writes `public/q35/`.
- `src/qwen/data.ts` loads those files, one layer at a time.
- `src/qwen/layout.ts` builds one small scene per section. To change a figure, start here.
- `src/qwen/steps.tsx` holds the text and captions. Numbers in the text are computed from the loaded trace, so
  they stay correct for every prompt and layer; don't type results in by hand.
- `src/qwen/program.ts` is the frame loop, camera framing, arrows and hit testing.
- `src/qwen/View.tsx` is the page, the input handling and the inspector.

## Writing and figure conventions

The text follows the style of Long, Sebo et al. (2026) and Gurnee et al. (2026). Each section asks a question or
states a claim, then gives the method, the result and what was not shown. Interpretation is labelled as such, and
derived quantities are labelled as derived. Each figure makes one point and has a numbered caption that says what
is plotted. A signed value uses a blue-to-red scale and a value in [0, 1] uses a light-to-dark ink scale.

The default view is 3D, with thin weight sheets and thicker computed blocks; a control switches to a front view.
Every stored weight must appear in some figure. If a figure becomes too dense to label, split it rather than
dropping a weight.

## What is real and what is sampled

- Activations are stored for every dimension at 8 bits per value (scaled per tensor), so each is within
  `max|x| / 254` of the true value. The inspector shows all of them.
- The 3D view shows one value in 40 on long axes, so blocks stay to scale. Each cell is an exact stored value,
  not an average.
- Weights are stored only at that 1-in-40 sampling. Their mean, std and range cover the full matrix.
- The memory `S` is shown at one value in 8 per axis. The 128 x 128 originals are not stored.
- `read[t, s]`, shown in linear layers, is derived by this project. The model never computes it.
- Arrows show the main path through a layer. To keep them readable, not every input has one: the normed input
  also feeds `in_proj_a`, `in_proj_b`, `in_proj_z`, `k_proj` and `v_proj`.
- `mtp.*` (0.12B parameters in the checkpoint) isn't loaded by the model code, so it isn't shown or counted.
  The vision tower isn't shown because text prompts don't use it.

## Checks the tracer enforces

The run aborts if any of these fail:

- Logits with capture hooks are exactly equal to a clean run.
- Logits rebuilt from the stored final norm are exactly equal to the model's.
- Both residual additions rebuild each layer's output exactly.
- Replaying Gated DeltaNet token by token, and summing `read[t, s] * delta[s]`, each match the model's chunked
  kernel to within 0.5%. The kernel rounds to bf16, which accounts for up to 0.39%; the measured worst case is 0.33%.
- Every stored byte decodes to within half a quantisation step.

## Rebuild

Node is in `~/tools/node`. Install with `npm install --ignore-scripts`.

    # New traces. Edit PROMPTS in tracer/trace.py first. About 1 minute and 9 GB on the GPU.
    HF_HUB_OFFLINE=1 ~/JimmyProj/oss/sglang-omni-pr2803/.venv/bin/python tracer/trace.py

    # Site. The build replaces out/, so the container must restart to see the new directory.
    PATH=~/tools/node/bin:$PATH npx next build
    docker compose -f deploy/docker-compose.yml restart

    # Serving
    docker compose -f deploy/docker-compose.yml up -d
    tailscale serve --bg --https=8444 http://127.0.0.1:8096

The tracer needs a GPU-capable torch and a transformers with `qwen3_5`; the venv above is borrowed from another
project and pinned to nothing here. Before running it, check that nothing large is on the GPU: the Spark can
freeze when unified memory fills up.
