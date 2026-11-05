#!/usr/bin/env python3
"""Run Qwen3.5-4B on fixed prompts, record every intermediate value, and write
the binary files the 3D viewer loads.

Layout of the output (public/q35/):
  index.json                       prompts + model constants
  weights/manifest.json + LL.bin   sampled weights, with stats over the FULL matrices
  <prompt>/manifest.json           tokens, predictions, logit lens, tensor table
  <prompt>/overview.bin            residual stream for all layers (every dim)
  <prompt>/layer_LL.bin            every intermediate for layer LL (every dim)

All tensors are int8, scaled per-tensor by max|x|: value = byte / 127 * scale.
Weight matrices are sampled with a fixed stride (exact values, no averaging).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
WEIGHT_STRIDE = 40          # same stride on every long axis so blocks stay to scale
HEAD_DIM_STRIDE = 8         # 128-dim head axes -> 16 cells
TOPK = 10
PROMPTS = [
    ("capital", "The capital of France is"),
    ("story", "Alice gave the book to Bob, and then Bob handed it back to"),
    ("code", "def add(a, b):\n    return"),
    ("chinese", "中国的首都是"),
]


def to_np(t):
    return t.detach().float().cpu().numpy()


class Writer:
    """Append int8-quantised tensors to one file and remember how to read them."""

    def __init__(self, path):
        self.path, self.buf, self.entries, self.off = path, [], {}, 0

    def add(self, name, arr, **extra):
        arr = np.asarray(arr, dtype=np.float32)
        scale = float(np.abs(arr).max()) or 1.0
        q = np.clip(np.rint(arr / scale * 127), -127, 127).astype(np.int8)
        err = float(np.abs(q.astype(np.float32) / 127 * scale - arr).max())
        assert err <= scale / 127 * 0.5001 + 1e-6, (name, err, scale)
        self.entries[name] = dict(
            shape=list(arr.shape), offset=self.off, scale=scale,
            p99=float(np.percentile(np.abs(arr), 99)) or scale,
            mean=float(arr.mean()), std=float(arr.std()), **extra,
        )
        b = q.tobytes()
        self.buf.append(b)
        self.off += len(b)

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"".join(self.buf))
        return self.entries


class GdnTap:
    """Capture Gated DeltaNet internals without changing what it computes."""

    def __init__(self, mod):
        self.mod, self.rec, self.handles = mod, {}, []
        self.orig_chunk = mod.chunk_gated_delta_rule

    def __enter__(self):
        m, r = self.mod, self.rec
        hook = lambda name: (lambda _m, _i, o: r.__setitem__(name, o.detach()))
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "out_proj", "norm"):
            self.handles.append(getattr(m, n).register_forward_hook(hook(n)))

        def chunk(query, key, value, g, beta, **kw):
            out, st = self.orig_chunk(query, key, value, g=g, beta=beta, **kw)
            r.update(query=query.detach(), key=key.detach(), value=value.detach(),
                     g=g.detach(), beta=beta.detach(), core=out.detach())
            return out, st

        m.chunk_gated_delta_rule = chunk
        return self

    def __exit__(self, *a):
        self.mod.chunk_gated_delta_rule = self.orig_chunk
        for h in self.handles:
            h.remove()


def gated_delta_replay(rec, use_l2=True):
    """Re-run the delta rule one token at a time (float32, on CPU-friendly shapes).

    Returns:
      states: (T, heads, 128, 128)  memory S after each token (row = key dim)
      delta:  (T, heads, 128)       what each token wrote, after error correction
      read:   (T, T, heads)         read[t, s] = decay(s->t) * (q_t . k_s)
      out:    (T, heads, 128)       output; must match the model's own kernel
    The output is exactly linear in delta: out[t] = sum_{s<=t} read[t,s] * delta[s].
    """
    # Same order as the kernel: L2-normalise in the model's bf16, then move to float32.
    l2 = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q, k = l2(rec["query"][0]).float(), l2(rec["key"][0]).float()      # (T, H, D)
    v = rec["value"][0].float()
    g, b = rec["g"][0].float(), rec["beta"][0].float()                 # (T, H)
    q = q / (q.shape[-1] ** 0.5)
    T, H, D = q.shape
    S = torch.zeros(H, D, D, device=q.device)
    states, deltas, outs = [], [], []
    for t in range(T):
        S = S * g[t].exp()[:, None, None]
        err = v[t] - (S * k[t][:, :, None]).sum(1)
        delta = err * b[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        outs.append((S * q[t][:, :, None]).sum(1))
        states.append(S)
        deltas.append(delta)
    states, deltas, outs = map(torch.stack, (states, deltas, outs))
    # decay(s->t) = exp(sum_{r=s+1..t} g_r); read[t,s] is zero for s>t (causal)
    cg = g.cumsum(0)
    read = torch.zeros(T, T, H, device=q.device)
    for t in range(T):
        for s in range(t + 1):
            read[t, s] = (cg[t] - cg[s]).exp() * (q[t] * k[s]).sum(-1)
    recon = torch.einsum("tsh,shd->thd", read, deltas)
    return states, deltas, read, outs, recon


@torch.no_grad()
def trace_prompt(model, tok, key, text, out_dir, device):
    lm = model.model.language_model
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    ids = enc.input_ids
    T = ids.shape[1]

    # A clean run with no hooks, to compare against afterwards.
    clean = model(**enc, use_cache=False).logits.float()

    captured = {}
    handles = []

    def save(name):
        def f(_m, _i, o):
            captured[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return f

    handles.append(lm.embed_tokens.register_forward_hook(save("embed")))
    handles.append(lm.norm.register_forward_hook(save("final_norm")))
    taps = {}
    for i, layer in enumerate(lm.layers):
        p = f"L{i}."
        handles.append(layer.register_forward_pre_hook(
            (lambda n: lambda _m, a, kw: captured.__setitem__(n, (a[0] if a else kw["hidden_states"]).detach()))(p + "x_in"),
            with_kwargs=True))
        handles.append(layer.register_forward_hook(save(p + "x_out")))
        handles.append(layer.input_layernorm.register_forward_hook(save(p + "norm1")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save(p + "norm2")))
        for n in ("gate_proj", "up_proj", "down_proj"):
            handles.append(getattr(layer.mlp, n).register_forward_hook(save(p + "mlp_" + n)))
        if layer.layer_type == "linear_attention":
            taps[i] = GdnTap(layer.linear_attn).__enter__()
        else:
            sa = layer.self_attn
            def save_attn(_m, _i, o, n=p):
                # Must return None: a forward hook's return value replaces the output.
                captured[n + "attn_out"] = o[0].detach()
                captured[n + "attn_w"] = o[1].detach()
            handles.append(sa.register_forward_hook(save_attn))
            for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                handles.append(getattr(sa, n).register_forward_hook(save(p + n)))

    logits = model(**enc, use_cache=False, output_attentions=True).logits.float()
    for h in handles:
        h.remove()
    for t in taps.values():
        t.__exit__()

    checks = {}
    checks["hooks_change_nothing_max_abs"] = float((logits - clean).abs().max())
    assert checks["hooks_change_nothing_max_abs"] == 0.0, checks

    # Final logits rebuilt from what we stored: tied embedding x final norm.
    E = lm.embed_tokens.weight.float()
    rebuilt = model.lm_head(captured["final_norm"]).float()
    checks["logits_from_final_norm_max_abs"] = float((rebuilt - logits).abs().max())
    assert checks["logits_from_final_norm_max_abs"] == 0.0, checks

    probs = logits[0].softmax(-1)
    top_p, top_i = probs.topk(TOPK, dim=-1)
    dec = lambda i: tok.decode([int(i)])

    tokens = [dict(id=int(i), text=dec(i)) for i in ids[0]]
    predictions = [[dict(id=int(i), text=dec(i), p=float(p)) for i, p in zip(top_i[t], top_p[t])] for t in range(T)]

    ov = Writer(out_dir / "overview.bin")
    ov.add("embed", to_np(captured["embed"][0]))
    lens = []
    attn_last = []   # full-attention layers: the final token's weights, averaged over heads
    residual_gap = 0.0
    for i, layer in enumerate(lm.layers):
        p = f"L{i}."
        x_in, x_out = captured[p + "x_in"][0].float(), captured[p + "x_out"][0].float()
        ov.add(f"x_out_{i}", to_np(x_out))
        # Logit lens: read this layer's residual with the model's final norm and head.
        l = lm.norm(captured[p + "x_out"]).float() @ E.T
        pr = l[0, -1].softmax(-1)
        tp, ti = pr.topk(3)
        lens.append([dict(id=int(a), text=dec(a), p=float(b)) for a, b in zip(ti, tp)])

        w = Writer(out_dir / f"layer_{i:02d}.bin")
        w.add("x_in", to_np(x_in))
        w.add("norm1", to_np(captured[p + "norm1"][0]))
        if layer.layer_type == "linear_attention":
            r = taps[i].rec
            a = layer.linear_attn
            qkv_conv_in = r["in_proj_qkv"][0].float()
            conv = r["conv1d"][0].float().T[:T]
            qkv = F.silu(conv)
            w.add("qkv_raw", to_np(qkv_conv_in))
            w.add("qkv", to_np(qkv))
            w.add("z", to_np(r["in_proj_z"][0].float()))
            w.add("a", to_np(r["in_proj_a"][0].float()))
            w.add("b", to_np(r["in_proj_b"][0].float()))
            w.add("beta", to_np(r["beta"][0]))
            w.add("g", to_np(r["g"][0]))
            # query/key are repeated from 16 to 32 heads inside the layer; store the 16.
            w.add("q", to_np(r["query"][0][:, ::2]))
            w.add("k", to_np(r["key"][0][:, ::2]))
            w.add("v", to_np(r["value"][0]))
            states, deltas, read, outs, recon = gated_delta_replay(r)
            core = r["core"][0].float()
            # Relative to the largest output value; the kernel's result is rounded to bf16.
            denom = float(core.abs().max())
            gap = float((outs - core).abs().max()) / denom
            gap2 = float((recon - core).abs().max()) / denom
            checks[f"L{i}_replay_vs_kernel"] = gap
            checks[f"L{i}_read_decomposition"] = gap2
            assert max(gap, gap2) < 0.005, (i, gap, gap2)   # bf16 rounding is at most 0.39%
            w.add("core", to_np(core))
            w.add("delta", to_np(deltas))
            w.add("read", to_np(read.permute(2, 0, 1)))      # (H, T, T)
            s = HEAD_DIM_STRIDE
            w.add("state", to_np(states[:, :, ::s, ::s]))     # (T, H, 16, 16)
            w.add("state_norm", to_np(states.flatten(2).norm(dim=-1)))  # (T, H)
            w.add("gated", to_np(r["norm"].float().reshape(T, -1)))
            w.add("attn_out", to_np(r["out_proj"][0]))
            attn_out = r["out_proj"][0].float()
        else:
            q2 = captured[p + "q_proj"][0].float().reshape(T, 16, 512)
            w.add("q", to_np(captured[p + "q_norm"][0]))
            w.add("q_gate", to_np(q2[..., 256:]))
            w.add("k", to_np(captured[p + "k_norm"][0]))
            w.add("v", to_np(captured[p + "v_proj"][0].float().reshape(T, 4, 256)))
            w.add("attn_w", to_np(captured[p + "attn_w"][0]))  # (16, T, T)
            attn_last.append(dict(layer=i, weights=[float(x) for x in captured[p + "attn_w"][0].float().mean(0)[-1]]))
            attn_out = captured[p + "o_proj"][0].float()
            w.add("attn_out", to_np(attn_out))
        x_mid = x_in + attn_out
        w.add("x_mid", to_np(x_mid))
        w.add("norm2", to_np(captured[p + "norm2"][0]))
        gate, up = captured[p + "mlp_gate_proj"][0].float(), captured[p + "mlp_up_proj"][0].float()
        w.add("gate", to_np(gate))
        w.add("up", to_np(up))
        w.add("glu", to_np(F.silu(gate) * up))
        down = captured[p + "mlp_down_proj"][0].float()
        w.add("mlp_out", to_np(down))
        w.add("x_out", to_np(x_out))
        # Rebuild both residual additions in bf16, as the model does them.
        raw_mid = captured[p + "x_in"] + (taps[i].rec["out_proj"] if i in taps else captured[p + "o_proj"])
        raw_out = raw_mid + captured[p + "mlp_down_proj"]
        residual_gap = max(residual_gap, float((raw_out - captured[p + "x_out"]).abs().max()))
        entries = w.close()
        (out_dir / f"layer_{i:02d}.json").write_text(json.dumps(entries))

    checks["residual_identity_max_abs"] = residual_gap
    ov.add("final_norm", to_np(captured["final_norm"][0]))
    overview = ov.close()

    tv = [json.loads((out_dir / f"layer_{i:02d}.json").read_text()) for i in range(len(lm.layers))]
    tv_sizes = sum((out_dir / f"layer_{i:02d}.bin").stat().st_size for i in range(len(lm.layers)))
    manifest = dict(key=key, text=text, T=T, tokens=tokens, predictions=predictions, lens=lens,
                    overview=overview, checks=checks, layer_bytes=tv_sizes, attn_last=attn_last)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    worst = {k: v for k, v in checks.items() if "L" not in k}
    worst["replay_vs_kernel_max"] = max(v for k, v in checks.items() if k.endswith("replay_vs_kernel"))
    worst["read_decomposition_max"] = max(v for k, v in checks.items() if k.endswith("read_decomposition"))
    print(f"[{key}] T={T} top={predictions[-1][0]} checks={worst}", flush=True)
    return dict(key=key, text=text, T=T, top=predictions[-1][0]["text"], p=predictions[-1][0]["p"])


def stride2(t):
    """Sample every WEIGHT_STRIDE-th index on each axis longer than the stride."""
    idx = []
    for s in t.shape:
        idx.append(torch.arange(0, s, WEIGHT_STRIDE, device=t.device) if s > WEIGHT_STRIDE else torch.arange(s, device=t.device))
    return t[torch.meshgrid(*idx, indexing="ij")], [int(i[1] - i[0]) if len(i) > 1 else 1 for i in idx]


@torch.no_grad()
def write_weights(model, out):
    lm = model.model.language_model
    w_dir = out / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    ov = Writer(w_dir / "embed.bin")
    E = lm.embed_tokens.weight.float()
    ov.add("embed", to_np(E[::4000, ::WEIGHT_STRIDE]), full_shape=list(E.shape), full_std=float(E.std()),
           stride=[4000, WEIGHT_STRIDE])
    (w_dir / "embed.json").write_text(json.dumps(ov.close()))
    for i, layer in enumerate(lm.layers):
        w = Writer(w_dir / f"layer_{i:02d}.bin")

        def add(name, param):
            t = param.float()
            s, st = stride2(t)
            w.add(name, to_np(s), full_shape=list(t.shape), stride=st,
                  full_mean=float(t.mean()), full_std=float(t.std()),
                  full_min=float(t.min()), full_max=float(t.max()))

        add("norm1_w", layer.input_layernorm.weight)
        add("norm2_w", layer.post_attention_layernorm.weight)
        add("gate_w", layer.mlp.gate_proj.weight)
        add("up_w", layer.mlp.up_proj.weight)
        add("down_w", layer.mlp.down_proj.weight)
        if layer.layer_type == "linear_attention":
            a = layer.linear_attn
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                add(n, getattr(a, n).weight)
            add("conv1d", a.conv1d.weight[:, 0, :])
            add("A_log", a.A_log)
            add("dt_bias", a.dt_bias)
            add("norm_w", a.norm.weight)
        else:
            a = layer.self_attn
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(n, getattr(a, n).weight)
            add("q_norm_w", a.q_norm.weight)
            add("k_norm_w", a.k_norm.weight)
        (w_dir / f"layer_{i:02d}.json").write_text(json.dumps(w.close()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "public" / "q35"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager").to(a.device).eval()
    lm = model.model.language_model
    print("loaded", round(time.time() - t0, 1), "s; fla kernel:",
          lm.layers[0].linear_attn.chunk_gated_delta_rule.__module__, flush=True)
    assert lm.layers[0].linear_attn.causal_conv1d_fn is None, "conv hook expects the plain torch path"
    tied = model.lm_head.weight is lm.embed_tokens.weight
    print("lm_head tied:", tied, flush=True)
    assert tied
    write_weights(model, out)
    prompts = []
    for key, text in PROMPTS:
        prompts.append(trace_prompt(model, tok, key, text, out / key, a.device))
    n = model.config.text_config
    index = dict(
        model=MODEL, revision=REVISION, prompts=prompts,
        weight_stride=WEIGHT_STRIDE, head_dim_stride=HEAD_DIM_STRIDE,
        layer_types=list(n.layer_types), hidden=n.hidden_size, mlp=n.intermediate_size,
        vocab=n.vocab_size,
        params=dict(text=sum(p.numel() for x, p in model.named_parameters() if "visual" not in x),
                    vision=sum(p.numel() for x, p in model.named_parameters() if "visual" in x)),
    )
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print("done in", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
