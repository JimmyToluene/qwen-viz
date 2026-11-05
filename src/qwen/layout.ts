// Places every tensor of one forward pass in 3D.
//
// Scale: one cell is one real value, and every long axis shows one value in 40, so
// a 9216-wide block really is 3.6x a 2560-tall one. Tokens are always shown in full.
// x = tokens / input dims, y = output dims (downwards), z = heads.
import { IBlkDef } from "@/src/llm/types";
import { Mat4f } from "@/src/utils/matrix";
import { IBufferTex } from "@/src/utils/renderPhases";
import { IIndex, ITensor, ITensorSet, ITrace } from "./data";

export type Axis = { name: string; stride: number; of: number; kind?: 'token' | 'head' | 'plain'; base?: number };

export interface IQBlock extends IBlkDef {
    key: string;          // stable id, e.g. "L12.qkv"
    title: string;
    group: string;
    layer: number | null;
    tensor?: ITensor;     // source for the inspector
    axes?: { x?: Axis; y?: Axis; z?: Axis };
    tokenAnimated?: { perToken: number };
    label: boolean;
    slice?: { offset: number; rows: number; cols: number; tokenAxes?: boolean; note: string; perToken?: number };
}

export interface ITag { layer: number; x: number; y: number; z: number; }
export type IArrow = { from: IQBlock; to: IQBlock; mode: 'h' | 'v' | 'bl' | 'rt' };

export interface IQLayout {
    cell: number;
    margin: number;
    height: number;
    cubes: IQBlock[];
    byKey: Record<string, IQBlock>;
    tags: ITag[];
    arrows: IArrow[];
    tokenAnchor: { x: number; y: number; z: number }[];
    expanded: number | null;
}

export type TexMaker = (key: string, w: number, h: number, build: () => Float32Array) => IBufferTex;

function values(t: ITensor) {
    let out = new Float32Array(t.data.length);
    let f = t.scale / 127;
    for (let i = 0; i < out.length; i++) out[i] = t.data[i] * f;
    return out;
}

// rows x (flattened rest of the axes), keeping every `stride`-th column.
function sampleCols(t: ITensor, rowAxes: number, stride: number) {
    let rows = t.shape.slice(0, rowAxes).reduce((a, b) => a * b, 1);
    let cols = t.shape.slice(rowAxes).reduce((a, b) => a * b, 1);
    let outCols = Math.ceil(cols / stride);
    return {
        w: outCols, h: rows,
        build: () => {
            let v = values(t);
            let out = new Float32Array(rows * outCols);
            for (let r = 0; r < rows; r++)
                for (let c = 0; c < outCols; c++)
                    out[r * outCols + c] = v[r * cols + c * stride];
            return out;
        },
    };
}

export function genLayout(index: IIndex, trace: ITrace, embedW: ITensorSet | null, expanded: number | null, tex: TexMaker, tokenCursor: number): IQLayout {
    const cell = 1.5, margin = 12;
    const T = trace.manifest.T;
    const S = index.weight_stride;
    const Tw = T * cell;
    const pkey = trace.manifest.key;
    const data = expanded !== null ? trace.layers.get(expanded) : undefined;
    let cubes: IQBlock[] = [];
    let byKey: Record<string, IQBlock> = {};
    let arrows: IArrow[] = [];
    let tags: ITag[] = [];

    function block(a: {
        key: string; title: string; group: string; layer?: number | null; t: 'w' | 'i' | 'a';
        x: number; y: number; z?: number; cx: number; cy: number; cz?: number; dz?: number;
        tensor?: ITensor; tw?: number; th?: number; build?: () => Float32Array; texKey?: string;
        ax?: number[]; ay?: number[]; axes?: IQBlock['axes']; label?: boolean; perToken?: number; slice?: IQBlock['slice'];
    }): IQBlock {
        let cz = a.cz ?? 1;
        let b: IQBlock = {
            idx: cubes.length, t: a.t, x: a.x, y: a.y, z: (a.z ?? 0) - cz * cell / 2,
            dx: a.cx * cell, dy: a.cy * cell, dz: cz * cell,
            cx: a.cx, cy: a.cy, cz, name: a.title, highlight: 0, opacity: 1, small: false,
            key: a.key, title: a.title, group: a.group, layer: a.layer ?? null, tensor: a.tensor,
            axes: a.axes, label: a.label ?? true, slice: a.slice,
            tokenAnimated: a.perToken ? { perToken: a.perToken } : undefined,
        };
        if (a.build && a.tw && a.th) {
            let src = tex(a.texKey ?? `${pkey}/${a.key}`, a.tw, a.th, a.build);
            let scale = a.tensor ? 1 / (a.tensor.p99 || a.tensor.scale || 1) : 1;
            let ax = [...(a.ax ?? [0, 1, 0]), 0].slice(0, 4), ay = [...(a.ay ?? [1, 0, 0]), 0].slice(0, 4);
            b.access = { src, channel: 'r', scale, mat: Mat4f.fromColMajor([...ax, ...ay, 0, 0, 0, 0, 0, 0, 0, 0]) };
        }
        cubes.push(b);
        byKey[a.key] = b;
        return b;
    }

    // (T, D...) tensor: x = token, y = sampled dim
    function tokenBlock(k: string, title: string, group: string, layer: number, t: ITensor, x: number, y: number, tt: 'i' | 'a' = 'i', stride = S, rowAxes = 1) {
        let s = sampleCols(t, rowAxes, stride);
        return block({
            key: k, title, group, layer, t: tt, x, y, cx: T, cy: s.w, tensor: t, tw: s.w, th: s.h, build: s.build,
            axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'dim', stride, of: s.w * stride } },
        });
    }
    // (T, D): x = sampled dim, y = token. Used for the wide MLP blocks.
    function wideBlock(k: string, title: string, group: string, layer: number, t: ITensor, x: number, y: number) {
        let s = sampleCols(t, 1, S);
        return block({
            key: k, title, group, layer, t: 'i', x, y, cx: s.w, cy: T, tensor: t, tw: s.w, th: s.h, build: s.build,
            ax: [1, 0, 0], ay: [0, 1, 0],
            axes: { x: { name: 'dim', stride: S, of: t.shape[1] }, y: { name: 'token', stride: 1, of: T, kind: 'token' } },
        });
    }
    // Weight (out, in), already sampled by the tracer: x = in, y = out.
    function weightBlock(k: string, title: string, group: string, layer: number, t: ITensor, x: number, y: number, flip = false) {
        let r = t.shape[0], c = t.shape[1] ?? 1;
        let st = t.stride ?? [1, 1], fs = t.full_shape ?? t.shape;
        let v = () => values(t);
        return block({
            key: k, title, group, layer, t: 'w', x, y, cx: flip ? r : c, cy: flip ? c : r, tensor: t, tw: c, th: r, build: v,
            texKey: `w/${k}`,
            ax: flip ? [0, 1, 0] : [1, 0, 0], ay: flip ? [1, 0, 0] : [0, 1, 0],
            axes: flip ? {
                x: { name: 'out', stride: st[0], of: fs[0] }, y: { name: 'in', stride: st[1] ?? 1, of: fs[1] ?? 1 },
            } : {
                x: { name: 'in', stride: st[1] ?? 1, of: fs[1] ?? 1 }, y: { name: 'out', stride: st[0], of: fs[0] },
            },
        });
    }
    function arrow(from: IQBlock, to: IQBlock, mode: IArrow['mode'] = 'h') { arrows.push({ from, to, mode }); }

    let y = 0;
    let cx0 = -Tw / 2;

    // ---- Input ----
    let tokenIds = new Float32Array(trace.manifest.tokens.map(t => t.id / index.vocab));
    let tokens = block({
        key: 'tokens', title: 'Tokens', group: 'input', t: 'i', x: cx0, y, cx: T, cy: 1, label: false,
        tw: T, th: 1, build: () => tokenIds, ax: [1, 0, 0], ay: [0, 1, 0],
        axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' } },
    });
    let tokenAnchor = trace.manifest.tokens.map((_, i) => ({ x: cx0 + (i + 0.5) * cell, y: y - 2, z: 0 }));
    y += cell + margin;

    let embed = tokenBlock('embed', 'Token embeddings', 'input', -1, trace.overview.embed, cx0, y);
    embed.layer = null;
    arrow(tokens, embed, 'v');
    if (embedW) {
        let t = embedW.embed;
        let table = block({
            key: 'embed_table', title: 'Embedding table (248,320 x 2560)', group: 'input', t: 'w',
            x: cx0 - 3 * margin - t.shape[0] * cell, y, cx: t.shape[0], cy: t.shape[1], tensor: t,
            tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0],
            axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
        });
        arrow(table, embed);
    }
    y += embed.dy + 2 * margin;

    // ---- Layer tower ----
    let prev: IQBlock = embed;
    const collapsedCells = 16;
    const collapsedStride = S * 4;
    for (let i = 0; i < index.layer_types.length; i++) {
        let full = index.layer_types[i] === 'full_attention';
        let xOut = trace.overview[`x_out_${i}`];
        let tag: ITag = { layer: i, x: Tw / 2 + 4, y, z: 0 };
        if (expanded === i && data) {
            let r = layoutLayer(i, full, y, data.act, data.w);
            y = r.y;
            prev = r.out;
            tag.y = r.tagY;
        } else {
            let s = sampleCols(xOut, 1, S);
            let b = block({
                key: `L${i}.x_out_small`, title: `Layer ${i} output`, group: 'tower', layer: i, t: full ? 'a' : 'i',
                x: cx0, y, cx: T, cy: collapsedCells, tensor: xOut, tw: s.w, th: s.h, build: s.build,
                texKey: `${pkey}/x_out_${i}`, ax: [0, 4, 0], ay: [1, 0, 0], label: false,
                axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'dim', stride: collapsedStride, of: 2560 } },
            });
            arrow(prev, b, 'v');
            prev = b;
            y += b.dy + margin * 0.75;
        }
        tags.push(tag);
    }
    y += margin;

    // ---- Output ----
    let finalNorm = tokenBlock('final_norm', 'Final RMSNorm', 'output', -1, trace.overview.final_norm, cx0, y);
    finalNorm.layer = null;
    arrow(prev, finalNorm, 'v');
    if (embedW) {
        let t = embedW.embed;
        let head = block({
            key: 'lm_head', title: 'Output head (same weights as the embedding table)', group: 'output', t: 'w',
            x: cx0 - 3 * margin - t.shape[0] * cell, y, cx: t.shape[0], cy: t.shape[1], tensor: t,
            tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0],
            axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
        });
        arrow(head, finalNorm);
    }
    y += finalNorm.dy + margin;
    let topK = trace.manifest.predictions[0].length;
    let probs = new Float32Array(T * topK);
    trace.manifest.predictions.forEach((row, t) => row.forEach((p, k) => probs[t * topK + k] = p.p));
    let probT: ITensor = {
        name: 'top_probs', shape: [T, topK], offset: 0, scale: 1, p99: 1, mean: 0, std: 0,
        data: new Int8Array(probs.map(p => Math.round(p * 127))),
    };
    let top = block({
        key: 'top_probs', title: 'Top 10 probs', group: 'output', t: 'i', x: cx0, y,
        cx: T, cy: topK, tensor: probT, tw: topK, th: T, build: () => probs,
        axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'rank', stride: 1, of: topK } },
    });
    arrow(finalNorm, top, 'v');
    y += top.dy;

    cubes.forEach((c, i) => c.idx = i);

    return { cell, margin, height: y, cubes, byKey, tags, arrows, tokenAnchor, expanded };

    function layoutLayer(i: number, full: boolean, startY: number, a: ITensorSet, w: ITensorSet) {
        let L = `L${i}.`;
        let y = startY;
        let right = cx0 - 4 * margin;          // right edge of the side pathway
        let at = (xr: number, width: number) => xr - width * cell;

        // Pre-attention norm
        let norm1 = tokenBlock(L + 'norm1', 'RMSNorm', 'norm1', i, a.norm1, at(right, T), y);
        let gamma1 = weightBlock(L + 'norm1_w', 'Norm scale', 'norm1', i, w.norm1_w, at(norm1.x - margin, 1), y);
        arrow(prev, norm1, 'rt');
        arrow(gamma1, norm1);
        y += norm1.dy + 2 * margin;

        let attnOut: IQBlock;
        let rowEnd: number;
        let attnBottom: number;
        if (!full) {
            // ---- QKV and the causal convolution ----
            let qkv = tokenBlock(L + 'qkv', 'Q,K,V after conv + SiLU', 'qkv', i, a.qkv, at(right, T), y);
            let conv = weightBlock(L + 'conv1d', 'Conv kernel (4 taps)', 'qkv', i, w.conv1d, at(qkv.x - margin, 4), y);
            let raw = tokenBlock(L + 'qkv_raw', 'Projected Q,K,V', 'qkv', i, a.qkv_raw, at(conv.x - margin, T), y);
            let wq = weightBlock(L + 'in_proj_qkv', 'in_proj_qkv', 'qkv', i, w.in_proj_qkv, at(raw.x - margin, 64), y);
            arrow(norm1, wq, 'bl'); arrow(wq, raw); arrow(raw, conv); arrow(conv, qkv);
            y += qkv.dy + 2 * margin;

            // ---- Split ----
            let qB = tokenBlock(L + 'q', 'Q (16 heads)', 'split', i, a.q, at(right, T), y, 'i', S, 1);
            let kB = tokenBlock(L + 'k', 'K (16 heads)', 'split', i, a.k, at(qB.x - margin, T), y, 'i', S, 1);
            let vB = tokenBlock(L + 'v', 'V (32 heads)', 'split', i, a.v, at(kB.x - margin, T), y, 'i', S, 1);
            arrow(qkv, qB, 'v');
            y += Math.max(qB.dy, vB.dy) + 2 * margin;

            // ---- Gates ----
            let g = tokenBlock(L + 'g', 'g: log decay', 'gates', i, a.g, at(right, T), y, 'i', 1);
            let dt = weightBlock(L + 'dt_bias', 'dt_bias', 'gates', i, w.dt_bias, at(g.x - margin, 1), y);
            let aLog = weightBlock(L + 'A_log', 'A_log', 'gates', i, w.A_log, at(dt.x - margin, 1), y);
            let wa = weightBlock(L + 'in_proj_a', 'in_proj_a', 'gates', i, w.in_proj_a, at(aLog.x - margin, 64), y);
            let beta = tokenBlock(L + 'beta', 'beta: write gate', 'gates', i, a.beta, at(wa.x - 2 * margin, T), y, 'i', 1);
            let wb = weightBlock(L + 'in_proj_b', 'in_proj_b', 'gates', i, w.in_proj_b, at(beta.x - margin, 64), y);
            arrow(wb, beta); arrow(wa, aLog); arrow(aLog, dt); arrow(dt, g);
            y += Math.max(beta.dy, wb.dy) + 3 * margin;

            // ---- Memory: one slab per value head ----
            const H = 32, cellsPerHead = 16;
            const slabStep = 9;
            let rd = sampleCols(a.read, 2, 1);          // rows: H*T, cols: T
            let st = sampleCols(a.state, 3, 1);         // rows: T*H*16, cols: 16
            let readRight = right;
            let memRows = Math.max(T, cellsPerHead);
            let midY = y + (memRows * cell) / 2;
            let readY = midY - T * cell / 2, stY = midY - cellsPerHead * cell / 2;
            let readX = at(readRight, T);
            let stateX = at(readX - 3 * margin, cellsPerHead);
            for (let h = 0; h < H; h++) {
                let z = (h - (H - 1) / 2) * slabStep;
                block({
                    key: `${L}read${h}`, title: `Read weights, head ${h}`, group: 'memory', layer: i, t: 'i', x: readX, y: readY, z,
                    cx: T, cy: T, tensor: a.read, tw: rd.w, th: rd.h, build: rd.build, texKey: `${pkey}/${L}read`,
                    ax: [1, 0, 0, 0], ay: [0, 1, 0, h * T], label: h === H - 1,
                    slice: { offset: h * T * T, rows: T, cols: T, tokenAxes: true, note: `Head ${h} of ${H}.` },
                    axes: { x: { name: 'written at', stride: 1, of: T, kind: 'token' }, y: { name: 'read at', stride: 1, of: T, kind: 'token' } },
                });
                block({
                    key: `${L}state${h}`, title: `Memory S, head ${h}`, group: 'memory', layer: i, t: 'a', x: stateX, y: stY, z,
                    cx: cellsPerHead, cy: cellsPerHead, tensor: a.state, tw: st.w, th: st.h, build: st.build, texKey: `${pkey}/${L}state`,
                    ax: [1, 0, 0, 0], ay: [0, 1, 0, cellsPerHead * (h + H * tokenCursor)], label: h === H - 1, perToken: H * cellsPerHead,
                    slice: { offset: (tokenCursor * H + h) * cellsPerHead * cellsPerHead, rows: cellsPerHead, cols: cellsPerHead, note: `Head ${h} of ${H}, after token ${tokenCursor}. One value in 8 per axis.` },
                    axes: { x: { name: 'value dim', stride: 8, of: 128 }, y: { name: 'key dim', stride: 8, of: 128 } },
                });
            }
            arrow(beta, byKey[`${L}state${H - 1}`], 'v');
            arrow(g, byKey[`${L}read${H - 1}`], 'v');
            y += memRows * cell + 3 * margin;

            // ---- Output of the mixer ----
            let core = tokenBlock(L + 'core', 'Memory output', 'mixout', i, a.core, at(right, T), y, 'i', S, 1);
            let gated = tokenBlock(L + 'gated', 'Normed and gated', 'mixout', i, a.gated, at(core.x - margin, T), y);
            let z = tokenBlock(L + 'z', 'z: output gate', 'mixout', i, a.z, at(gated.x - margin, T), y);
            let wz = weightBlock(L + 'in_proj_z', 'in_proj_z', 'mixout', i, w.in_proj_z, at(z.x - margin, 64), y);
            arrow(wz, z); arrow(z, gated); arrow(core, gated);
            y += Math.max(core.dy, wz.dy) + 2 * margin;
            attnOut = tokenBlock(L + 'attn_out', 'Attention output', 'mixout', i, a.attn_out, at(right, T), y);
            let wo = weightBlock(L + 'out_proj', 'out_proj', 'mixout', i, w.out_proj, at(attnOut.x - margin, w.out_proj.shape[1]), y);
            arrow(gated, wo, 'v'); arrow(wo, attnOut);
            arrow(byKey[`${L}read${H - 1}`], core, 'v');
            rowEnd = y;
            attnBottom = y + Math.max(attnOut.dy, wo.dy);
        } else {
            // ---- Query (+ gate), key, value ----
            let qB = tokenBlock(L + 'q', 'Q (16 heads x 256)', 'fullqkv', i, a.q, at(right, T), y);
            let gB = tokenBlock(L + 'q_gate', 'Output gate (16 x 256)', 'fullqkv', i, a.q_gate, at(qB.x - margin, T), y);
            let wq = weightBlock(L + 'q_proj', 'q_proj (query + gate)', 'fullqkv', i, w.q_proj, at(gB.x - margin, 64), y);
            arrow(norm1, wq, 'bl'); arrow(wq, gB); arrow(gB, qB);
            y += wq.dy + 2 * margin;
            let kB = tokenBlock(L + 'k', 'K (4 heads x 256)', 'fullqkv', i, a.k, at(right, T), y);
            let wk = weightBlock(L + 'k_proj', 'k_proj', 'fullqkv', i, w.k_proj, at(kB.x - margin, 64), y);
            let vB = tokenBlock(L + 'v', 'V (4 heads x 256)', 'fullqkv', i, a.v, at(wk.x - 2 * margin, T), y);
            let wv = weightBlock(L + 'v_proj', 'v_proj', 'fullqkv', i, w.v_proj, at(vB.x - margin, 64), y);
            arrow(wk, kB); arrow(wv, vB);
            y += Math.max(kB.dy, wk.dy) + 3 * margin;

            // ---- Attention matrices, one per query head ----
            const H = 16, slabStep = 12;
            let rd = sampleCols(a.attn_w, 2, 1);
            let ax = at(right, T);
            for (let h = 0; h < H; h++) {
                let z = (h - (H - 1) / 2) * slabStep;
                block({
                    key: `${L}attn${h}`, title: `Attention, head ${h}`, group: 'attn', layer: i, t: 'a', x: ax, y, z,
                    cx: T, cy: T, tensor: a.attn_w, tw: rd.w, th: rd.h, build: rd.build, texKey: `${pkey}/${L}attn_w`,
                    ax: [1, 0, 0, 0], ay: [0, 1, 0, h * T], label: h === H - 1,
                    slice: { offset: h * T * T, rows: T, cols: T, tokenAxes: true, note: `Head ${h} of ${H}. Each row sums to 1.` },
                    axes: { x: { name: 'attends to', stride: 1, of: T, kind: 'token' }, y: { name: 'at', stride: 1, of: T, kind: 'token' } },
                });
            }
            arrow(qB, byKey[`${L}attn${H - 1}`], 'v');
            y += T * cell + 3 * margin;
            attnOut = tokenBlock(L + 'attn_out', 'Attention output', 'attnout', i, a.attn_out, at(right, T), y);
            let wo = weightBlock(L + 'o_proj', 'o_proj', 'attnout', i, w.o_proj, at(attnOut.x - margin, w.o_proj.shape[1]), y);
            arrow(wo, attnOut);
            arrow(byKey[`${L}attn${H - 1}`], attnOut, 'v');
            rowEnd = y;
            attnBottom = y + Math.max(attnOut.dy, wo.dy);
        }

        // First residual: x + attention
        let xMid = tokenBlock(L + 'x_mid', 'Residual: x + attention', full ? 'attnout' : 'mixout', i, a.x_mid, cx0, rowEnd);
        arrow(prev, xMid, 'v'); arrow(attnOut, xMid);
        y = Math.max(rowEnd + xMid.dy, attnBottom) + 2 * margin;

        // ---- MLP (SwiGLU) ----
        let norm2 = tokenBlock(L + 'norm2', 'RMSNorm', 'mlp', i, a.norm2, at(right, T), y);
        let gamma2 = weightBlock(L + 'norm2_w', 'Norm scale', 'mlp', i, w.norm2_w, at(norm2.x - margin, 1), y);
        arrow(xMid, norm2, 'rt'); arrow(gamma2, norm2);
        y += norm2.dy + 2 * margin;
        let mw = w.gate_w.shape[0];
        let gateW = weightBlock(L + 'gate_w', 'gate_proj', 'mlp', i, w.gate_w, at(right, mw), y, true);
        let upW = weightBlock(L + 'up_w', 'up_proj', 'mlp', i, w.up_w, at(gateW.x - 2 * margin, mw), y, true);
        arrow(norm2, gateW, 'v'); arrow(norm2, upW, 'bl');
        y += gateW.dy + margin;
        let gate = wideBlock(L + 'gate', 'gate', 'mlp', i, a.gate, gateW.x, y);
        let up = wideBlock(L + 'up', 'up', 'mlp', i, a.up, upW.x, y);
        arrow(gateW, gate, 'v'); arrow(upW, up, 'v');
        y += gate.dy + margin;
        let glu = wideBlock(L + 'glu', 'SiLU(gate) x up', 'mlp', i, a.glu, gateW.x, y);
        arrow(gate, glu, 'v'); arrow(up, glu, 'h');
        y += glu.dy + margin;
        let downW = weightBlock(L + 'down_w', 'down_proj', 'mlp', i, w.down_w, gateW.x, y);
        arrow(glu, downW, 'v');
        let mlpOut = tokenBlock(L + 'mlp_out', 'MLP output', 'mlp', i, a.mlp_out, at(right, T) + 0, y);
        // keep the output right of the down projection
        mlpOut.x = right - T * cell;
        downW.x = mlpOut.x - margin - downW.dx;
        let xOutB = tokenBlock(L + 'x_out', `Layer ${i} output`, 'mlp', i, a.x_out, cx0, y);
        arrow(downW, mlpOut); arrow(mlpOut, xOutB); arrow(xMid, xOutB, 'v');
        y += xOutB.dy + 2 * margin;
        return { y, out: xOutB, tagY: xOutB.y };
    }
}
