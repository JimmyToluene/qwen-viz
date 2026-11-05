// One small scene per section of the walkthrough. Each scene holds only the tensors that
// section is about, laid out left to right in the order they are computed.
//
// Conventions: one cell is one real value. Long axes show one value in 40, so a 9216-wide
// block really is 3.6 times a 2560-tall one. The token axis is always shown in full, and is
// stretched when a prompt is short so that short and long prompts remain legible.
import { IBlkDef } from "@/src/llm/types";
import { Mat4f } from "@/src/utils/matrix";
import { IBufferTex } from "@/src/utils/renderPhases";
import { IIndex, ITensor, ITensorSet, ITrace } from "./data";

export type Axis = { name: string; stride: number; of: number; kind?: 'token' };
export type SceneId = 'tower' | 'input' | 'layer' | 'norm' | 'qkv' | 'split' | 'gates' | 'memory' | 'read' | 'mixout' | 'mlpup' | 'mlp'
    | 'fullq' | 'fullkv' | 'attn' | 'attnout' | 'output';

export interface IQBlock extends IBlkDef {
    key: string;
    title: string;
    shapeText: string;
    layer: number | null;     // set on tower slabs: clicking one opens that layer
    tensor?: ITensor;
    axes?: { x?: Axis; y?: Axis; z?: Axis };
    label: boolean;
    slice?: { offset: number; rows: number; cols: number; tokenAxes?: boolean; note: string };
}

export interface IArrow { from: IQBlock; to: IQBlock; mode: 'h' | 'v' | 'side'; }
// A glyph between blocks, or a short note: 'right' starts at the point, 'above' ends at it.
export interface IOp { text: string; x: number; y: number; z: number; kind?: 'right' | 'above'; }
export interface ITokenAxis { key: string; axis: 'top' | 'left'; }
export interface ITag { layer: number; x: number; y: number; z: number; }

export interface IQLayout {
    cell: number;
    margin: number;
    height: number;
    cubes: IQBlock[];
    byKey: Record<string, IQBlock>;
    arrows: IArrow[];
    ops: IOp[];
    tokenAxes: ITokenAxis[];
    tags: ITag[];
    scene: SceneId;
    version: number;
}

export type TexMaker = (key: string, w: number, h: number, build: () => Float32Array) => IBufferTex;

const cell = 1.5, gap = 22;
// Blocks have real depth, as in llm-viz: thin weight sheets, thicker computed values.
const depthW = 3, depthI = 10;
let versionCounter = 0;

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

// Average a (heads, T, T) tensor over heads, as a new tensor the inspector can show.
export function meanOverHeads(t: ITensor, name: string): ITensor {
    let [H, T] = t.shape, v = values(t), m = new Float32Array(T * T);
    for (let h = 0; h < H; h++) for (let i = 0; i < T * T; i++) m[i] += v[h * T * T + i] / H;
    let scale = m.reduce((a, b) => Math.max(a, Math.abs(b)), 0) || 1;
    let sorted = Array.from(m, Math.abs).sort((a, b) => a - b);
    let mean = m.reduce((a, b) => a + b, 0) / m.length;
    let std = Math.sqrt(m.reduce((a, b) => a + (b - mean) ** 2, 0) / m.length);
    return {
        name, shape: [T, T], offset: 0, scale, mean, std, p99: sorted[Math.floor(sorted.length * 0.99)] || scale,
        data: new Int8Array(m.map(x => Math.round(x / scale * 127))),
    };
}

export interface ISceneArgs {
    index: IIndex;
    trace: ITrace;
    embedW: ITensorSet | null;
    scene: SceneId;
    layer: number;
    tokenCursor: number;
    tex: TexMaker;
}

export function genScene(args: ISceneArgs): IQLayout {
    let { index, trace, embedW, layer, tokenCursor, tex } = args;
    const T = trace.manifest.T;
    const S = index.weight_stride;
    const wT = Math.max(T * 3, 36);                // width of a token axis, wide enough to label
    const pkey = trace.manifest.key;
    const data = trace.layers.get(layer);
    const a = data?.act ?? {}, w = data?.w ?? {};
    const L = `L${layer}.`;
    // A layer's files may still be loading; show the tower until they arrive.
    let scene: SceneId = (args.scene === 'tower' || args.scene === 'input' || args.scene === 'output' || data) ? args.scene : 'tower';

    let cubes: IQBlock[] = [], byKey: Record<string, IQBlock> = {}, arrows: IArrow[] = [];
    let ops: IOp[] = [], tokenAxes: ITokenAxis[] = [], tags: ITag[] = [];

    function block(o: {
        key: string; title: string; shape?: string; t: 'w' | 'i' | 'a'; x: number; y: number; z?: number;
        cx: number; cy: number; dx?: number; dy?: number; dz?: number;
        tensor?: ITensor; tw?: number; th?: number; build?: () => Float32Array; texKey?: string;
        ax?: number[]; ay?: number[]; axes?: IQBlock['axes']; label?: boolean; slice?: IQBlock['slice'];
        layer?: number | null; scaleRef?: ITensor; plainScale?: number;
    }): IQBlock {
        let dz = o.dz ?? (o.t === 'w' ? depthW : depthI);
        let b: IQBlock = {
            idx: cubes.length, t: o.t, x: o.x, y: o.y, z: (o.z ?? 0) - dz / 2,
            dx: o.dx ?? o.cx * cell, dy: o.dy ?? o.cy * cell, dz,
            cx: o.cx, cy: o.cy, cz: 1, name: o.title, highlight: 0, opacity: 1, small: false,
            key: o.key, title: o.title, shapeText: o.shape ?? '', layer: o.layer ?? null, tensor: o.tensor,
            axes: o.axes, label: o.label ?? true, slice: o.slice,
        };
        if (o.build && o.tw && o.th) {
            let src = tex(o.texKey ?? `${pkey}/${o.key}`, o.tw, o.th, o.build);
            let ref = o.scaleRef ?? o.tensor;
            let scale = o.plainScale ?? (ref ? 1 / (ref.p99 || ref.scale || 1) : 1);
            let ax = [...(o.ax ?? [0, 1, 0]), 0, 0, 0, 0].slice(0, 4), ay = [...(o.ay ?? [1, 0, 0]), 0, 0, 0, 0].slice(0, 4);
            b.access = { src, channel: 'r', scale, mat: Mat4f.fromColMajor([...ax, ...ay, 0, 0, 0, 0, 0, 0, 0, 0]) };
        }
        cubes.push(b);
        byKey[o.key] = b;
        return b;
    }

    const shapeOf = (t: ITensor) => (t.full_shape ?? t.shape).map(n => n === T ? 'T' : String(n)).join(' × ');

    // (T, D...): x = token, y = sampled dim.
    function tok(key: string, title: string, t: ITensor, x: number, y: number, o: { stride?: number; scaleRef?: ITensor; tt?: 'i' | 'a'; plainScale?: number; label?: boolean; texKey?: string; dx?: number } = {}) {
        let stride = o.stride ?? S;
        let s = sampleCols(t, 1, stride);
        return block({
            key, title, shape: shapeOf(t), t: o.tt ?? 'i', x, y, cx: T, cy: s.w, dx: o.dx ?? wT, tensor: t, tw: s.w, th: s.h, build: s.build,
            scaleRef: o.scaleRef, plainScale: o.plainScale, label: o.label, texKey: o.texKey,
            axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'dim', stride, of: s.w * stride } },
        });
    }
    // (T, D): x = sampled dim, y = token. For the wide MLP blocks.
    function wide(key: string, title: string, t: ITensor, x: number, y: number) {
        let s = sampleCols(t, 1, S);
        return block({
            key, title, shape: shapeOf(t), t: 'i', x, y, cx: s.w, cy: T, dy: Math.max(T * cell, 24), tensor: t, tw: s.w, th: s.h, build: s.build,
            ax: [1, 0, 0], ay: [0, 1, 0],
            axes: { x: { name: 'dim', stride: S, of: t.shape[1] }, y: { name: 'token', stride: 1, of: T, kind: 'token' } },
        });
    }
    // Weight matrix (out, in), sampled by the tracer. x = in, y = out; `flip` swaps the two.
    function wt(key: string, title: string, t: ITensor, x: number, y: number, o: { flip?: boolean; dy?: number } = {}) {
        let r = t.shape[0], c = t.shape[1] ?? 1;
        let st = t.stride ?? [1, 1], fs = t.full_shape ?? t.shape;
        let inAxis = { name: 'in', stride: st[1] ?? 1, of: fs[1] ?? 1 }, outAxis = { name: 'out', stride: st[0], of: fs[0] };
        return block({
            key, title, shape: shapeOf(t), t: 'w', x, y, cx: o.flip ? r : c, cy: o.flip ? c : r,
            dx: !o.flip && c === 1 ? 8 : undefined, dy: o.dy, tensor: t, tw: c, th: r, build: () => values(t),
            texKey: `w/${key}`, ax: o.flip ? [0, 1, 0] : [1, 0, 0], ay: o.flip ? [1, 0, 0] : [0, 1, 0],
            axes: o.flip ? { x: outAxis, y: inAxis } : { x: inAxis, y: outAxis },
        });
    }
    // One (T, T) matrix drawn from a (heads, T, T) tensor, or a mean.
    function matrix(key: string, title: string, t: ITensor, head: number, x: number, y: number, side: number, tt: 'i' | 'a', o: { label?: boolean; scale?: number; note: string; xName: string; yName: string; texKey: string }) {
        let rows = t.shape.length === 3 ? t.shape[0] * T : T;
        let build = () => values(t);
        return block({
            key, title, shape: o.label === false ? '' : (head < 0 ? 'T × T' : 'T × T'), t: tt, x, y, z: 0, cx: T, cy: T, dx: side, dy: side, tensor: t, tw: T, th: rows, build,
            texKey: o.texKey, ax: [1, 0, 0, 0], ay: [0, 1, 0, Math.max(head, 0) * T], label: o.label, plainScale: o.scale,
            slice: { offset: Math.max(head, 0) * T * T, rows: T, cols: T, tokenAxes: true, note: o.note },
            axes: { x: { name: o.xName, stride: 1, of: T, kind: 'token' }, y: { name: o.yName, stride: 1, of: T, kind: 'token' } },
        });
    }
    let arrow = (from: IQBlock, to: IQBlock, mode: IArrow['mode'] = 'h') => { arrows.push({ from, to, mode }); };
    let op = (text: string, l: IQBlock, r: IQBlock) => {
        ops.push({ text, x: (l.x + l.dx + r.x) / 2, y: l.y + Math.min(l.dy, r.dy) / 2, z: 0 });
    };
    let after = (b: IQBlock, g = gap) => b.x + b.dx + g;

    if (scene === 'tower') sceneTower();
    if (scene === 'input') sceneInput();
    if (scene === 'output') sceneOutput();
    if (scene === 'layer') sceneLayer();
    if (scene === 'norm') sceneNorm();
    if (scene === 'qkv') sceneQkv();
    if (scene === 'split') sceneSplit();
    if (scene === 'gates') sceneGates();
    if (scene === 'memory') sceneMemory();
    if (scene === 'read') sceneRead();
    if (scene === 'mixout') sceneMixout();
    if (scene === 'mlpup') sceneMlpUp();
    if (scene === 'mlp') sceneMlp();
    if (scene === 'fullq') sceneFullQ();
    if (scene === 'fullkv') sceneFullKv();
    if (scene === 'attn') sceneAttn();
    if (scene === 'attnout') sceneAttnOut();

    cubes.forEach((c, i) => c.idx = i);
    let height = cubes.reduce((m, c) => Math.max(m, c.y + c.dy), 0);
    return { cell, margin: gap, height, cubes, byKey, arrows, ops, tokenAxes, tags, scene, version: ++versionCounter };

    // The whole model as one object, from the tokens down to the output probabilities.
    function sceneTower() {
        const slabW = 190, slabD = 44, stride = S * 4, cells = 16;
        let note = (text: string, b: IQBlock) => ops.push({ text, x: b.x + b.dx + 5, y: b.y + b.dy / 2, z: -slabD / 2, kind: 'right' });
        let noteAbove = (text: string, b: IQBlock) => ops.push({ text, x: b.x + b.dx, y: b.y, z: b.z + b.dz, kind: 'above' });
        let ids = new Float32Array(trace.manifest.tokens.map(t => t.id / index.vocab));
        let tokens = block({
            key: 'tokens', title: 'Tokens', t: 'a', x: 0, y: 0, cx: T, cy: 1, dx: slabW, dy: 6, dz: slabD, tw: T, th: 1, build: () => ids, ax: [1, 0, 0], ay: [0, 1, 0], plainScale: 1, label: false,
            axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' } },
        });
        let y = 26;
        let embed = tok('embed', 'Token embeddings', trace.overview.embed, 0, y, { dx: slabW, label: false });
        embed.z = -slabD / 2; embed.dz = slabD;
        note('Tokens', tokens); note('Token embeddings', embed);
        arrow(tokens, embed, 'v');
        if (embedW) {
            let t = embedW.embed;
            let table = block({
                key: 'embed_table', title: 'Embedding table', shape: '248,320 × 2560', t: 'w', x: -gap * 2 - t.shape[0] * cell, y, cx: t.shape[0], cy: t.shape[1], tensor: t,
                tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0],
                label: false,
                axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
            });
            noteAbove('Embedding table', table);
            arrow(table, embed);
        }
        y += embed.dy + gap;
        let prev: IQBlock = embed;
        for (let i = 0; i < index.layer_types.length; i++) {
            let t = trace.overview[`x_out_${i}`];
            let s = sampleCols(t, 1, S);
            let b = block({
                key: `x_out_small_${i}`, title: `Output of layer ${i}`, shape: 'T × 2560', t: 'i', x: 0, y, cx: T, cy: cells, dx: slabW, dy: cells * cell, dz: slabD,
                tensor: t, tw: s.w, th: s.h, build: s.build, texKey: `${pkey}/x_out_${i}`, ax: [0, 4, 0], ay: [1, 0, 0], label: false, layer: i,
                axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'dim', stride, of: 2560 } },
            });
            if (index.layer_types[i] === 'full_attention') {
                // Solid tick: the one mark that separates the two kinds of layer.
                block({ key: `tick_${i}`, title: `Layer ${i} is a full-attention layer`, t: 'a', x: -7, y, cx: 1, cy: 1, dx: 4, dy: cells * cell, dz: slabD, label: false, layer: i });
            }
            tags.push({ layer: i, x: slabW + 5, y: y + cells * cell / 2, z: -slabD / 2 });
            prev = b;
            y += b.dy + 8;
        }
        y += gap - 8;
        let norm = tok('final_norm', 'Final RMSNorm', trace.overview.final_norm, 0, y, { dx: slabW, label: false });
        norm.z = -slabD / 2; norm.dz = slabD;
        note('Final RMSNorm', norm);
        arrow(prev, norm, 'v');
        if (embedW) {
            let t = embedW.embed;
            block({
                key: 'lm_head', title: 'Output head', shape: 'the embedding table', t: 'w', x: -gap * 2 - t.shape[0] * cell, y, cx: t.shape[0], cy: t.shape[1], tensor: t,
                tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0], label: false,
                axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
            });
            noteAbove('Output head', byKey['lm_head']);
            arrow(byKey['lm_head'], norm);
        }
        y += norm.dy + gap * 1.6;
        let topK = trace.manifest.predictions[0].length;
        let probs = new Float32Array(T * topK);
        trace.manifest.predictions.forEach((row, t) => row.forEach((p, k) => probs[t * topK + k] = p.p));
        let probT: ITensor = {
            name: 'top_probs', shape: [T, topK], offset: 0, scale: 1, p99: 1, mean: 0, std: 0,
            data: new Int8Array(probs.map(p => Math.round(p * 127))),
        };
        let top = block({
            key: 'top_probs', title: 'Top 10 probabilities', shape: 'T × 10', t: 'a', x: 0, y, cx: T, cy: topK, dx: slabW, dy: topK * 4, dz: slabD, tensor: probT, tw: topK, th: T, build: () => probs,
            plainScale: 1, label: false, axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'rank', stride: 1, of: topK } },
        });
        note('Top 10 probabilities', top);
        arrow(norm, top, 'v');
    }

    function sceneInput() {
        let ids = new Float32Array(trace.manifest.tokens.map(t => t.id / index.vocab));
        let x = 0, y = 0;
        let tEmbed = embedW?.embed;
        let table: IQBlock | null = null;
        if (tEmbed) {
            let t = tEmbed;
            table = block({
                key: 'embed_table', title: 'Embedding table', shape: '248,320 × 2560', t: 'w', x, y: 26, cx: t.shape[0], cy: t.shape[1], tensor: t,
                tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0],
                axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
            });
            x = after(table, gap * 2);
        }
        let tokens = block({
            key: 'tokens', title: 'Tokens', t: 'a', x, y, cx: T, cy: 1, dx: wT, dy: 6, tw: T, th: 1, build: () => ids, ax: [1, 0, 0], ay: [0, 1, 0], label: false, plainScale: 1,
            axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' } },
        });
        tokenAxes.push({ key: 'tokens', axis: 'top' });
        let embed = tok('embed', 'Token embeddings', trace.overview.embed, x, 26);
        arrow(tokens, embed, 'v');
        if (table) arrow(table, embed);
    }

    function sceneOutput() {
        let fn = trace.overview.final_norm;
        let x = 0;
        let head: IQBlock | null = null;
        if (embedW) {
            let t = embedW.embed;
            head = block({
                key: 'lm_head', title: 'Output head (the embedding table)', shape: '248,320 × 2560', t: 'w', x, y: 0, cx: t.shape[0], cy: t.shape[1], tensor: t,
                tw: t.shape[1], th: t.shape[0], build: () => values(t), texKey: 'w/embed', ax: [0, 1, 0], ay: [1, 0, 0],
                axes: { x: { name: 'token id', stride: 4000, of: 248320 }, y: { name: 'dim', stride: S, of: 2560 } },
            });
            x = after(head, gap * 2);
        }
        let norm = tok('final_norm', 'Final RMSNorm', fn, x, 0);
        if (head) op('×', head, norm); // a dot product with every row of the table
        let topK = trace.manifest.predictions[0].length;
        let probs = new Float32Array(T * topK);
        trace.manifest.predictions.forEach((row, t) => row.forEach((p, k) => probs[t * topK + k] = p.p));
        let probT: ITensor = {
            name: 'top_probs', shape: [T, topK], offset: 0, scale: 1, p99: 1, mean: 0, std: 0,
            data: new Int8Array(probs.map(p => Math.round(p * 127))),
        };
        let top = block({
            key: 'top_probs', title: 'Top 10 probabilities', shape: 'T × 10', t: 'a', x: after(norm, gap * 2), y: 0, cx: T, cy: topK, dx: wT, dy: topK * 4, tensor: probT, tw: topK, th: T, build: () => probs,
            plainScale: 1, axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'rank', stride: 1, of: topK } },
        });
        tokenAxes.push({ key: 'top_probs', axis: 'top' });
        ops.push({ text: '→ softmax', x: x + wT + gap, y: 18, z: 0 });
    }

    // x + attention = x', x' + MLP = output. All five on one scale so the sizes can be compared.
    function sceneLayer() {
        let ref = a.x_out;
        let g = 26;
        let x0 = tok(L + 'x_in', 'Input', a.x_in, 0, 0, { scaleRef: ref });
        tokenAxes.push({ key: L + 'x_in', axis: 'top' });
        let d1 = tok(L + 'attn_out', 'Mixer', a.attn_out, after(x0, g), 0, { scaleRef: ref });
        let x1 = tok(L + 'x_mid', 'Sum', a.x_mid, after(d1, g), 0, { scaleRef: ref });
        let d2 = tok(L + 'mlp_out', 'MLP', a.mlp_out, after(x1, g), 0, { scaleRef: ref });
        let x2 = tok(L + 'x_out', 'Output', a.x_out, after(d2, g), 0, { scaleRef: ref });
        op('+', x0, d1); op('=', d1, x1); op('+', x1, d2); op('=', d2, x2);
    }

    function sceneNorm() {
        let x0 = tok(L + 'x_in', 'Input x', a.x_in, 0, 0);
        tokenAxes.push({ key: L + 'x_in', axis: 'top' });
        let g1 = wt(L + 'norm1_w', 'Scale', w.norm1_w, after(x0, gap * 1.5), 0);
        let n = tok(L + 'norm1', 'Normed x', a.norm1, after(g1, gap * 1.5), 0);
        arrow(x0, g1); arrow(g1, n);
    }

    function sceneQkv() {
        let n = tok(L + 'norm1', 'Normed x', a.norm1, 0, 0);
        let wq = wt(L + 'in_proj_qkv', 'in_proj_qkv', w.in_proj_qkv, after(n), 0);
        let raw = tok(L + 'qkv_raw', 'Projected', a.qkv_raw, after(wq), 0);
        let conv = wt(L + 'conv1d', 'Kernel', w.conv1d, after(raw), 0);
        conv.dx = 16;
        let q = tok(L + 'qkv', 'After conv, SiLU', a.qkv, after(conv), 0);
        arrow(n, wq); arrow(wq, raw); arrow(raw, conv); arrow(conv, q);
    }

    function sceneSplit() {
        let q = tok(L + 'qkv', 'Q, K, V together', a.qkv, 0, 0);
        tokenAxes.push({ key: L + 'qkv', axis: 'top' });
        let x = after(q, gap * 2.2);
        let qB = tok(L + 'q', 'Q', a.q, x, 0);
        let kB = tok(L + 'k', 'K', a.k, x, qB.dy + gap);
        let vB = tok(L + 'v', 'V', a.v, x, qB.dy + kB.dy + 2 * gap);
        arrow(q, qB); arrow(q, kB); arrow(q, vB);
    }

    function sceneGates() {
        const rowH = 32 * cell;
        let n = tok(L + 'norm1', 'Normed x', a.norm1, 0, (2 * rowH + 2.4 * gap - a.norm1.shape[1] / S * cell) / 2);
        let x1 = after(n, gap * 1.6);
        // Row 2: retention. e^g is stored as g, so convert it here for display.
        let y2 = rowH + 2.4 * gap;
        let wa = wt(L + 'in_proj_a', 'in_proj_a', w.in_proj_a, x1, y2);
        let aB = tok(L + 'a', 'a', a.a, after(wa), y2, { stride: 1 });
        let dt = wt(L + 'dt_bias', 'dt_bias', w.dt_bias, after(aB, gap * 1.3), y2);
        let aLog = wt(L + 'A_log', 'A_log', w.A_log, after(dt, gap * 1.8), y2);
        let lastX = after(aLog, gap * 1.8);
        let vals = values(a.g).map(v => Math.exp(v));
        let egT: ITensor = {
            name: 'eg', shape: a.g.shape, offset: 0, scale: 1, p99: 1, mean: vals.reduce((p, c) => p + c, 0) / vals.length, std: 0,
            data: new Int8Array(vals.map(v => Math.round(v * 127))),
        };
        egT.std = Math.sqrt(vals.reduce((p, c) => p + (c - egT.mean) ** 2, 0) / vals.length);
        let eg = block({
            key: L + 'eg', title: 'e^g', shape: 'T × 32', t: 'a', x: lastX, y: y2, cx: T, cy: 32, dx: wT, tensor: egT, tw: 32, th: T, build: () => vals,
            plainScale: 1, axes: { x: { name: 'token', stride: 1, of: T, kind: 'token' }, y: { name: 'head', stride: 1, of: 32 } },
        });
        arrow(n, wa); arrow(wa, aB); arrow(aB, dt); arrow(dt, aLog); arrow(aLog, eg);
        // Row 1: write gate.
        let wb = wt(L + 'in_proj_b', 'in_proj_b', w.in_proj_b, x1, 0);
        let bB = tok(L + 'b', 'b', a.b, after(wb), 0, { stride: 1 });
        let beta = tok(L + 'beta', 'β', a.beta, lastX, 0, { stride: 1, tt: 'a', plainScale: 1 });
        arrow(n, wb); arrow(wb, bB); arrow(bB, beta);
        tokenAxes.push({ key: L + 'beta', axis: 'top' });
    }

    function sceneMemory() {
        const H = 32, C = 16, side = 54, step = side + 12;
        let st = sampleCols(a.state, 3, 1);     // rows: T * H * 16, cols: 16
        // One colour scale for all heads and all tokens, so growth over tokens is visible.
        let scale = 1 / (a.state.p99 || 1);
        for (let h = 0; h < H; h++) {
            let col = h % 8, row = Math.floor(h / 8);
            block({
                key: `${L}state${h}`, title: `Memory S, head ${h}`, t: 'i', x: col * step, y: row * step, cx: C, cy: C, dx: side, dy: side,
                tensor: a.state, tw: st.w, th: st.h, build: st.build, texKey: `${pkey}/${L}state`,
                ax: [1, 0, 0, 0], ay: [0, 1, 0, C * (h + H * tokenCursor)], label: false, plainScale: scale,
                slice: { offset: (tokenCursor * H + h) * C * C, rows: C, cols: C, note: `Head ${h} of ${H}, after token ${tokenCursor}. One value in 8 per axis.` },
                axes: { x: { name: 'value dim', stride: 8, of: 128 }, y: { name: 'key dim', stride: 8, of: 128 } },
            });
        }
    }

    function sceneRead() {
        const side = 150, small = 34, stepS = small + 7;
        let mean = meanOverHeads(a.read, `${L}read_mean`);
        matrix(`${L}read_mean`, 'Mean over 32 heads', mean, -1, 0, 0, side, 'i', {
            texKey: `${pkey}/${L}read_mean`, note: 'Mean over the 32 heads.', xName: 'written at', yName: 'read at',
        });
        tokenAxes.push({ key: `${L}read_mean`, axis: 'top' }, { key: `${L}read_mean`, axis: 'left' });
        let x0 = side + gap * 2.2;
        let scale = 1 / (a.read.p99 || 1);
        for (let h = 0; h < 32; h++) {
            matrix(`${L}read${h}`, `Read weights, head ${h}`, a.read, h, x0 + (h % 8) * stepS, Math.floor(h / 8) * stepS, small, 'i', {
                label: false, scale, texKey: `${pkey}/${L}read`, note: `Head ${h} of 32.`, xName: 'written at', yName: 'read at',
            });
        }
    }

    function sceneMixout() {
        let h = a.z.shape[1] / S * cell;
        let y2 = h + gap * 2.2;
        let wz = wt(L + 'in_proj_z', 'in_proj_z', w.in_proj_z, 0, 0);
        let z = tok(L + 'z', 'z', a.z, after(wz), 0);
        let core = tok(L + 'core', 'Memory output', a.core, 0, y2);
        let nw = wt(L + 'norm_w', 'Norm scale', w.norm_w, z.x, y2, { dy: h });
        let g = tok(L + 'gated', 'Normed and gated', a.gated, after(z, gap * 2.4), (y2 + h - h) / 2);
        let wo = wt(L + 'out_proj', 'out_proj', w.out_proj, after(g), g.y);
        let out = tok(L + 'attn_out', 'Output', a.attn_out, after(wo), g.y);
        arrow(wz, z); arrow(core, nw); arrow(z, g); arrow(nw, g); arrow(g, wo); arrow(wo, out);
    }

    function sceneMlpUp() {
        let wH = 64 * cell, vH = Math.max(T * cell, 24), g2 = gap * 2.6;
        let nw = wt(L + 'norm2_w', 'Norm scale', w.norm2_w, 0, 0);
        let n = tok(L + 'norm2', 'Normed sum', a.norm2, after(nw, gap * 2.6), 0);
        arrow(nw, n);
        let x = after(n, gap * 1.8);
        let gw = wt(L + 'gate_w', 'gate_proj', w.gate_w, x, 0, { flip: true });
        let gB = wide(L + 'gate', 'gate', a.gate, x, wH + g2);
        let y = wH + vH + 2 * g2;
        let uw = wt(L + 'up_w', 'up_proj', w.up_w, x, y, { flip: true });
        let uB = wide(L + 'up', 'up', a.up, x, y + wH + g2);
        arrow(n, gw); arrow(n, uw); arrow(gw, gB, 'v'); arrow(uw, uB, 'v');
    }

    function sceneMlp() {
        let vH = Math.max(T * cell, 24), g2 = gap * 2.6;
        let gB = wide(L + 'gate', 'gate', a.gate, 0, 0);
        let uB = wide(L + 'up', 'up', a.up, 0, vH + g2);
        let m = wide(L + 'glu', 'SiLU(gate) × up', a.glu, 0, 2 * vH + 2 * g2);
        let d = wt(L + 'down_w', 'down_proj', w.down_w, 0, 3 * vH + 3 * g2);
        let out = tok(L + 'mlp_out', 'MLP output', a.mlp_out, after(d, gap * 1.2), d.y);
        arrow(gB, m, 'side'); arrow(uB, m, 'v'); arrow(m, d, 'v'); arrow(d, out);
    }

    function sceneFullQ() {
        let n = tok(L + 'norm1', 'Normed x', a.norm1, 0, 0);
        let qH = a.q.shape[1] * a.q.shape[2] / S * cell;
        let wq = wt(L + 'q_proj', 'q_proj', w.q_proj, after(n, gap * 1.6), 0);
        let qn = wt(L + 'q_norm_w', 'Norm scale', w.q_norm_w, after(wq, gap * 1.6), 0, { dy: qH });
        let q = tok(L + 'q', 'Q', a.q, after(qn, gap * 1.8), 0);
        let g = tok(L + 'q_gate', 'Gate, before sigmoid', a.q_gate, q.x, qH + gap * 1.5);
        arrow(n, wq); arrow(wq, qn); arrow(qn, q); arrow(wq, g);
    }

    function sceneFullKv() {
        let n = tok(L + 'norm1', 'Normed x', a.norm1, 0, 0);
        let x = after(n, gap * 1.6);
        let kH = 26 * cell;
        let wk = wt(L + 'k_proj', 'k_proj', w.k_proj, x, 0);
        let kn = wt(L + 'k_norm_w', 'Norm scale', w.k_norm_w, after(wk, gap * 1.6), 0, { dy: kH });
        let k = tok(L + 'k', 'K', a.k, after(kn, gap * 1.8), 0);
        let y = kH + gap * 2;
        let wv = wt(L + 'v_proj', 'v_proj', w.v_proj, x, y);
        let v = tok(L + 'v', 'V', a.v, k.x, y);
        arrow(n, wk); arrow(n, wv); arrow(wk, kn); arrow(kn, k); arrow(wv, v);
    }

    function sceneAttn() {
        const side = 150, small = 42, stepS = small + 8;
        let mean = meanOverHeads(a.attn_w, `${L}attn_mean`);
        matrix(`${L}attn_mean`, 'Mean over 16 heads', mean, -1, 0, 0, side, 'a', {
            texKey: `${pkey}/${L}attn_mean`, note: 'Mean over the 16 heads. Each row sums to 1.', xName: 'attends to', yName: 'at', scale: 1 / (mean.p99 || 1),
        });
        tokenAxes.push({ key: `${L}attn_mean`, axis: 'top' }, { key: `${L}attn_mean`, axis: 'left' });
        let x0 = side + gap * 2.2;
        for (let h = 0; h < 16; h++) {
            matrix(`${L}attn${h}`, `Attention, head ${h}`, a.attn_w, h, x0 + (h % 4) * stepS, Math.floor(h / 4) * stepS, small, 'a', {
                label: false, scale: 1, texKey: `${pkey}/${L}attn_w`, note: `Head ${h} of 16. Each row sums to 1.`, xName: 'attends to', yName: 'at',
            });
        }
    }

    function sceneAttnOut() {
        let g = tok(L + 'q_gate', 'Gate, before sigmoid', a.q_gate, 0, 0);
        tokenAxes.push({ key: L + 'q_gate', axis: 'top' });
        let wo = wt(L + 'o_proj', 'o_proj', w.o_proj, after(g, gap * 1.6), 0);
        let out = tok(L + 'attn_out', 'Output', a.attn_out, after(wo), 0);
        arrow(g, wo); arrow(wo, out);
    }
}
