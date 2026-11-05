import React from 'react';
import { fmt, IIndex, IPred, ITensor, ITensorSet, ITrace, showTok } from './data';
import { SceneId } from './layout';

export type LayerKind = 'none' | 'any' | 'linear' | 'full';

export interface ICtx {
    index: IIndex;
    trace: ITrace;
    layer: number;
    act: ITensorSet | null;
    w: ITensorSet | null;
    tokenCursor: number;
}

export interface IStep {
    id: string;
    num: string;
    title: string;
    layer: LayerKind;
    scene: SceneId;
    legend: 'div' | 'seq' | 'both';
    tokenSlider?: boolean;
    lensTags?: boolean;
    body: (c: ICtx) => React.ReactNode;
    caption: (c: ICtx) => React.ReactNode;
}

// ---- Numbers read from the loaded trace, so the text stays true for every prompt and layer ----

const Tk: React.FC<{ t: string }> = ({ t }) => <code className="q-t">{showTok(t)}</code>;
const Eq: React.FC<{ children: React.ReactNode }> = ({ children }) => <div className="q-eq">{children}</div>;

function floats(t: ITensor) {
    let f = t.scale / 127, o = new Float64Array(t.data.length);
    for (let i = 0; i < o.length; i++) o[i] = t.data[i] * f;
    return o;
}
function median(a: number[]) {
    let s = [...a].sort((x, y) => x - y), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function rowNorms(t: ITensor) {
    let v = floats(t), T = t.shape[0], d = v.length / T, out: number[] = [];
    for (let r = 0; r < T; r++) {
        let s = 0;
        for (let c = 0; c < d; c++) s += v[r * d + c] ** 2;
        out.push(Math.sqrt(s));
    }
    return out;
}
function normRatio(a: ITensor, b: ITensor) {
    let x = rowNorms(a), y = rowNorms(b);
    return median(x.map((v, i) => v / y[i]));
}
const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
const mean = (v: ArrayLike<number>) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s / v.length; };
const sig = (x: number) => 1 / (1 + Math.exp(-x));
function topOf(row: number[], tokens: string[], n: number) {
    return row.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, n).map(o => ({ t: tokens[o.i], v: o.v, i: o.i }));
}
const toks = (c: ICtx) => c.trace.manifest.tokens.map(t => t.text);
const last = (c: ICtx): IPred[] => c.trace.manifest.predictions[c.trace.manifest.T - 1];

function firstStable(c: ICtx) {
    let lens = c.trace.manifest.lens, id = last(c)[0].id, first = -1, stable = -1;
    lens.forEach((l, i) => { if (first < 0 && l[0].id === id) first = i; });
    for (let i = lens.length - 1; i >= 0 && lens[i][0].id === id; i--) stable = i;
    return { first, stable };
}

const SampleNote = ' Long axes show one value in 40.';

export const STEPS: IStep[] = [
    {
        id: 'overview', num: '1', title: 'Introduction', layer: 'none', scene: 'tower', legend: 'div',
        body: c => <>
            <p>What does a single forward pass of a language model look like? In this walkthrough, we record every intermediate value computed by Qwen3.5-4B while it reads a short prompt, and draw those values in three dimensions, following Bycroft's visualization of GPT-style models [1].</p>
            <p>Qwen3.5-4B departs from the GPT design in one respect that shapes the rest of this document. Of its 32 layers, only 8 use full self-attention. The other 24 use Gated DeltaNet, which has no attention matrix. They instead read tokens in order and maintain a fixed-size memory.</p>
            <p>Our subject is narrow. Long, Sebo et al. distinguish several entities that one might study in a language model, including the model, a persona, an instance and a forward pass [2]. We describe only the last. We also pass the prompts as raw text, without a chat template, so no Assistant turn takes place.</p>
            <p>§8 states what these figures can and cannot show.</p>
        </>,
        caption: () => <>The whole model, read from top to bottom: tokens and their embeddings, the residual stream after each of the 32 layers, the final norm and the output probabilities. Each slab shows tokens along the horizontal axis and one value in 160 along the vertical axis. A dark tick marks a full-attention layer. Click a slab to open that layer, or any other block for its values. Drag to pan, shift-drag to rotate, and scroll to zoom.</>,
    },
    {
        id: 'input', num: '2', title: 'Input', layer: 'none', scene: 'input', legend: 'div',
        body: c => <>
            <p>The prompt is split into {c.trace.manifest.T} tokens. Each token selects one row of a table with 248,320 rows of 2,560 values. The table holds 636 million parameters, about 15% of the text model.</p>
            <p>GPT-2 adds a learned position vector at this point. Qwen3.5-4B does not. Position enters later, through the order in which linear layers read tokens (§4) and through the rotation applied to queries and keys in full-attention layers (§6).</p>
        </>,
        caption: () => <>Token embeddings (right) are rows of the embedding table (left). The table is shown at one row in 4,000 and one column in 40, so most selected rows are not among those drawn.</>,
    },
    {
        id: 'layer', num: '3.1', title: 'A layer computes two corrections', layer: 'any', scene: 'layer', legend: 'div',
        body: c => {
            let r1 = c.act && normRatio(c.act.attn_out, c.act.x_in), r2 = c.act && normRatio(c.act.mlp_out, c.act.x_mid);
            let kind = c.index.layer_types[c.layer] === 'full_attention' ? 'attention' : 'Gated DeltaNet';
            return <>
                <p>Every layer has the same outer form. It computes a token-mixing term and adds it to the residual stream, then computes an MLP term and adds that too. Tokens exchange information only in the first term.</p>
                <Eq>x′ = x + mixer(norm(x))<br />y = x′ + mlp(norm(x′))</Eq>
                <p>How large are these corrections? At layer {c.layer} on this prompt, the {kind} term has a median norm of {r1 !== null && pct(r1)} of the incoming vector, and the MLP term {r2 !== null && pct(r2)} of the sum. Figure 3 shares one colour scale across all five blocks, so these differences in size can be seen directly.</p>
                <p className="q-note">The names "memory", "attention" and "gate" refer to operations in the architecture. We do not intend them to imply cognitive counterparts (§8).</p>
            </>;
        },
        caption: c => <>The two additions in layer {c.layer}, where the mixer is {c.index.layer_types[c.layer] === 'full_attention' ? 'full attention' : 'Gated DeltaNet'}. All five blocks use the 99th percentile of the output as their colour limit.{SampleNote}</>,
    },
    {
        id: 'norm', num: '3.2', title: 'RMSNorm', layer: 'any', scene: 'norm', legend: 'div',
        body: c => {
            let rms = c.act && median(rowNorms(c.act.x_in)) / Math.sqrt(2560);
            return <>
                <p>Before each term, the vector for each token is divided by its root mean square, then multiplied dimension by dimension by a learned scale.</p>
                <Eq>norm(x) = x / rms(x) · weight</Eq>
                <p>Unlike LayerNorm, RMSNorm subtracts no mean and has no bias. At layer {c.layer}, the incoming vectors have a median root mean square of {rms !== null && fmt(rms)}; the learned scale has a mean of {c.w && fmt(c.w.norm1_w.full_mean ?? 0)} over all 2,560 dimensions.</p>
            </>;
        },
        caption: c => <>Input, learned scale and normalized output in layer {c.layer}. Each block uses its own colour limit, so these colours cannot be compared across blocks.{SampleNote}</>,
    },
    {
        id: 'qkv', num: '4.1', title: 'Projection and convolution', layer: 'linear', scene: 'qkv', legend: 'div',
        body: c => {
            let tap: number[] = [];
            if (c.w) {
                let v = floats(c.w.conv1d), [r, k] = c.w.conv1d.shape;
                for (let j = 0; j < k; j++) { let s = 0; for (let i = 0; i < r; i++) s += Math.abs(v[i * k + j]); tap.push(s / r); }
            }
            return <>
                <p>A Gated DeltaNet layer first maps 2,560 dimensions to 8,192. A short causal convolution then acts on each of those channels separately, so that each output depends on the current token and the three before it. A SiLU nonlinearity follows.</p>
                {tap.length === 4 && <p>In the sampled channels of layer {c.layer}, the mean absolute kernel weight is {fmt(tap[3])} for the current token, and {fmt(tap[2])}, {fmt(tap[1])} and {fmt(tap[0])} for the tokens one, two and three positions back.</p>}
            </>;
        },
        caption: c => <>From the normalized input to the convolved channels in layer {c.layer}. The kernel has four taps; its rightmost column is the current token.{SampleNote}</>,
    },
    {
        id: 'split', num: '4.2', title: 'Queries, keys and values', layer: 'linear', scene: 'split', legend: 'div',
        body: () => <>
            <p>The 8,192 channels are divided into 16 query heads, 16 key heads and 32 value heads, each 128 wide. Queries and keys are then repeated, so that each of the 32 value heads has its own pair. Before use, queries and keys are normalized to unit length, and queries are also divided by √128.</p>
        </>,
        caption: c => <>Division of the convolved channels into Q, K and V in layer {c.layer}. Heights are to scale.{SampleNote}</>,
    },
    {
        id: 'gates', num: '4.3', title: 'How much to write and how much to retain', layer: 'linear', scene: 'gates', legend: 'both',
        body: c => {
            let beta = c.act && floats(c.act.beta), g = c.act && floats(c.act.g);
            let worst = '';
            let meanEg = 0, minEg = 0;
            if (g && c.act) {
                let [T, H] = c.act.g.shape, row: number[] = [];
                for (let t = 0; t < T; t++) { let s = 0; for (let h = 0; h < H; h++) s += Math.exp(g[t * H + h]); row.push(s / H); }
                let idx = row.indexOf(Math.min(...row));
                worst = toks(c)[idx];
                meanEg = mean(row); minEg = row[idx];
            }
            return <>
                <p>Each token sets two values per head. The first, β, controls how strongly it writes to memory. The second, e<sup>g</sup>, controls how much of the existing memory is retained.</p>
                <Eq>β = sigmoid(in_proj_b(x))</Eq>
                <Eq>g = −exp(A_log) · softplus(in_proj_a(x) + dt_bias)</Eq>
                <p>Because g is always negative, e<sup>g</sup> lies between 0 and 1. Older contents can only shrink, except through later writes.</p>
                {beta && <p>At layer {c.layer} on this prompt, β has a mean of {fmt(mean(beta))} and e<sup>g</sup> a mean of {fmt(meanEg)}. Retention is lowest at <Tk t={worst} />, with a mean of {fmt(minEg)} over heads.</p>}
            </>;
        },
        caption: c => <>The weights and values that produce β and e<sup>g</sup> in layer {c.layer}, for all 32 heads. In a, b, β and e<sup>g</sup>, each row is a head and each column a token. β and e<sup>g</sup> use the scale from 0 to 1.</>,
    },
    {
        id: 'memory', num: '4.4', title: 'The memory', layer: 'linear', scene: 'memory', legend: 'div', tokenSlider: true,
        body: c => {
            let sn = c.act && floats(c.act.state_norm), rows: number[] = [];
            if (sn && c.act) {
                let [T, H] = c.act.state_norm.shape;
                for (let t = 0; t < T; t++) { let s = 0; for (let h = 0; h < H; h++) s += sn[t * H + h]; rows.push(s / H); }
            }
            return <>
                <p>Each of the 32 heads holds a matrix S of 128 × 128 values. Its size does not depend on the length of the prompt. Each token updates S in three steps and then reads from it:</p>
                <Eq>S ← e<sup>g</sup> S</Eq>
                <Eq>δ = β (v − Sᵀk)</Eq>
                <Eq>S ← S + k δᵀ</Eq>
                <Eq>out = Sᵀq</Eq>
                <p>The term v − Sᵀk is an error: the difference between the token's value and what the memory currently returns for its key. A token therefore writes only the part that the memory does not already predict.</p>
                <p>S starts at zero, so after the first token each head holds a rank-one matrix, which appears as a striped pattern in Figure 8. Each later token adds one further rank-one term.</p>
                {rows.length > 0 && <p>At layer {c.layer}, the mean Frobenius norm of S over heads rises from {fmt(rows[0])} after the first token to {fmt(rows[rows.length - 1])} after the last, and is {fmt(rows[c.tokenCursor] ?? 0)} at the token selected above.</p>}
            </>;
        },
        caption: c => <>The memory S of all 32 heads in layer {c.layer}, after the selected token. One value in 8 is shown per axis. Rows are key dimensions and columns are value dimensions. One colour limit is used for all heads and tokens, so growth across tokens can be seen.</>,
    },
    {
        id: 'read', num: '4.5', title: 'A read matrix for Gated DeltaNet', layer: 'linear', scene: 'read', legend: 'div',
        body: c => {
            let top: { t: string; v: number }[] = [], neg = 0, gap = 0;
            if (c.act) {
                let r = floats(c.act.read), [H, T] = c.act.read.shape, row = new Array(T).fill(0);
                for (let h = 0; h < H; h++) for (let s = 0; s < T; s++) row[s] += r[h * T * T + (T - 1) * T + s] / H;
                top = topOf(row, toks(c), 3);
                let n = 0, tot = 0;
                for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let s = 0; s <= t; s++) { tot++; if (r[h * T * T + t * T + s] < 0) n++; }
                neg = n / tot;
                gap = c.trace.manifest.checks[`L${c.layer}_read_decomposition`] ?? 0;
            }
            return <>
                <p>Gated DeltaNet does not compute an attention matrix. Can its memory read nonetheless be expressed as a relation between tokens? Expanding the update rule shows that it can, because the output at token <i>t</i> is exactly a weighted sum of the corrections δ written by earlier tokens:</p>
                <Eq>out<sub>t</sub> = Σ<sub>s≤t</sub> read[t, s] δ<sub>s</sub></Eq>
                <Eq>read[t, s] = exp(Σ<sub>r=s+1..t</sub> g<sub>r</sub>) (q<sub>t</sub> · k<sub>s</sub>)</Eq>
                <p>We verified this expansion against the model's own kernel: on this layer, the sum differs from the kernel's output by at most {pct(gap, 2)} of the largest output value, within the rounding of its bf16 output.</p>
                {top.length > 0 && <p>At layer {c.layer}, averaged over heads, the final token reads most strongly from {top.map((o, i) => <React.Fragment key={i}>{i > 0 && (i === top.length - 1 ? ' and ' : ', ')}<Tk t={o.t} /></React.Fragment>)}. In {pct(neg)} of causal cells, the read weight is negative.</p>}
                <p className="q-note">We derived this matrix for this document; the model does not compute it. It is not a probability distribution, and the δ it multiplies themselves depend on earlier tokens. It describes the arithmetic of the read, and does not show that any token mattered for the prediction.</p>
            </>;
        },
        caption: c => <>Left: read[t, s] in layer {c.layer}, averaged over heads, where rows are the reading token and columns the writing token. Right: each of the 32 heads, on a shared colour limit.</>,
    },
    {
        id: 'mixout', num: '4.6', title: 'Return to the residual stream', layer: 'linear', scene: 'mixout', legend: 'div',
        body: c => {
            let off = 0;
            if (c.act) { let z = floats(c.act.z); let n = 0; z.forEach(v => { if (v < 0) n++; }); off = n / z.length; }
            return <>
                <p>The memory output is normalized within each head and multiplied by a gate computed from the normalized input. A projection then maps 4,096 dimensions back to 2,560.</p>
                <Eq>out / rms(out) · weight · SiLU(z)</Eq>
                {c.act && <p>At layer {c.layer}, z is negative for {pct(off)} of entries, where SiLU is small and slightly negative rather than zero.</p>}
            </>;
        },
        caption: c => <>From memory output to the term added to the residual stream in layer {c.layer}. in_proj_z produces the gate z from the normalized input. Each block uses its own colour limit.{SampleNote}</>,
    },
    {
        id: 'mlpup', num: '5.1', title: 'The MLP: two projections', layer: 'any', scene: 'mlpup', legend: 'div',
        body: c => {
            let frac = 0;
            if (c.act) { let g = floats(c.act.gate); let n = 0; g.forEach(v => { if (v > 0) n++; }); frac = n / g.length; }
            return <>
                <p>The second term in every layer is a SwiGLU MLP. It begins with two separate projections, each of which expands a token from 2,560 to 9,216 dimensions.</p>
                <p>Each projection holds 23.6 million parameters. Including the return projection in §5.2, the MLP has more parameters than the mixer in the same layer.</p>
                {c.act && <p>At layer {c.layer}, the output of gate_proj is positive for {pct(frac)} of entries.</p>}
            </>;
        },
        caption: c => <>The two projections in layer {c.layer}, each applied to the normalized sum. In gate and up, rows are tokens and columns are dimensions.{SampleNote}</>,
    },
    {
        id: 'mlp', num: '5.2', title: 'The MLP: gating and return', layer: 'any', scene: 'mlp', legend: 'div',
        body: c => <>
            <p>One projection gates the other, entry by entry, and a third projection returns to 2,560 dimensions.</p>
            <Eq>mlp(x) = down( SiLU(gate(x)) × up(x) )</Eq>
            <p>This term acts on each token independently. Tokens exchange information only in the mixer.</p>
        </>,
        caption: c => <>The product of gate and up, and its projection back to 2,560 dimensions, in layer {c.layer}.{SampleNote}</>,
    },
    {
        id: 'fullq', num: '6.1', title: 'Queries in full attention', layer: 'full', scene: 'fullq', legend: 'div',
        body: () => <>
            <p>Every fourth layer uses self-attention, with several changes from the GPT design. The first concerns queries: q_proj produces twice the width needed for them. One half becomes the 16 query heads, and the other half becomes an output gate, used in §6.4.</p>
            <p>Queries then pass through their own RMSNorm. RoPE rotates a quarter of each query according to token position.</p>
        </>,
        caption: c => <>q_proj, the query norm scale, the queries and the gate in layer {c.layer}. The gate is shown before its sigmoid.{SampleNote}</>,
    },
    {
        id: 'fullkv', num: '6.2', title: 'Keys and values in full attention', layer: 'full', scene: 'fullkv', legend: 'div',
        body: () => <>
            <p>There are fewer keys and values than queries: 16 query heads share 4 key heads and 4 value heads, so k_proj and v_proj are each an eighth the size of q_proj. Keys also pass through RMSNorm and are rotated by position in the same way as queries. Values are not.</p>
        </>,
        caption: c => <>k_proj, the key norm scale, the keys, v_proj and the values in layer {c.layer}.{SampleNote}</>,
    },
    {
        id: 'attn', num: '6.3', title: 'Attention', layer: 'full', scene: 'attn', legend: 'seq',
        body: c => {
            let tk = toks(c), T = c.trace.manifest.T;
            let cur = c.trace.manifest.attn_last.find(a => a.layer === c.layer);
            let top = cur ? topOf(cur.weights, tk, 3) : [];
            return <>
                <p>These matrices are computed by the model. Row <i>t</i> is a probability distribution over tokens up to and including <i>t</i>.</p>
                <Eq>softmax(Q Kᵀ / √256)</Eq>
                {top.length > 0 && <p>At layer {c.layer}, averaged over heads, the final token <Tk t={tk[T - 1]} /> places its largest weights on {top.map((o, i) => <React.Fragment key={i}>{i > 0 && (i === top.length - 1 ? ' and ' : ', ')}<Tk t={o.t} /> ({fmt(o.v)})</React.Fragment>)}.</p>}
                <p>The largest weight changes from layer to layer:</p>
                <ol className="q-layers">
                    {c.trace.manifest.attn_last.map(a => { let o = topOf(a.weights, tk, 1)[0]; return <li key={a.layer}><span>L{a.layer}</span><Tk t={o.t} /><span>{fmt(o.v)}</span></li>; })}
                </ol>
                <p className="q-note">We report these weights as a description. They show where a head reads, not what it takes from there or whether that affected the output. Averaging over heads also discards differences between them, which can be seen in the right of the figure.</p>
            </>;
        },
        caption: c => <>Left: attention in layer {c.layer}, averaged over 16 heads, on a colour limit set by this matrix. Right: each head, with a colour scale from 0 to 1. Rows are the attending token and columns the attended token; later tokens are masked.</>,
    },
    {
        id: 'attnout', num: '6.4', title: 'The output gate', layer: 'full', scene: 'attnout', legend: 'div',
        body: c => {
            let m = 0;
            if (c.act) { let g = floats(c.act.q_gate); let s = 0; g.forEach(v => { s += sig(v); }); m = s / g.length; }
            return <>
                <p>The output of each head is multiplied, entry by entry, by sigmoid(gate). A final projection maps 4,096 dimensions back to 2,560, and the result is added to the residual stream.</p>
                {c.act && <p>At layer {c.layer}, sigmoid(gate) has a mean of {fmt(m)}.</p>}
            </>;
        },
        caption: c => <>The gate before its sigmoid, the o_proj weights and the resulting output in layer {c.layer}. Each block uses its own colour limit.{SampleNote}</>,
    },
    {
        id: 'lens', num: '7.1', title: 'A readout at each layer', layer: 'none', scene: 'tower', legend: 'div', lensTags: true,
        body: c => {
            let { first, stable } = firstStable(c);
            return <>
                <p>What would each layer predict if the model stopped there? We apply the logit lens [3], which passes the residual stream of each layer through the final norm and output head.</p>
                <p>On this prompt, the model's eventual answer <Tk t={last(c)[0].text} /> first ranks highest at layer {first}, and remains highest from layer {stable}. Many earlier readouts have no apparent relation to the prompt.</p>
                <p className="q-note">We do not interpret those earlier readouts. The logit lens assumes that all layers share the same coordinates. Gurnee et al. report that it gives uninterpretable readouts at earlier layers, and introduce the Jacobian lens to correct for changes in representation across layers [4]. We have not applied that lens here.</p>
            </>;
        },
        caption: () => <>The whole model of Figure 1, with the logit-lens readout at the final token of each layer and its probability. A dark tick marks a full-attention layer.</>,
    },
    {
        id: 'output', num: '7.2', title: 'The next token', layer: 'none', scene: 'output', legend: 'both',
        body: c => <>
            <p>The final residual stream passes through one more RMSNorm, and then through a dot product with the same table that embedded the input. This gives one score for each of the 248,320 tokens, and softmax converts the scores to probabilities.</p>
            <p>The model predicts <Tk t={last(c)[0].text} /> with probability {fmt(last(c)[0].p)}. The ten most probable tokens are:</p>
            <ol className="q-preds">
                {last(c).map(p => <li key={p.id}>
                    <Tk t={p.text} />
                    <span className="q-pred-bar"><span style={{ width: `${Math.max(0.5, p.p * 100)}%` }} /></span>
                    <span className="q-pred-p">{(p.p * 100).toFixed(1)}%</span>
                </li>)}
            </ol>
        </>,
        caption: () => <>The output head (left) and final norm (centre) produce a score for every token. Right: the ten largest probabilities at each position, in order of rank, on a scale from 0 to 1. The head is shown at one row in 4,000.</>,
    },
    {
        id: 'limits', num: '8', title: 'What these figures do and do not show', layer: 'none', scene: 'tower', legend: 'div',
        body: () => <>
            <p><b>They are descriptive.</b> We show values computed on four prompts. We made no interventions, so no figure establishes that a given value caused a given output.</p>
            <p><b>Structure does not establish function.</b> Long, Sebo et al. separate architectural evidence, about which computations a design makes possible, from interpretability evidence, about which computations occur [2]. As they note, two systems with identical architectures might implement very different computations. The layout of these figures is architectural; the values are one trace of one model.</p>
            <p><b>We have not tested for a global workspace.</b> Gurnee et al. report that verbalizable representations in Claude models show functional properties of a global workspace, and locate them in an intermediate band of layers [4]. Their measurements do not transfer to the layers drawn here, and we have run none of their experiments.</p>
            <p><b>This architecture differs in a relevant way.</b> Gurnee et al. describe two axes along which information moves in a transformer, depth and token position, and observe that attention can retrieve any earlier representation [4]. In 24 of the 32 layers here, the token axis instead passes through a fixed-size memory that decays. We note this as a structural difference, and have not tested whether it has any functional consequence.</p>
            <p><b>Nothing here bears directly on consciousness, sentience or welfare.</b> Words like "memory" and "attention" name operations. Whether such operations correspond to anything in human cognition depends on the level of abstraction at which the comparison is made, which Long, Sebo et al. call the specificity problem [2].</p>
            <p><b>Technical limits.</b> Activations are stored at 8 bits per value, with an error of at most 0.4% of each tensor's largest value. Long axes in the 3D view show one value in 40, and memory one value in 8; each drawn cell is an exact stored value rather than an average. Clicking a block shows all its stored values. Weights are stored only at the sampled resolution, although their statistics cover the full matrices. Not shown are the 24-block vision tower, which text prompts do not use, and the checkpoint's mtp.* weights, which this model code does not load.</p>
            <h3>References</h3>
            <ol className="q-refs">
                <li>Bycroft, B. <a href="https://bbycroft.net/llm" target="_blank" rel="noreferrer">LLM Visualization</a>. The renderer here is adapted from its <a href="https://github.com/bbycroft/llm-viz" target="_blank" rel="noreferrer">source</a> (MIT licence).</li>
                <li>Long, R., Sebo, J., et al. (2026). <a href="https://nonhumanminds.org/wp-content/uploads/2026/07/Studying-AI-Welfare-Empirically.pdf" target="_blank" rel="noreferrer">Studying AI Welfare Empirically</a>. NYU Center for Mind, Ethics, and Policy and Eleos AI Research.</li>
                <li>nostalgebraist (2020). <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noreferrer">Interpreting GPT: the logit lens</a>.</li>
                <li>Gurnee, W., Sofroniew, N., et al. (2026). <a href="https://transformer-circuits.pub/2026/workspace/" target="_blank" rel="noreferrer">Verbalizable Representations Form a Global Workspace in Language Models</a>. Anthropic.</li>
            </ol>
        </>,
        caption: () => <>The whole model of Figure 1, repeated for reference.</>,
    },
];
