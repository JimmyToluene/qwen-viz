import React from 'react';

export type LayerSel = 'none' | 'linear' | 'full';

export interface IStep {
    id: string;
    title: string;
    section: string;
    layer: LayerSel;
    groups: string[];   // block groups to show at full strength; [] = everything
    fit?: string[];     // groups the camera frames (defaults to `groups`)
    view?: 'front' | 'angled';
    tokenSlider?: boolean;
    predictions?: boolean;
    body: React.ReactNode;
}

const Eq: React.FC<{ children: React.ReactNode }> = ({ children }) =>
    <div className="q-eq">{children}</div>;

export const STEPS: IStep[] = [
    {
        id: 'overview', section: 'Overview', title: 'The whole model', layer: 'none', groups: [], fit: ['tower', 'input', 'output'],
        body: <>
            <p>This is one real forward pass of <b>Qwen3.5-4B</b> on the prompt above. Every coloured cell is a real number from the model: a learned weight or a value computed for this prompt.</p>
            <p>Each slab is the output of one layer, read from top to bottom. There are 32. They come in a repeating pattern: three <span className="q-linear">linear-attention</span> layers, then one <span className="q-full">full-attention</span> layer.</p>
            <p>Click a slab or layer tag to open that layer, and click again to close it. Click any block for its detail. Drag to pan, shift-drag or right-drag to rotate, and scroll to zoom.</p>
            <p className="q-note">Everything is to scale, but a block with thousands of values can't be drawn cell by cell, so long axes show one real value in every 40. Clicking a block opens all of its activation values.</p>
        </>,
    },
    {
        id: 'embed', section: 'Input', title: 'Tokens become vectors', layer: 'none', groups: ['input'],
        body: <>
            <p>The prompt is split into tokens. Each one selects a row from a table of 248,320 rows, each 2,560 numbers long. That table holds 636 million parameters, about one seventh of the text model.</p>
            <p>GPT-2 would now add a learned position vector. Qwen3.5 does not. Position comes from elsewhere: the linear-attention layers read tokens in order, and the full-attention layers rotate their queries and keys by position.</p>
        </>,
    },
    {
        id: 'linear', section: 'Linear-attention layer', title: 'Inside a linear-attention layer', layer: 'linear',
        groups: ['norm1', 'qkv', 'split', 'gates', 'memory', 'mixout', 'mlp'], view: 'angled',
        body: <>
            <p>24 of the 32 layers look like this. They use <b>Gated DeltaNet</b>. Instead of comparing every token to every earlier token, it reads tokens in order and keeps a small <b>memory</b>, so the cost grows linearly with the length of the prompt.</p>
            <p>The next steps go through it one piece at a time. Green blocks are values computed for this prompt, blue blocks are learned weights, and amber is the memory.</p>
        </>,
    },
    {
        id: 'norm', section: 'Linear-attention layer', title: 'RMSNorm', layer: 'linear', groups: ['norm1'],
        body: <>
            <p>The incoming vector for each token is rescaled so its typical size is 1, then multiplied dim by dim by a learned scale.</p>
            <Eq>x / rms(x) · weight</Eq>
            <p>Unlike GPT's LayerNorm, this doesn't subtract a mean, and there is no bias term anywhere in the layer.</p>
        </>,
    },
    {
        id: 'qkv', section: 'Linear-attention layer', title: 'Queries, keys and values', layer: 'linear', groups: ['qkv', 'split'],
        body: <>
            <p>One big matrix projects 2,560 dims to 8,192 numbers. A small convolution then lets each of those numbers see the three tokens before it, and SiLU adds a nonlinearity.</p>
            <p>The result is split into <b>16 query heads</b>, <b>16 key heads</b> and <b>32 value heads</b>, each 128 wide. Queries and keys are repeated so that every value head has one.</p>
        </>,
    },
    {
        id: 'gates', section: 'Linear-attention layer', title: 'How much to write, how much to forget', layer: 'linear', groups: ['gates'],
        body: <>
            <p>Each token also decides, per head, how it will treat the memory.</p>
            <Eq>β = sigmoid(in_proj_b · x)</Eq>
            <p><b>β</b> is the write strength, between 0 and 1.</p>
            <Eq>g = −exp(A_log) · softplus(in_proj_a · x + dt_bias)</Eq>
            <p><b>g</b> is always negative, so multiplying by e<sup>g</sup> makes older memories fade. A large |g| means this token wipes most of what the head remembered.</p>
        </>,
    },
    {
        id: 'memory', section: 'Linear-attention layer', title: 'The memory', layer: 'linear', groups: ['memory'], view: 'angled', tokenSlider: true,
        body: <>
            <p>Each amber slab is one head's memory S, a 128 × 128 matrix (one value in 8 per axis shown). Move the slider to watch it change token by token.</p>
            <Eq>S ← e<sup>g</sup> · S</Eq>
            <Eq>δ = β · (v − Sᵀk)</Eq>
            <Eq>S ← S + k δᵀ</Eq>
            <Eq>out = Sᵀ q</Eq>
            <p>The "delta" is the correction: the token writes only the difference between its value and what the memory already says about its key. Queries and keys are L2-normalised first, and queries are also divided by √128.</p>
            <p>GDN has no attention matrix, but the green slabs show a close relative. Because the output is exactly a sum over earlier writes, row <i>t</i>, column <i>s</i> says how strongly token <i>t</i> reads what token <i>s</i> wrote:</p>
            <Eq>read[t, s] = decay(s→t) · (q<sub>t</sub> · k<sub>s</sub>)</Eq>
            <p className="q-note">This matrix is derived by this project, not something the model computes. Unlike attention it isn't a probability and can be negative. The tracer checks that summing read[t, s] · δ<sub>s</sub> rebuilds the model's memory output.</p>
        </>,
    },
    {
        id: 'mixout', section: 'Linear-attention layer', title: 'Back to the residual stream', layer: 'linear', groups: ['mixout'],
        body: <>
            <p>The memory output is normalised, multiplied by a final gate, and projected from 4,096 dims back to 2,560.</p>
            <Eq>out / rms(out) · weight · SiLU(z)</Eq>
            <p>It is then <b>added</b> to the vector that came into the layer. Everything the layer does is a correction to the stream running down the centre.</p>
        </>,
    },
    {
        id: 'mlp', section: 'Linear-attention layer', title: 'The MLP', layer: 'linear', groups: ['mlp'],
        body: <>
            <p>Every layer ends with the same block, a <b>SwiGLU</b> MLP. Each token passes through two separate projections up to 9,216 dims, where one gates the other, then one projection back down.</p>
            <Eq>down( SiLU(gate(x)) × up(x) )</Eq>
            <p>GPT's MLP has one up-projection and a GELU. This one uses three matrices, and they hold most of the layer's parameters. This step treats tokens independently; only the mixer above moves information between them.</p>
        </>,
    },
    {
        id: 'full', section: 'Full-attention layer', title: 'A full-attention layer', layer: 'full', groups: ['norm1', 'fullqkv'],
        body: <>
            <p>Every fourth layer is closer to classic attention, with a few changes.</p>
            <ul>
                <li><b>Fewer keys than queries.</b> 16 query heads share 4 key and value heads.</li>
                <li><b>Normalised.</b> Queries and keys each go through their own RMSNorm.</li>
                <li><b>Rotated by position.</b> RoPE rotates a quarter of each query and key by the token's position.</li>
                <li><b>A gate.</b> q_proj is twice as wide as it needs to be for queries. The other half becomes an output gate.</li>
            </ul>
        </>,
    },
    {
        id: 'attn', section: 'Full-attention layer', title: 'Attention', layer: 'full', groups: ['attn', 'attnout'], fit: ['attn'], view: 'angled',
        body: <>
            <p>These are real attention matrices, one per query head. Row <i>t</i> is a probability distribution over tokens up to <i>t</i>; later tokens are masked.</p>
            <Eq>softmax(Q Kᵀ / √256)</Eq>
            <p>Each head's output is multiplied by sigmoid(gate), then o_proj maps 4,096 dims back to 2,560 and the result is added to the stream. The MLP that follows is the same design as before.</p>
        </>,
    },
    {
        id: 'lens', section: 'Output', title: 'What each layer would predict', layer: 'none', groups: ['tower'], fit: ['tower'],
        body: <>
            <p>The tags beside the tower show a <b>logit lens</b>: the residual stream at that layer, pushed through the model's final norm and output head as if the model stopped there. Each tag shows the top guess for the last token and its probability.</p>
            <p>Watch how far down the tower the final answer appears.</p>
            <p className="q-note">The lens is an interpretability tool, not something the model does. Its early-layer guesses are often noise.</p>
        </>,
    },
    {
        id: 'output', section: 'Output', title: 'The next token', layer: 'none', groups: ['output'], predictions: true,
        body: <>
            <p>A final RMSNorm, then a dot product against <b>the same table used for the input embeddings</b>. That gives one score for each of the 248,320 tokens, and softmax turns the scores into probabilities.</p>
            <p>The model makes this prediction at every position. Only the last one predicts new text.</p>
            <p className="q-note">Not shown: the 24-block vision tower (0.33B parameters), which this text prompt never uses.</p>
        </>,
    },
];
