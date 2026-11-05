// Loads the files written by tracer/trace.py. Every tensor is int8, value = byte / 127 * scale.
export interface ITensor {
    name: string;
    shape: number[];
    offset: number;
    scale: number;
    p99: number;
    mean: number;
    std: number;
    full_shape?: number[];
    stride?: number[];
    full_mean?: number;
    full_std?: number;
    full_min?: number;
    full_max?: number;
    data: Int8Array;
}

export interface IPred { id: number; text: string; p: number; }

export interface IIndex {
    model: string;
    revision: string;
    prompts: { key: string; text: string; T: number; top: string; p: number }[];
    weight_stride: number;
    head_dim_stride: number;
    layer_types: string[];
    hidden: number;
    mlp: number;
    vocab: number;
    params: { text: number; vision: number };
}

export interface IManifest {
    key: string;
    text: string;
    T: number;
    tokens: { id: number; text: string }[];
    predictions: IPred[][];
    lens: IPred[][];
    checks: Record<string, number>;
}

export type ITensorSet = Record<string, ITensor>;

export interface ITrace {
    manifest: IManifest;
    overview: ITensorSet;
    layers: Map<number, { act: ITensorSet; w: ITensorSet }>;
    pending: Set<number>;
}

const BASE = 'q35';

async function getJson(path: string) {
    let r = await fetch(`${BASE}/${path}`);
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
}

async function getTensors(jsonPath: string | Record<string, any>, binPath: string): Promise<ITensorSet> {
    let meta = typeof jsonPath === 'string' ? await getJson(jsonPath) : jsonPath;
    let r = await fetch(`${BASE}/${binPath}`);
    if (!r.ok) throw new Error(`${binPath}: ${r.status}`);
    let buf = await r.arrayBuffer();
    let out: ITensorSet = {};
    for (let name in meta) {
        let m = meta[name];
        let n = m.shape.reduce((a: number, b: number) => a * b, 1);
        out[name] = { ...m, name, data: new Int8Array(buf, m.offset, n) };
    }
    return out;
}

export const fetchIndex = (): Promise<IIndex> => getJson('index.json');

let weightsCache = new Map<number, Promise<ITensorSet>>();
let embedCache: Promise<ITensorSet> | null = null;

export function fetchEmbedWeights() {
    embedCache = embedCache ?? getTensors('weights/embed.json', 'weights/embed.bin');
    return embedCache;
}

export async function fetchTrace(key: string): Promise<ITrace> {
    let manifest = await getJson(`${key}/manifest.json`);
    let overview = await getTensors((manifest as any).overview, `${key}/overview.bin`);
    return { manifest, overview, layers: new Map(), pending: new Set() };
}

export async function fetchLayer(trace: ITrace, i: number) {
    let existing = trace.layers.get(i);
    if (existing) return existing;
    let ll = String(i).padStart(2, '0');
    if (!weightsCache.has(i)) {
        weightsCache.set(i, getTensors(`weights/layer_${ll}.json`, `weights/layer_${ll}.bin`));
    }
    let [act, w] = await Promise.all([
        getTensors(`${trace.manifest.key}/layer_${ll}.json`, `${trace.manifest.key}/layer_${ll}.bin`),
        weightsCache.get(i)!,
    ]);
    let res = { act, w };
    trace.layers.set(i, res);
    return res;
}

export function fmt(v: number) {
    let a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
    return v.toFixed(a < 1 ? 3 : 2);
}
