import { cameraToMatrixView, genModelViewMatrices, ICamera, updateCamera } from "@/src/llm/Camera";
import { IFontAtlasData, measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "@/src/llm/render/modelRender";
import { RenderPhase } from "@/src/llm/render/sharedRender";
import { addPrimitiveRestart, addVert } from "@/src/llm/render/triRender";
import { Mat4f } from "@/src/utils/matrix";
import { IBufferTex } from "@/src/utils/renderPhases";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IIndex, ITensorSet, ITrace } from "./data";
import { genLayout, IArrow, IQBlock, IQLayout } from "./layout";
import { IStep, STEPS } from "./steps";

export const DEFAULT_LINEAR = 12;
export const DEFAULT_FULL = 15;

export interface IHover { block: IQBlock; cell: [number, number, number]; value: number | null; screen: [number, number]; }

export interface IQState {
    render: IRenderState;
    camera: ICamera;
    layout: IQLayout;
    examples: [];
    index: IIndex;
    trace: ITrace;
    embedW: ITensorSet | null;
    textures: Map<string, IBufferTex>;
    expanded: number | null;
    stepIdx: number;
    tokenCursor: number;
    layoutDirty: boolean;
    pendingFit: boolean;
    firstFit: boolean;
    hover: IHover | null;
    mouse: [number, number];
    overlay: { onFrame?: (s: IQState) => void };
    selectedKey: string | null;
}

export function initState(canvas: HTMLCanvasElement, font: IFontAtlasData, index: IIndex, trace: ITrace): IQState | null {
    let render = initRender(canvas, font);
    if (!render) return null;
    let camera: ICamera = {
        angle: new Vec3(285, 10, 20), center: new Vec3(0, 0, -1000), transition: {},
        modelMtx: new Mat4f(), viewMtx: new Mat4f(), lookAtMtx: new Mat4f(), camPos: new Vec3(), camPosModel: new Vec3(),
    };
    let st: IQState = {
        render, camera, examples: [], index, trace, embedW: null, textures: new Map(), expanded: null,
        stepIdx: 0, tokenCursor: 0, layoutDirty: true, pendingFit: true, firstFit: true, hover: null,
        mouse: [-1, -1], overlay: {}, selectedKey: null, layout: null!,
    };
    st.layout = buildLayout(st);
    return st;
}

function makeTex(st: IQState, key: string, w: number, h: number, build: () => Float32Array): IBufferTex {
    let t = st.textures.get(key);
    if (t) return t;
    let gl = st.render.gl;
    let data = build();
    let texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    t = { width: w, height: h, channels: 1, texture, localBuffer: data };
    st.textures.set(key, t);
    return t;
}

export function buildLayout(st: IQState) {
    return genLayout(st.index, st.trace, st.embedW, st.expanded, (k, w, h, b) => makeTex(st, k, w, h, b), st.tokenCursor);
}

// Drop textures that belong to a prompt; weights are shared between prompts.
export function setTrace(st: IQState, trace: ITrace) {
    let gl = st.render.gl;
    for (let [k, t] of st.textures) {
        if (!k.startsWith('w/')) {
            gl.deleteTexture(t.texture);
            st.textures.delete(k);
        }
    }
    st.trace = trace;
    st.tokenCursor = Math.min(st.tokenCursor, trace.manifest.T - 1);
    st.selectedKey = null;
    st.layoutDirty = true;
    st.pendingFit = true;
}

export function currentStep(st: IQState): IStep { return STEPS[st.stepIdx]; }

function isFocused(step: IStep, b: IQBlock, expanded: number | null) {
    if (step.groups.length === 0) return true;
    if (b.group === 'tower') return step.groups.includes('tower');
    if (b.layer !== null && b.layer !== expanded) return false;
    return step.groups.includes(b.group);
}

// Frame the blocks of the current step from a fixed viewing direction.
export function fitCamera(st: IQState, instant = false) {
    let step = currentStep(st);
    let groups = step.fit ?? step.groups;
    let bl = st.layout.cubes.filter(c => groups.length === 0 ? true : groups.includes(c.group));
    if (bl.length === 0) bl = st.layout.cubes;
    let min = new Vec3(1e12, 1e12, 1e12), max = new Vec3(-1e12, -1e12, -1e12);
    for (let c of bl) {
        min.x = Math.min(min.x, c.x); min.y = Math.min(min.y, c.y); min.z = Math.min(min.z, c.z);
        max.x = Math.max(max.x, c.x + c.dx); max.y = Math.max(max.y, c.y + c.dy); max.z = Math.max(max.z, c.z + c.dz);
    }
    // Names start at a block's left edge and can run past its right edge.
    for (let c of bl) if (c.label) max.x = Math.max(max.x, c.x + c.title.length * 2.6);
    let w = max.x - min.x, h = max.y - min.y, d = max.z - min.z;
    let size = st.render.size;
    let aspect = size.x / Math.max(1, size.y);
    let tan = Math.tan(20 * Math.PI / 180);
    let angled = step.view === 'angled';
    let dist = Math.max((h + (angled ? d * 0.25 : 0)) / (2 * tan), (w + (angled ? d * 0.5 : 0)) / (2 * tan * aspect)) * 1.25 + d / 2;
    dist = Math.max(dist, 160);
    // Model (x, y, z) -> camera space (x, -z, -y)
    let center = new Vec3((min.x + max.x) / 2, -(min.z + max.z) / 2, -(min.y + max.y) / 2);
    let angle = new Vec3(angled ? 296 : 276, angled ? 16 : 6, dist / 200);
    if (instant) {
        st.camera.center = center;
        st.camera.angle = angle;
        st.camera.desiredCamera = undefined;
        st.camera.desiredCameraTransition = undefined;
    } else {
        st.camera.desiredCamera = { center, angle };
    }
}

const inkHead = new Vec3(0, 0, 1);
function arrowLine(r: IRenderState, a: Vec3, b: Vec3, color: Vec4) {
    addLine(r.lineRender, 1.2, color, a, b, new Vec3(0, 0, 1));
}
function arrowHead(r: IRenderState, tip: Vec3, dir: Vec3, color: Vec4, size = 2.6) {
    let side = new Vec3(-dir.y, dir.x, 0);
    let base = tip.sub(dir.mul(size * 1.5));
    addVert(r.triRender, base.add(side.mul(size)), color, inkHead);
    addVert(r.triRender, tip, color, inkHead);
    addVert(r.triRender, base.sub(side.mul(size)), color, inkHead);
    addPrimitiveRestart(r.triRender);
}

function drawArrow(r: IRenderState, ar: IArrow, step: IStep, st: IQState) {
    let { from: f, to: t, mode } = ar;
    let o = Math.min(f.opacity, t.opacity);
    if (o <= 0.05) return;
    let isW = f.t === 'w';
    let color = (isW ? new Vec4(0.2, 0.2, 0.67, 1) : new Vec4(0.2, 0.6, 0.2, 1)).mul(0.75 * o);
    let zm = (b: IQBlock) => b.z + b.dz / 2;
    let pad = 1.5;
    let fz = zm(f), tz = zm(t);
    let pts: Vec3[];
    if (mode === 'h') {
        let ltr = f.x + f.dx <= t.x;
        let ay = f.y + f.dy / 2, by = t.y + t.dy / 2;
        let y = Math.abs(ay - by) < Math.min(f.dy, t.dy) / 2 ? (ay + by) / 2 : null;
        let ax = ltr ? f.x + f.dx + pad : f.x - pad;
        let bx = ltr ? t.x - pad : t.x + t.dx + pad;
        pts = y !== null
            ? [new Vec3(ax, y, fz), new Vec3(bx, y, tz)]
            : [new Vec3(ax, ay, fz), new Vec3((ax + bx) / 2, ay, fz), new Vec3((ax + bx) / 2, by, tz), new Vec3(bx, by, tz)];
    } else if (mode === 'v') {
        let ax = f.x + f.dx / 2, bx = t.x + t.dx / 2;
        let ay = f.y + f.dy + pad, by = t.y - pad;
        pts = Math.abs(ax - bx) < 0.01
            ? [new Vec3(ax, ay, fz), new Vec3(bx, by, tz)]
            : [new Vec3(ax, ay, fz), new Vec3(ax, (ay + by) / 2, fz), new Vec3(bx, (ay + by) / 2, tz), new Vec3(bx, by, tz)];
    } else if (mode === 'rt') {
        // Residual stream, branching left into a side block.
        let ax = f.x - pad, ay = f.y + f.dy - 2;
        let bx = t.x + t.dx + pad, by = t.y + t.dy / 2;
        let midX = (ax + bx) / 2;
        pts = [new Vec3(ax, ay, fz), new Vec3(midX, ay, fz), new Vec3(midX, by, tz), new Vec3(bx, by, tz)];
    } else {
        // From the bottom of `from`, across, then into the top of `to`.
        let ax = f.x + f.dx / 2, ay = f.y + f.dy + pad;
        let bx = t.x + t.dx / 2, by = t.y - pad;
        let my = ay + (by - ay) * 0.5;
        pts = [new Vec3(ax, ay, fz), new Vec3(ax, my, fz), new Vec3(bx, my, tz), new Vec3(bx, by, tz)];
    }
    for (let i = 0; i + 1 < pts.length; i++) arrowLine(r, pts[i], pts[i + 1], color);
    let n = pts.length;
    let dir = pts[n - 1].sub(pts[n - 2]).normalize();
    arrowHead(r, pts[n - 1], dir, color);
}

function rayBox(tl: Vec3, br: Vec3, o: Vec3, d: Vec3) {
    let tmin = -Infinity, tmax = Infinity;
    for (let a of ['x', 'y', 'z'] as const) {
        if (Math.abs(d[a]) < 1e-9) {
            if (o[a] < tl[a] || o[a] > br[a]) return -1;
            continue;
        }
        let t1 = (tl[a] - o[a]) / d[a], t2 = (br[a] - o[a]) / d[a];
        tmin = Math.max(tmin, Math.min(t1, t2));
        tmax = Math.min(tmax, Math.max(t1, t2));
    }
    return tmax >= Math.max(tmin, 0) ? tmin : -1;
}

export function hitTest(st: IQState) {
    let [mx, my] = st.mouse;
    let size = st.render.size;
    st.hover = null;
    if (mx < 0) return;
    let viewInv = st.camera.viewMtx.invert(), modelInv = st.camera.modelMtx.invert();
    let unproject = (z: number) => modelInv.mulVec4(viewInv.mulVec4(new Vec4(mx / size.x * 2 - 1, 1 - my / size.y * 2, z, 1))).projToVec3();
    let a = unproject(-1), b = unproject(1);
    let dir = b.sub(a).normalize();
    let best: IQBlock | null = null, bestT = 0;
    for (let c of st.layout.cubes) {
        if (c.opacity < 0.5) continue;
        let t = rayBox(new Vec3(c.x, c.y, c.z), new Vec3(c.x + c.dx, c.y + c.dy, c.z + c.dz), a, dir);
        if (t > 0 && (!best || t < bestT)) { best = c; bestT = t; }
    }
    if (!best) return;
    let p = a.add(dir.mul(bestT + 0.01));
    let cell: [number, number, number] = [
        Math.min(best.cx - 1, Math.max(0, Math.floor((p.x - best.x) / best.dx * best.cx))),
        Math.min(best.cy - 1, Math.max(0, Math.floor((p.y - best.y) / best.dy * best.cy))),
        Math.min(best.cz - 1, Math.max(0, Math.floor((p.z - best.z) / best.dz * best.cz))),
    ];
    let value: number | null = null;
    if (best.access?.src.localBuffer) {
        let m = best.access.mat, src = best.access.src;
        let tx = m[0] * cell[0] + m[1] * cell[1] + m[2] * cell[2] + m[3];
        let ty = m[4] * cell[0] + m[5] * cell[1] + m[6] * cell[2] + m[7];
        value = src.localBuffer![ty * src.width + tx];
    }
    st.hover = { block: best, cell, value, screen: [mx, my] };
}

export function project(st: IQState, p: Vec3): [number, number, boolean] {
    let clip = st.camera.viewMtx.mulVec4(st.camera.modelMtx.mulVec4(new Vec4(p.x, p.y, p.z, 1)));
    if (clip.w <= 0) return [0, 0, false];
    let s = st.render.size;
    return [(clip.x / clip.w + 1) / 2 * s.x, (1 - clip.y / clip.w) / 2 * s.y, true];
}

export function runFrame(view: IRenderView, st: IQState) {
    let render = st.render;
    resetRenderBuffers(render);
    render.sharedRender.activePhase = RenderPhase.Opaque;

    if (st.layoutDirty) {
        st.layout = buildLayout(st);
        st.layoutDirty = false;
    }
    let step = currentStep(st);
    if (st.pendingFit && render.size.x > 1) {
        fitCamera(st, st.firstFit);
        st.firstFit = false;
        st.pendingFit = false;
    }

    // Dim everything the current step isn't about.
    for (let c of st.layout.cubes) {
        c.opacity = isFocused(step, c, st.expanded) ? 1 : 0.14;
        c.highlight = 0;
    }

    genModelViewMatrices(st as any, st.layout);
    updateCamera(st as any, view);
    hitTest(st);
    if (st.hover) st.hover.block.highlight = 0.35;
    if (st.selectedKey && st.layout.byKey[st.selectedKey]) st.layout.byKey[st.selectedKey].highlight = 0.5;

    render.sharedRender.activePhase = RenderPhase.Arrows;
    for (let ar of st.layout.arrows) drawArrow(render, ar, step, st);

    // Block names. ASCII only: the font atlas has no CJK; tokens are HTML labels instead.
    // Drawn after the arrows so an arrow never covers a name.
    render.sharedRender.activePhase = RenderPhase.Overlay;
    let fontSize = Math.max(3.2, Math.min(st.camera.angle.z * 0.55, 16));
    let rows = new Map<number, IQBlock[]>();
    for (let c of st.layout.cubes) {
        if (!c.label || c.opacity < 0.5 || c.group === 'tower') continue;
        let k = Math.round(c.y);
        if (!rows.has(k)) rows.set(k, []);
        rows.get(k)!.push(c);
    }
    for (let row of rows.values()) {
        // Narrow neighbours would collide, so step their labels upwards from right to left.
        row.sort((a, b) => b.x - a.x);
        let lane = 0, prevLeft = Infinity;
        for (let c of row) {
            let w = measureTextWidth(render.modelFontBuf, c.title, fontSize);
            lane = c.x + w + 2 > prevLeft ? lane + 1 : 0;
            prevLeft = lane === 0 ? c.x : Math.min(prevLeft, c.x);
            let mtx = new Mat4f();
            mtx[14] = c.z + c.dz;
            writeTextToBuffer(render.modelFontBuf, c.title, new Vec4(0.15, 0.15, 0.2, 1), c.x, c.y - (lane + 1) * (fontSize * 1.15) - 1, fontSize, mtx);
        }
    }

    render.sharedRender.activePhase = RenderPhase.Opaque;
    renderModel(st as any);
    render.gl.flush();
    st.overlay.onFrame?.(st);
}

export function axisText(st: IQState, h: IHover) {
    let parts: string[] = [];
    let ax = h.block.axes;
    if (!ax) return parts;
    let tokens = st.trace.manifest.tokens;
    let one = (a: IQBlock['axes'] extends infer _ ? any : never, c: number) => {
        if (!a) return;
        if (a.kind === 'token') {
            parts.push(`${a.name} ${c} ${JSON.stringify(tokens[c]?.text ?? '')}`);
        } else {
            parts.push(`${a.name} ${c * a.stride} of ${a.of}`);
        }
    };
    one(ax.x, h.cell[0]);
    one(ax.y, h.cell[1]);
    one(ax.z, h.cell[2]);
    return parts;
}

export { cameraToMatrixView };
