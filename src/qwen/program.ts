import { genModelViewMatrices, ICamera, updateCamera } from "@/src/llm/Camera";
import { IFontAtlasData } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "@/src/llm/render/modelRender";
import { RenderPhase } from "@/src/llm/render/sharedRender";
import { addPrimitiveRestart, addVert } from "@/src/llm/render/triRender";
import { Mat4f } from "@/src/utils/matrix";
import { IBufferTex } from "@/src/utils/renderPhases";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IIndex, ITensorSet, ITrace } from "./data";
import { genScene, IArrow, IQBlock, IQLayout } from "./layout";
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
    layer: number;
    stepIdx: number;
    tokenCursor: number;
    layoutDirty: boolean;
    pendingFit: boolean;
    firstFit: boolean;
    hover: IHover | null;
    mouse: [number, number];
    overlay: { onFrame?: (s: IQState) => void; onLayout?: (version: number) => void };
    selectedKey: string | null;
    frontView: boolean;
}

export function initState(canvas: HTMLCanvasElement, font: IFontAtlasData, index: IIndex, trace: ITrace): IQState | null {
    let render = initRender(canvas, font);
    if (!render) return null;
    let camera: ICamera = {
        angle: new Vec3(298, 18, 5), center: new Vec3(0, 0, 0), transition: {},
        modelMtx: new Mat4f(), viewMtx: new Mat4f(), lookAtMtx: new Mat4f(), camPos: new Vec3(), camPosModel: new Vec3(),
    };
    let st: IQState = {
        render, camera, examples: [], index, trace, embedW: null, textures: new Map(), layer: DEFAULT_LINEAR,
        stepIdx: 0, tokenCursor: 0, layoutDirty: true, pendingFit: true, firstFit: true, hover: null,
        mouse: [-1, -1], overlay: {}, selectedKey: null, layout: null!, frontView: false,
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
    return genScene({
        index: st.index, trace: st.trace, embedW: st.embedW, scene: STEPS[st.stepIdx].scene, layer: st.layer,
        tokenCursor: st.tokenCursor, tex: (k, w, h, b) => makeTex(st, k, w, h, b),
    });
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

// Frame the whole scene from one consistent viewing direction, leaving room for the labels.
export function fitCamera(st: IQState, instant = false) {
    let bl = st.layout.cubes;
    if (bl.length === 0) return;
    let tower = st.layout.scene === 'tower';
    let ax = st.frontView ? 272 : tower ? 300 : 300, ay = st.frontView ? 2 : tower ? 16 : 20;
    let ar = ax * Math.PI / 180, er = ay * Math.PI / 180;
    // Camera axes in camera-space coordinates, where model (x, y, z) maps to (x, -z, -y).
    let back = new Vec3(Math.cos(er) * Math.cos(ar), Math.cos(er) * Math.sin(ar), Math.sin(er));
    let right = new Vec3(-Math.sin(ar), Math.cos(ar), 0);
    let up = new Vec3(-Math.sin(er) * Math.cos(ar), -Math.sin(er) * Math.sin(ar), Math.cos(er));
    let dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
    // Extents of every block corner along those axes; lens tags and titles widen the frame.
    let extra = tower ? 120 : 0;
    let rMin = 1e12, rMax = -1e12, uMin = 1e12, uMax = -1e12, bMin = 1e12, bMax = -1e12;
    for (let c of bl) {
        for (let i = 0; i < 8; i++) {
            let x = c.x + (i & 1 ? c.dx + (c.layer !== null ? extra : 0) : 0), y = c.y + (i & 2 ? c.dy : 0), z = c.z + (i & 4 ? c.dz : 0);
            let p = new Vec3(x, -z, -y);
            let r = dot(p, right), u = dot(p, up), k = dot(p, back);
            rMin = Math.min(rMin, r); rMax = Math.max(rMax, r);
            uMin = Math.min(uMin, u); uMax = Math.max(uMax, u);
            bMin = Math.min(bMin, k); bMax = Math.max(bMax, k);
        }
    }
    let w = rMax - rMin, h = uMax - uMin;
    let size = st.render.size;
    let aspect = size.x / Math.max(1, size.y);
    let tan = Math.tan(20 * Math.PI / 180);
    let dist = Math.max(h / (2 * tan), w / (2 * tan * aspect)) * (tower ? 1.03 : 1.28) + (bMax - bMin) / 2;
    dist = Math.max(dist, 170);
    // Look at the middle of the frame, a little higher than centre for the titles above blocks.
    let midR = (rMin + rMax) / 2, midU = (uMin + uMax) / 2 + h * (tower ? 0.01 : 0.04), midB = (bMin + bMax) / 2;
    let center = new Vec3(
        right.x * midR + up.x * midU + back.x * midB,
        right.y * midR + up.y * midU + back.y * midB,
        right.z * midR + up.z * midU + back.z * midB);
    let angle = new Vec3(ax, ay, dist / 200);
    if (instant) {
        st.camera.center = center;
        st.camera.angle = angle;
        st.camera.desiredCamera = undefined;
        st.camera.desiredCameraTransition = undefined;
    } else {
        st.camera.desiredCamera = { center, angle };
    }
}

const worldPerPx = (st: IQState) => 2 * st.camera.angle.z * 200 * Math.tan(20 * Math.PI / 180) / Math.max(1, st.render.size.y);
const inkN = new Vec3(0, 0, 1);

function drawArrow(st: IQState, ar: IArrow) {
    let r = st.render, { from: f, to: t, mode } = ar;
    let px = worldPerPx(st);
    let color = new Vec4(0.42, 0.42, 0.45, 1);
    let pad = 3 * px;
    let zf = f.z + f.dz / 2, zt = t.z + t.dz / 2;
    let pts: Vec3[];
    if (mode === 'h') {
        let ltr = f.x + f.dx <= t.x;
        let ay = f.y + f.dy / 2, by = t.y + t.dy / 2;
        let ax = ltr ? f.x + f.dx + pad : f.x - pad;
        let bx = ltr ? t.x - pad : t.x + t.dx + pad;
        pts = Math.abs(ay - by) < 0.5
            ? [new Vec3(ax, ay, zf), new Vec3(bx, by, zt)]
            : [new Vec3(ax, ay, zf), new Vec3((ax + bx) / 2, ay, zf), new Vec3((ax + bx) / 2, by, zt), new Vec3(bx, by, zt)];
    } else if (mode === 'side') {
        let ax = f.x + f.dx + pad, ay = f.y + f.dy / 2, by = t.y + t.dy / 2;
        let bx = t.x + t.dx + pad, out = Math.max(ax, bx) + 10;
        pts = [new Vec3(ax, ay, zf), new Vec3(out, ay, zf), new Vec3(out, by, zt), new Vec3(bx, by, zt)];
    } else {
        let ax = f.x + f.dx / 2, bx = t.x + t.dx / 2;
        let ay = f.y + f.dy + pad, by = t.y - pad;
        pts = Math.abs(ax - bx) < 0.5
            ? [new Vec3(ax, ay, zf), new Vec3(bx, by, zt)]
            : [new Vec3(ax, ay, zf), new Vec3(ax, (ay + by) / 2, zf), new Vec3(bx, (ay + by) / 2, zt), new Vec3(bx, by, zt)];
    }
    let thick = 0.75 * px;
    for (let i = 0; i + 1 < pts.length; i++) addLine(r.lineRender, thick, color, pts[i], pts[i + 1], inkN);
    let n = pts.length;
    let dir = pts[n - 1].sub(pts[n - 2]).normalize();
    let side = new Vec3(-dir.y, dir.x, 0), size = 3.6 * px;
    let tip = pts[n - 1], base = tip.sub(dir.mul(size * 1.6));
    addVert(r.triRender, base.add(side.mul(size)), color, inkN);
    addVert(r.triRender, tip, color, inkN);
    addVert(r.triRender, base.sub(side.mul(size)), color, inkN);
    addPrimitiveRestart(r.triRender);
}

function outline(st: IQState, b: IQBlock) {
    let r = st.render, z = b.z + b.dz + 0.05, px = worldPerPx(st), m = 1.5 * px;
    let c = new Vec4(0.1, 0.1, 0.14, 1), t = 0.8 * px;
    let tl = new Vec3(b.x - m, b.y - m, z), tr = new Vec3(b.x + b.dx + m, b.y - m, z);
    let bl = new Vec3(b.x - m, b.y + b.dy + m, z), br = new Vec3(b.x + b.dx + m, b.y + b.dy + m, z);
    addLine(r.lineRender, t, c, tl, tr, inkN); addLine(r.lineRender, t, c, tr, br, inkN);
    addLine(r.lineRender, t, c, br, bl, inkN); addLine(r.lineRender, t, c, bl, tl, inkN);
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
        let prevScene = st.layout?.scene;
        st.layout = buildLayout(st);
        st.layoutDirty = false;
        if (st.layout.scene !== prevScene) st.pendingFit = true;
        st.overlay.onLayout?.(st.layout.version);
    }
    if (st.pendingFit && render.size.x > 1) {
        fitCamera(st, st.firstFit);
        st.firstFit = false;
        st.pendingFit = false;
    }

    genModelViewMatrices(st as any, st.layout as any);
    updateCamera(st as any, view);
    hitTest(st);

    render.sharedRender.activePhase = RenderPhase.Arrows;
    for (let ar of st.layout.arrows) drawArrow(st, ar);
    render.sharedRender.activePhase = RenderPhase.Overlay;
    if (st.hover && st.hover.block.tensor) outline(st, st.hover.block);
    if (st.selectedKey && st.layout.byKey[st.selectedKey]) outline(st, st.layout.byKey[st.selectedKey]);

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
    let one = (a: { name: string; stride: number; of: number; kind?: string } | undefined, c: number) => {
        if (!a) return;
        if (a.kind === 'token') parts.push(`${a.name} ${c} ${JSON.stringify(tokens[c]?.text ?? '')}`);
        else parts.push(`${a.name} ${c * a.stride} of ${a.of}`);
    };
    one(ax.x, h.cell[0]);
    one(ax.y, h.cell[1]);
    return parts;
}
