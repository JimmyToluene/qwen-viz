'use client';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { fetchFontAtlasData, IFontAtlasData } from '@/src/llm/render/fontRender';
import { Vec3 } from '@/src/utils/vector';
import { fetchEmbedWeights, fetchIndex, fetchLayer, fetchTrace, fmt, IIndex, ITensor, ITrace } from './data';
import { IQBlock } from './layout';
import { axisText, buildLayout, currentStep, DEFAULT_FULL, DEFAULT_LINEAR, fitCamera, hitTest, initState, IQState, project, runFrame, setTrace } from './program';
import { STEPS } from './steps';
import './qwen.scss';

const SHORT_LAYER = (t: string) => t === 'full_attention' ? 'full' : 'linear';

export function View() {
    let [index, setIndex] = useState<IIndex | null>(null);
    let [font, setFont] = useState<IFontAtlasData | null>(null);
    let [promptKey, setPromptKey] = useState('story');
    let [trace, setTraceS] = useState<ITrace | null>(null);
    let [error, setError] = useState<string | null>(null);

    useEffect(() => {
        Promise.all([fetchIndex(), fetchFontAtlasData()]).then(([i, f]) => { setIndex(i); setFont(f); })
            .catch(e => setError(String(e)));
    }, []);
    useEffect(() => {
        let stale = false;
        fetchTrace(promptKey).then(t => { if (!stale) setTraceS(t); }).catch(e => setError(String(e)));
        return () => { stale = true; };
    }, [promptKey]);

    if (error) return <div className="q-msg">Could not load the trace: {error}</div>;
    if (!index || !font || !trace) return <div className="q-msg">Loading Qwen3.5-4B trace…</div>;
    return <Scene index={index} font={font} trace={trace} promptKey={promptKey} setPromptKey={setPromptKey} />;
}

const Scene: React.FC<{ index: IIndex; font: IFontAtlasData; trace: ITrace; promptKey: string; setPromptKey: (k: string) => void }> = ({ index, font, trace, promptKey, setPromptKey }) => {
    let canvasRef = useRef<HTMLCanvasElement>(null);
    let overlayRef = useRef<HTMLDivElement>(null);
    let tipRef = useRef<HTMLDivElement>(null);
    let stRef = useRef<IQState | null>(null);
    let [glError, setGlError] = useState(false);
    let [stepIdx, setStepIdx] = useState(0);
    let [tokenCursor, setTokenCursor] = useState(0);
    let [playing, setPlaying] = useState(false);
    let [expanded, setExpanded] = useState<number | null>(null);
    let [loadingLayer, setLoadingLayer] = useState(false);
    let [selected, setSelected] = useState<IQBlock | null>(null);
    let dirtyRef = useRef<() => void>(() => { });
    let step = STEPS[stepIdx];
    let T = trace.manifest.T;

    // ---- Render loop: only draws when something changed ----
    useEffect(() => {
        let canvas = canvasRef.current!;
        let st = initState(canvas, font, index, trace);
        if (!st) { setGlError(true); return; }
        stRef.current = st;
        (window as any).__qwen = st;
        let raf = 0, dirty = true, prev = performance.now(), stopped = false, sizeDirty = true;
        let loop = (time: number) => {
            raf = 0;
            if (stopped || !dirty) return;
            dirty = false;
            if (sizeDirty) {
                let bcr = canvas.getBoundingClientRect();
                let dpr = window.devicePixelRatio;
                canvas.width = bcr.width * dpr;
                canvas.height = bcr.height * dpr;
                st!.render.size = new Vec3(bcr.width, bcr.height);
                sizeDirty = false;
            }
            let dt = Math.max(8, time - prev);
            prev = time;
            runFrame({ time, dt, markDirty }, st!);
        };
        function markDirty() {
            dirty = true;
            if (!raf) raf = requestAnimationFrame(loop);
        }
        dirtyRef.current = markDirty;
        let ro = new ResizeObserver(() => { sizeDirty = true; markDirty(); });
        ro.observe(canvas);
        let onWheel = (ev: WheelEvent) => ev.preventDefault();
        canvas.addEventListener('wheel', onWheel, { passive: false });
        fetchEmbedWeights().then(w => { st!.embedW = w; st!.layoutDirty = true; markDirty(); });
        markDirty();
        return () => { stopped = true; ro.disconnect(); canvas.removeEventListener('wheel', onWheel); };
        // The canvas lives for the whole page; prompt changes go through setTrace.
        // eslint-disable-next-line
    }, []);

    // ---- HTML labels that follow the 3D scene (they can show any Unicode token) ----
    useEffect(() => {
        let st = stRef.current;
        if (!st) return;
        st.overlay.onFrame = (s) => {
            let box = overlayRef.current;
            if (!box) return;
            let step = currentStep(s);
            let els = box.querySelectorAll<HTMLElement>('[data-a]');
            els.forEach(el => {
                let kind = el.dataset.a, i = +el.dataset.i!;
                let pos: Vec3 | null = null;
                if (kind === 'tok') {
                    let a = s.layout.tokenAnchor[i];
                    if (a) pos = new Vec3(a.x, a.y, a.z);
                } else {
                    let tag = s.layout.tags.find(t => t.layer === i);
                    if (tag) pos = new Vec3(tag.x, tag.y, tag.z);
                }
                if (!pos) { el.style.display = 'none'; return; }
                let [x, y, ok] = project(s, pos);
                let sz = s.render.size;
                let vis = ok && x > -50 && x < sz.x + 50 && y > -20 && y < sz.y + 20;
                if (kind === 'tok') {
                    // Only label tokens when there is room to read them.
                    let a0 = s.layout.tokenAnchor[0], a1 = s.layout.tokenAnchor[1];
                    if (a1) {
                        let [x0] = project(s, new Vec3(a0.x, a0.y, a0.z));
                        let [x1] = project(s, new Vec3(a1.x, a1.y, a1.z));
                        vis = vis && Math.abs(x1 - x0) > 7;
                    }
                }
                if (kind === 'tag') {
                    // tags only make sense when they don't overlap; check the spacing on screen
                    let tag2 = s.layout.tags.find(t => t.layer === (i === 0 ? 1 : i - 1));
                    if (tag2) {
                        let [, y2] = project(s, new Vec3(tag2.x, tag2.y, tag2.z));
                        let spacing = Math.abs(y - y2);
                        vis = vis && (spacing > 14 || i % 4 === 3 && spacing > 3.4);
                    }
                }
                el.style.display = vis ? 'block' : 'none';
                if (vis) el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
            });
            let h = s.hover, tip = tipRef.current;
            if (tip) {
                if (h) {
                    tip.style.display = 'block';
                    tip.style.transform = `translate(${h.screen[0] + 14}px, ${h.screen[1] + 14}px)`;
                    let lines = axisText(s, h);
                    let scale = h.block.tensor;
                    tip.innerHTML = '';
                    let add = (t: string, c?: string) => { let d = document.createElement('div'); d.textContent = t; if (c) d.className = c; tip!.appendChild(d); };
                    add(h.block.title, 'q-tip-title');
                    lines.forEach(l => add(l));
                    if (h.value !== null) add(`value ≈ ${fmt(h.value)}`, 'q-tip-val');
                    if (scale) add('click for detail', 'q-tip-hint');
                } else {
                    tip.style.display = 'none';
                }
            }
        };
        dirtyRef.current();
    }, [T]);

    // ---- Prompt change ----
    let firstTrace = useRef(true);
    useEffect(() => {
        if (firstTrace.current) { firstTrace.current = false; return; }
        let st = stRef.current;
        if (!st) return;
        setTrace(st, trace);
        setSelected(null);
        setTokenCursor(c => Math.min(c, trace.manifest.T - 1));
        if (st.expanded !== null) {
            let e = st.expanded;
            fetchLayer(trace, e).then(() => { st!.layoutDirty = true; st!.pendingFit = true; dirtyRef.current(); });
        }
        dirtyRef.current();
    }, [trace]);

    // ---- Expand a layer once its files have arrived ----
    let expand = useCallback(async (layer: number | null, refit: boolean) => {
        let st = stRef.current;
        if (!st) return;
        if (layer === null) {
            st.expanded = null;
            setExpanded(null);
        } else {
            setLoadingLayer(true);
            await fetchLayer(st.trace, layer);
            setLoadingLayer(false);
            st.expanded = layer;
            setExpanded(layer);
        }
        st.layoutDirty = true;
        if (refit) {
            st.layout = buildLayout(st);
            st.layoutDirty = false;
            fitCamera(st);
        }
        dirtyRef.current();
    }, []);

    // Open a layer from the scene. A second click closes it. If the current step is about the
    // other kind of layer, move to the matching section so the layer isn't dimmed.
    let browse = useCallback(async (layer: number) => {
        let st = stRef.current;
        if (!st) return;
        if (st.expanded === layer) {
            await expand(null, false);
            return;
        }
        let want = st.index.layer_types[layer] === 'full_attention' ? 'full' : 'linear';
        let cur = STEPS[st.stepIdx].layer;
        if (cur !== 'none' && cur !== want) {
            let i = STEPS.findIndex(s => s.layer === want);
            st.stepIdx = i;
            setStepIdx(i);
        }
        await expand(layer, false);
        let s = stRef.current!;
        s.layout = buildLayout(s);
        s.layoutDirty = false;
        frameLayer(s, layer);
    }, [expand]);

    let stepTo = useCallback(async (i: number) => {
        let st = stRef.current;
        i = Math.max(0, Math.min(STEPS.length - 1, i));
        setStepIdx(i);
        setPlaying(false);
        if (!st) return;
        st.stepIdx = i;
        let s = STEPS[i];
        let want: number | null = st.expanded;
        let types = st.index.layer_types;
        if (s.layer === 'none') want = null;
        else if (s.layer === 'linear' && (want === null || types[want] !== 'linear_attention')) want = DEFAULT_LINEAR;
        else if (s.layer === 'full' && (want === null || types[want] !== 'full_attention')) want = DEFAULT_FULL;
        await expand(want, true);
    }, [expand]);

    // ---- Keyboard ----
    useEffect(() => {
        let onKey = (e: KeyboardEvent) => {
            if ((e.target as HTMLElement)?.tagName === 'SELECT' || (e.target as HTMLElement)?.tagName === 'INPUT') return;
            if (e.key === 'ArrowRight' || e.key === 'PageDown') { stepTo(stepIdx + 1); e.preventDefault(); }
            if (e.key === 'ArrowLeft' || e.key === 'PageUp') { stepTo(stepIdx - 1); e.preventDefault(); }
            if (e.key === 'Escape') setSelected(null);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [stepIdx, stepTo]);

    // ---- Token cursor ----
    useEffect(() => {
        let st = stRef.current;
        if (!st) return;
        st.tokenCursor = tokenCursor;
        st.layoutDirty = true;
        dirtyRef.current();
    }, [tokenCursor]);
    useEffect(() => {
        if (!playing) return;
        let id = setInterval(() => {
            setTokenCursor(c => {
                if (c + 1 >= T) { setPlaying(false); return c; }
                return c + 1;
            });
        }, 700);
        return () => clearInterval(id);
    }, [playing, T]);

    // ---- Mouse ----
    let dragRef = useRef<{ x: number; y: number; moved: boolean; rotate: boolean; center: Vec3; angle: Vec3 } | null>(null);
    function onDown(e: React.MouseEvent) {
        let st = stRef.current;
        if (!st) return;
        dragRef.current = { x: e.clientX, y: e.clientY, moved: false, rotate: e.shiftKey || e.button !== 0, center: st.camera.center.clone(), angle: st.camera.angle.clone() };
    }
    useEffect(() => {
        let onMove = (e: MouseEvent) => {
            let st = stRef.current, d = dragRef.current;
            if (!st) return;
            let cv = canvasRef.current!.getBoundingClientRect();
            let inside = e.clientX >= cv.left && e.clientX <= cv.right && e.clientY >= cv.top && e.clientY <= cv.bottom && (e.target === canvasRef.current);
            st.mouse = inside ? [e.clientX - cv.left, e.clientY - cv.top] : [-1, -1];
            if (d) {
                let dx = e.clientX - d.x, dy = e.clientY - d.y;
                if (dx * dx + dy * dy > 16) d.moved = true;
                if (d.moved) {
                    st.camera.desiredCameraTransition = undefined;
                    if (d.rotate) {
                        st.camera.angle = new Vec3(d.angle.x - dx * 0.5, Math.max(-87, Math.min(87, d.angle.y + dy * 0.5)), d.angle.z);
                    } else {
                        let c = d.center.clone();
                        let k = d.angle.z * 0.2;
                        let sideMul = Math.sin(d.angle.x * Math.PI / 180) > 0 ? 1 : -1;
                        c.z += dy * k;
                        c.x += sideMul * dx * k;
                        st.camera.center = c;
                    }
                }
            }
            dirtyRef.current();
        };
        let onUp = () => {
            let st = stRef.current, d = dragRef.current;
            dragRef.current = null;
            if (st && d && !d.moved && st.hover) {
                let b = st.hover.block;
                if (b.group === 'tower' && b.layer !== null) {
                    browse(b.layer);
                } else {
                    setSelected(b.tensor ? b : null);
                    st.selectedKey = b.tensor ? b.key : null;
                }
            }
            dirtyRef.current();
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    }, [browse]);

    // ---- Touch: one finger pans, two fingers pinch-zoom and rotate, a tap selects ----
    useEffect(() => {
        let cv = canvasRef.current;
        if (!cv) return;
        let start: { pts: { x: number; y: number }[]; center: Vec3; angle: Vec3; moved: boolean } | null = null;
        let read = (e: TouchEvent) => Array.from(e.touches).map(t => ({ x: t.clientX, y: t.clientY }));
        let spread = (p: { x: number; y: number }[]) => Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
        let onStart = (e: TouchEvent) => {
            let st = stRef.current;
            if (!st) return;
            e.preventDefault();
            start = { pts: read(e), center: st.camera.center.clone(), angle: st.camera.angle.clone(), moved: false };
            let r = cv!.getBoundingClientRect(), t = e.touches[0];
            st.mouse = [t.clientX - r.left, t.clientY - r.top];
            dirtyRef.current();
        };
        let onMove = (e: TouchEvent) => {
            let st = stRef.current;
            if (!st || !start) return;
            e.preventDefault();
            st.camera.desiredCameraTransition = undefined;
            let p = read(e);
            if (p.length === 1 && start.pts.length === 1) {
                let dx = p[0].x - start.pts[0].x, dy = p[0].y - start.pts[0].y;
                if (dx * dx + dy * dy > 36) start.moved = true;
                if (start.moved) {
                    let c = start.center.clone(), k = start.angle.z * 0.2;
                    let side = Math.sin(start.angle.x * Math.PI / 180) > 0 ? 1 : -1;
                    c.z += dy * k;
                    c.x += side * dx * k;
                    st.camera.center = c;
                }
            } else if (p.length >= 2 && start.pts.length >= 2) {
                start.moved = true;
                let midDx = (p[0].x + p[1].x - start.pts[0].x - start.pts[1].x) / 2;
                let midDy = (p[0].y + p[1].y - start.pts[0].y - start.pts[1].y) / 2;
                st.camera.angle = new Vec3(
                    start.angle.x - midDx * 0.5,
                    Math.max(-87, Math.min(87, start.angle.y + midDy * 0.5)),
                    Math.max(0.6, Math.min(5000, start.angle.z * spread(start.pts) / Math.max(1, spread(p)))));
            }
            st.mouse = [-1, -1];
            dirtyRef.current();
        };
        let onEnd = (e: TouchEvent) => {
            let st = stRef.current;
            if (!st || !start || e.touches.length > 0) return;
            // A quick tap can end before the next frame has run the hit test.
            if (!start.moved) hitTest(st);
            if (!start.moved && st.hover) {
                let b = st.hover.block;
                if (b.group === 'tower' && b.layer !== null) {
                    browse(b.layer);
                } else {
                    setSelected(b.tensor ? b : null);
                    st.selectedKey = b.tensor ? b.key : null;
                }
            }
            start = null;
            st.mouse = [-1, -1];
            dirtyRef.current();
        };
        cv.addEventListener('touchstart', onStart, { passive: false });
        cv.addEventListener('touchmove', onMove, { passive: false });
        cv.addEventListener('touchend', onEnd);
        return () => {
            cv!.removeEventListener('touchstart', onStart);
            cv!.removeEventListener('touchmove', onMove);
            cv!.removeEventListener('touchend', onEnd);
        };
    }, [browse]);

    function onWheel(e: React.WheelEvent) {
        let st = stRef.current;
        if (!st) return;
        st.camera.desiredCameraTransition = undefined;
        let a = st.camera.angle;
        let z = Math.max(0.6, Math.min(5000, a.z * Math.pow(1.0013, e.deltaY)));
        st.camera.angle = new Vec3(a.x, a.y, z);
        dirtyRef.current();
    }

    function frameLayer(s: IQState, layer: number) {
        let bl = s.layout.cubes.filter(c => c.layer === layer);
        if (bl.length === 0) return;
        let minX = Math.min(...bl.map(c => c.x)), maxX = Math.max(...bl.map(c => c.x + c.dx));
        let minY = Math.min(...bl.map(c => c.y)), maxY = Math.max(...bl.map(c => c.y + c.dy));
        let tan = Math.tan(20 * Math.PI / 180), asp = s.render.size.x / s.render.size.y;
        let dist = Math.max((maxY - minY) / (2 * tan), (maxX - minX) / (2 * tan * asp)) * 1.2;
        s.camera.desiredCamera = { center: new Vec3((minX + maxX) / 2, 0, -(minY + maxY) / 2), angle: new Vec3(290, 12, dist / 200) };
        dirtyRef.current();
    }

    let last = trace.manifest.predictions[T - 1];
    let sections = STEPS.map(s => s.section).filter((s, i, a) => a.indexOf(s) === i);
    let currentType = expanded !== null ? index.layer_types[expanded] : null;

    return <div className="q-root">
        <aside className="q-side">
            <header className="q-head">
                <h1>Qwen3.5-4B, in 3D</h1>
                <div className="q-sub">A real forward pass. {((index.params.text + index.params.vision) / 1e9).toFixed(2)}B parameters, 32 layers.</div>
                <label className="q-prompt">
                    <span>Prompt</span>
                    <select aria-label="Prompt" value={promptKey} onChange={e => setPromptKey(e.target.value)}>
                        {index.prompts.map(p => <option key={p.key} value={p.key}>{p.text.replace('\n', ' ⏎ ')}</option>)}
                    </select>
                </label>
                <div className="q-answer">Predicts <b>{JSON.stringify(last[0].text)}</b> at {(last[0].p * 100).toFixed(1)}%</div>
            </header>

            <nav className="q-toc" aria-label="Walkthrough sections">
                {sections.map(sec => <div key={sec} className="q-toc-sec">
                    <div className="q-toc-name">{sec}</div>
                    <div className="q-toc-dots">
                        {STEPS.map((s, i) => s.section === sec &&
                            <button key={s.id} aria-label={s.title} title={s.title} className={i === stepIdx ? 'on' : ''} onClick={() => stepTo(i)} />)}
                    </div>
                </div>)}
            </nav>

            <article className="q-body" key={step.id}>
                <div className="q-eyebrow">{step.section}{step.layer !== 'none' && expanded !== null ? ` · layer ${expanded}` : ''}</div>
                <h2>{step.title}</h2>
                {step.tokenSlider && currentType === 'linear_attention' && <div className="q-slider">
                    <div className="q-slider-top">
                        <button onClick={() => { if (tokenCursor >= T - 1) setTokenCursor(0); setPlaying(p => !p); }}>{playing ? 'Pause' : 'Play'}</button>
                        <span>after token {tokenCursor}: <b>{JSON.stringify(trace.manifest.tokens[tokenCursor]?.text)}</b></span>
                    </div>
                    <input type="range" aria-label="Token" min={0} max={T - 1} value={tokenCursor} onChange={e => { setPlaying(false); setTokenCursor(+e.target.value); }} />
                </div>}
                {step.body}
                {step.predictions && <ol className="q-preds">
                    {last.map(p => <li key={p.id}>
                        <span className="q-pred-t">{JSON.stringify(p.text)}</span>
                        <span className="q-pred-bar"><span style={{ width: `${Math.max(0.5, p.p * 100)}%` }} /></span>
                        <span className="q-pred-p">{(p.p * 100).toFixed(1)}%</span>
                    </li>)}
                </ol>}
            </article>

            <footer className="q-nav">
                <button disabled={stepIdx === 0} onClick={() => stepTo(stepIdx - 1)}>Back</button>
                <span>{stepIdx + 1} / {STEPS.length}</span>
                <button className="primary" disabled={stepIdx === STEPS.length - 1} onClick={() => stepTo(stepIdx + 1)}>Next</button>
            </footer>
            <div className="q-credit">Built on <a href="https://github.com/bbycroft/llm-viz" target="_blank" rel="noreferrer">llm-viz</a> by Brendan Bycroft (MIT).</div>
        </aside>

        <main className="q-main">
            <canvas ref={canvasRef} className="q-canvas" onMouseDown={onDown} onWheel={onWheel} onContextMenu={e => e.preventDefault()} />
            {glError && <div className="q-msg">This page needs a browser with WebGL2.</div>}
            <div ref={overlayRef} className="q-overlay">
                {trace.manifest.tokens.map((t, i) => <div key={`t${i}`} data-a="tok" data-i={i} className="q-tok"><span>{t.text.replace(/\n/g, '⏎') || ' '}</span></div>)}
                {index.layer_types.map((t, i) => {
                    let lens = trace.manifest.lens[i][0];
                    return <div key={`l${i}`} data-a="tag" data-i={i} className={`q-tag q-tag-${SHORT_LAYER(t)}`}>
                        <button aria-label={`Open layer ${i}`} onClick={() => { browse(i); }}>L{i}</button>
                        <span className="q-tag-lens">{JSON.stringify(lens.text)} {(lens.p * 100).toFixed(0)}%</span>
                    </div>;
                })}
            </div>
            <div ref={tipRef} className="q-tip" style={{ display: 'none' }} />
            <div className="q-legend">
                <span><i className="k w" /> weights</span>
                <span><i className="k i" /> linear layer / computed value</span>
                <span><i className="k a" /> full-attention / memory</span>
                {loadingLayer && <span>loading layer…</span>}
            </div>
            {selected && selected.tensor && <Inspector block={selected} tokens={trace.manifest.tokens.map(t => t.text)} onClose={() => { setSelected(null); if (stRef.current) stRef.current.selectedKey = null; dirtyRef.current(); }} />}
        </main>
    </div>;
};

const Inspector: React.FC<{ block: IQBlock; tokens: string[]; onClose: () => void }> = ({ block, tokens, onClose }) => {
    let ref = useRef<HTMLCanvasElement>(null);
    let t = block.tensor!;
    let sl = block.slice;
    let rows = sl ? sl.rows : t.shape[0];
    let cols = sl ? sl.cols : t.data.length / t.shape[0];
    let offset = sl ? sl.offset : 0;
    let sampled = !!t.stride && t.stride.some(s => s > 1);
    let labelled = !!sl?.tokenAxes;
    useEffect(() => {
        let cv = ref.current;
        if (!cv) return;
        let pos = block.t === 'w' ? [70, 70, 235] : block.t === 'a' ? [235, 188, 70] : [94, 188, 94];
        let s = t.scale / 127 / (t.p99 || 1);
        let color = (i: number) => {
            // Same colour scale as the 3D view.
            let v = Math.max(-1, Math.min(1, t.data[offset + i] * s));
            let a = Math.pow(Math.abs(v), 0.7), tgt = v >= 0 ? pos : [26, 28, 43];
            return [237 + (tgt[0] - 237) * a, 237 + (tgt[1] - 237) * a, 232 + (tgt[2] - 232) * a];
        };
        let ctx = cv.getContext('2d')!;
        if (labelled) {
            // A token x token matrix, large enough to read which token looks at which.
            let dpr = 2, cw = Math.min(26, Math.floor(330 / cols)), gutter = 84, head = 66;
            cv.width = (gutter + cw * cols) * dpr;
            cv.height = (head + cw * rows) * dpr;
            cv.style.width = `${gutter + cw * cols}px`;
            ctx.scale(dpr, dpr);
            ctx.font = '11px system-ui, "Noto Sans CJK SC", sans-serif';
            ctx.fillStyle = '#22242a';
            let show = (x: string) => JSON.stringify(x).slice(1, -1).slice(0, 12);
            for (let r = 0; r < rows; r++) {
                ctx.textAlign = 'right';
                ctx.fillText(show(tokens[r]), gutter - 5, head + r * cw + cw * 0.7);
            }
            for (let c = 0; c < cols; c++) {
                ctx.save();
                ctx.translate(gutter + c * cw + cw * 0.7, head - 4);
                ctx.rotate(-Math.PI / 3);
                ctx.textAlign = 'left';
                ctx.fillText(show(tokens[c]), 0, 0);
                ctx.restore();
            }
            for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
                let k = color(r * cols + c);
                ctx.fillStyle = `rgb(${k[0] | 0},${k[1] | 0},${k[2] | 0})`;
                ctx.fillRect(gutter + c * cw, head + r * cw, cw - 1, cw - 1);
            }
        } else {
            cv.width = cols; cv.height = rows;
            cv.style.width = '100%';
            let img = ctx.createImageData(cols, rows);
            for (let i = 0; i < rows * cols; i++) {
                let k = color(i);
                img.data[i * 4] = k[0]; img.data[i * 4 + 1] = k[1]; img.data[i * 4 + 2] = k[2]; img.data[i * 4 + 3] = 255;
            }
            ctx.putImageData(img, 0, 0);
        }
    }, [block, cols, rows, t, offset, labelled, tokens]);
    let shown = t.shape.join(' × ');
    let full = (t.full_shape ?? t.shape).join(' × ');
    return <section className="q-insp" aria-label="Block detail">
        <div className="q-insp-head">
            <h3>{block.title}</h3>
            <button onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="q-insp-canvas"><canvas ref={ref} className={labelled ? 'labelled' : ''} style={labelled ? undefined : { aspectRatio: `${cols} / ${Math.max(rows, cols / 10)}` }} /></div>
        {labelled && <p className="q-note q-insp-axis">Rows: reading token. Columns: earlier token.</p>}
        <dl>
            <dt>Shape</dt><dd>{full}</dd>
            {sampled && <><dt>Shown</dt><dd>{shown} (exact values, every {t.stride!.join(' × ')}th)</dd></>}
            {!sampled && !sl && <><dt>Shown</dt><dd>every value</dd></>}
            {sl && <><dt>Shown</dt><dd>{sl.note}</dd></>}
            <dt>Mean</dt><dd>{fmt(t.full_mean ?? t.mean)}</dd>
            <dt>Std</dt><dd>{fmt(t.full_std ?? t.std)}</dd>
            {t.full_min !== undefined && <><dt>Range</dt><dd>{fmt(t.full_min)} to {fmt(t.full_max!)}</dd></>}
            {t.full_min === undefined && <><dt>Max |x|</dt><dd>{fmt(t.scale)}</dd></>}
        </dl>
        <p className="q-note">{sampled ? 'Mean, std and range cover the whole matrix. ' : sl ? 'Mean, std and max cover all heads. ' : ''}Stored at 8 bits, so values are accurate to ±{fmt(t.scale / 254)}. Colour saturates at the 99th percentile.</p>
    </section>;
};
