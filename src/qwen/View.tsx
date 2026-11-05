'use client';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { fetchFontAtlasData, IFontAtlasData } from '@/src/llm/render/fontRender';
import { Vec3 } from '@/src/utils/vector';
import { fetchEmbedWeights, fetchIndex, fetchLayer, fetchTrace, fmt, IIndex, ITrace, showTok } from './data';
import { IQBlock } from './layout';
import { axisText, DEFAULT_FULL, DEFAULT_LINEAR, hitTest, initState, IQState, project, runFrame, setTrace } from './program';
import { ICtx, STEPS } from './steps';
import './qwen.scss';

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

    if (error) return <div className="q-msg">The trace could not be loaded: {error}</div>;
    if (!index || !font || !trace) return <div className="q-msg">Loading the trace…</div>;
    return <Scene index={index} font={font} trace={trace} promptKey={promptKey} setPromptKey={setPromptKey} />;
}

const Scene: React.FC<{ index: IIndex; font: IFontAtlasData; trace: ITrace; promptKey: string; setPromptKey: (k: string) => void }> = ({ index, font, trace, promptKey, setPromptKey }) => {
    let canvasRef = useRef<HTMLCanvasElement>(null);
    let overlayRef = useRef<HTMLDivElement>(null);
    let tipRef = useRef<HTMLDivElement>(null);
    let bodyRef = useRef<HTMLDivElement>(null);
    let stRef = useRef<IQState | null>(null);
    let dirtyRef = useRef<() => void>(() => { });
    let [glError, setGlError] = useState(false);
    let [stepIdx, setStepIdx] = useState(0);
    let [layer, setLayer] = useState(DEFAULT_LINEAR);
    let [tokenCursor, setTokenCursor] = useState(0);
    let [playing, setPlaying] = useState(false);
    let [selected, setSelected] = useState<IQBlock | null>(null);
    let [showToc, setShowToc] = useState(false);
    let [frontView, setFrontView] = useState(false);
    let [, setLayoutVer] = useState(0);
    let [, setDataTick] = useState(0);
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
                if (sizeDirty) st!.pendingFit = true;
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
        st.overlay.onLayout = v => setLayoutVer(v);
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

    // ---- HTML labels that follow the 3D scene; they can show any Unicode token ----
    useEffect(() => {
        let st = stRef.current;
        if (!st) return;
        st.overlay.onFrame = (s) => {
            let box = overlayRef.current;
            if (!box) return;
            let sz = s.render.size;
            box.querySelectorAll<HTMLElement>('[data-a]').forEach(el => {
                let kind = el.dataset.a, p: Vec3 | null = null;
                let vis = true;
                if (kind === 'lbl') {
                    let b = s.layout.byKey[el.dataset.k!];
                    if (b) {
                        // A title moves below its block only when token labels really occupy the top edge.
                        let [x0] = project(s, new Vec3(b.x, b.y, b.z + b.dz));
                        let [x1] = project(s, new Vec3(b.x + b.dx / b.cx, b.y, b.z + b.dz));
                        let [xr] = project(s, new Vec3(b.x + b.dx + 18, b.y, b.z + b.dz));
                        let below = !!el.dataset.tokentop && Math.abs(x1 - x0) > 10.5;
                        el.classList.toggle('below', below);
                        (el.firstElementChild as HTMLElement).style.width = `${Math.max(58, Math.round(xr - x0))}px`;
                        p = new Vec3(b.x, below ? b.y + b.dy : b.y, b.z + b.dz);
                    }
                } else if (kind === 'op') {
                    let o = s.layout.ops[+el.dataset.i!];
                    if (o) p = new Vec3(o.x, o.y, o.z);
                } else if (kind === 'tok') {
                    let b = s.layout.byKey[el.dataset.k!], i = +el.dataset.i!;
                    if (b) {
                        if (el.dataset.axis === 'top') {
                            p = new Vec3(b.x + (i + 0.5) * b.dx / b.cx, b.y, b.z + b.dz);
                            let [x0] = project(s, new Vec3(b.x, b.y, b.z)), [x1] = project(s, new Vec3(b.x + b.dx / b.cx, b.y, b.z));
                            vis = Math.abs(x1 - x0) > 10.5;
                        } else {
                            p = new Vec3(b.x, b.y + (i + 0.5) * b.dy / b.cy, b.z + b.dz);
                        }
                    }
                } else if (kind === 'tag') {
                    let t = s.layout.tags.find(g => g.layer === +el.dataset.i!);
                    if (t) p = new Vec3(t.x, t.y, t.z);
                }
                if (!p) { el.style.display = 'none'; return; }
                let [x, y, ok] = project(s, p);
                if (kind === 'lbl' && el.classList.contains('below')) {
                    // The bottom edge slopes in 3D: sit below its lowest point, not its left end.
                    let b = s.layout.byKey[el.dataset.k!];
                    let [, yr] = project(s, new Vec3(b.x + b.dx, b.y + b.dy, b.z + b.dz));
                    y = Math.max(y, yr);
                }
                vis = vis && ok && x > -60 && x < sz.x + 60 && y > -20 && y < sz.y + 20;
                el.style.display = vis ? 'block' : 'none';
                if (vis) el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
            });
            let h = s.hover, tip = tipRef.current;
            if (tip) {
                if (h && h.block.tensor) {
                    tip.style.display = 'block';
                    let tx = Math.min(h.screen[0] + 14, sz.x - 230);
                    tip.style.transform = `translate(${tx}px, ${h.screen[1] + 14}px)`;
                    tip.innerHTML = '';
                    let add = (t: string, c?: string) => { let d = document.createElement('div'); d.textContent = t; if (c) d.className = c; tip!.appendChild(d); };
                    add(h.block.title, 'q-tip-title');
                    axisText(s, h).forEach(l => add(l));
                    if (h.value !== null) add(`value ≈ ${fmt(h.value)}`);
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
        setSelected(null);
        setTokenCursor(c => Math.min(c, trace.manifest.T - 1));
        let needLayer = STEPS[st.stepIdx].layer !== 'none';
        let apply = () => { setTrace(st!, trace); setDataTick(n => n + 1); dirtyRef.current(); };
        if (needLayer) fetchLayer(trace, st.layer).then(apply); else apply();
    }, [trace]);

    // ---- Moving between steps and layers ----
    let goTo = useCallback(async (i: number, requested?: number) => {
        let st = stRef.current;
        i = Math.max(0, Math.min(STEPS.length - 1, i));
        setPlaying(false);
        setShowToc(false);
        setSelected(null);
        if (!st) { setStepIdx(i); return; }
        st.selectedKey = null;
        let s = STEPS[i], types = st.index.layer_types;
        let l = requested ?? st.layer;
        if (s.layer === 'linear' && types[l] !== 'linear_attention') l = requested !== undefined ? l : DEFAULT_LINEAR;
        if (s.layer === 'full' && types[l] !== 'full_attention') l = requested !== undefined ? l : DEFAULT_FULL;
        if (s.layer !== 'none') await fetchLayer(st.trace, l);
        st.stepIdx = i;
        st.layer = l;
        st.layoutDirty = true;
        st.pendingFit = true;
        setStepIdx(i);
        setLayer(l);
        setDataTick(n => n + 1);
        bodyRef.current?.scrollTo(0, 0);
        dirtyRef.current();
    }, []);

    // Choose a layer. If this section does not apply to that kind of layer, open its section 3.1.
    let chooseLayer = useCallback((l: number) => {
        let st = stRef.current;
        if (!st) return;
        let kind = st.index.layer_types[l] === 'full_attention' ? 'full' : 'linear';
        let cur = STEPS[st.stepIdx].layer;
        let ok = cur === 'any' || cur === kind;
        goTo(ok ? st.stepIdx : STEPS.findIndex(s => s.id === 'layer'), l);
    }, [goTo]);

    // ---- Keyboard ----
    useEffect(() => {
        let onKey = (e: KeyboardEvent) => {
            let tag = (e.target as HTMLElement)?.tagName;
            if (tag === 'SELECT' || tag === 'INPUT') return;
            if (e.key === 'ArrowRight' || e.key === 'PageDown') { goTo(stepIdx + 1); e.preventDefault(); }
            if (e.key === 'ArrowLeft' || e.key === 'PageUp') { goTo(stepIdx - 1); e.preventDefault(); }
            if (e.key === 'Escape') {
                setSelected(null); setShowToc(false);
                if (stRef.current) stRef.current.selectedKey = null;
                dirtyRef.current();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [stepIdx, goTo]);

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
        }, 800);
        return () => clearInterval(id);
    }, [playing, T]);

    let activate = useCallback((b: IQBlock) => {
        let st = stRef.current;
        if (!st) return;
        if (b.layer !== null) { chooseLayer(b.layer); return; }
        setSelected(b.tensor ? b : null);
        st.selectedKey = b.tensor ? b.key : null;
    }, [chooseLayer]);

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
            if (st && d && !d.moved && st.hover) activate(st.hover.block);
            dirtyRef.current();
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    }, [activate]);

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
            if (start.moved) st.mouse = [-1, -1];
            dirtyRef.current();
        };
        let onEnd = (e: TouchEvent) => {
            let st = stRef.current;
            if (!st || !start || e.touches.length > 0) return;
            // A quick tap can end before the next frame has run the hit test.
            if (!start.moved) { hitTest(st); if (st.hover) activate(st.hover.block); }
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
    }, [activate]);

    function onWheel(e: React.WheelEvent) {
        let st = stRef.current;
        if (!st) return;
        st.camera.desiredCameraTransition = undefined;
        let a = st.camera.angle;
        st.camera.angle = new Vec3(a.x, a.y, Math.max(0.6, Math.min(5000, a.z * Math.pow(1.0013, e.deltaY))));
        dirtyRef.current();
    }

    let st = stRef.current;
    let data = trace.layers.get(layer) ?? null;
    let ctx: ICtx = { index, trace, layer, act: data?.act ?? null, w: data?.w ?? null, tokenCursor };
    let scene = st?.layout;
    let labelBlocks = scene ? scene.cubes.filter(c => c.label) : [];
    let tokenTexts = trace.manifest.tokens.map(t => showTok(t.text));
    let isTower = step.scene === 'tower';
    let hasTop = (key: string) => scene?.tokenAxes.some(a => a.key === key && a.axis === 'top');

    return <div className="q-root">
        <aside className="q-paper">
            <header className="q-running">
                <span>Qwen3.5-4B: one forward pass</span>
                <button onClick={() => setShowToc(v => !v)} aria-expanded={showToc}>Contents</button>
            </header>
            {showToc ? <nav className="q-toc" aria-label="Contents">
                <ol>
                    {STEPS.map((s, i) => <li key={s.id} className={s.num.includes('.') ? 'sub' : ''}>
                        <button className={i === stepIdx ? 'on' : ''} onClick={() => goTo(i)}><span>{s.num}</span> {s.title}</button>
                    </li>)}
                </ol>
            </nav> : <article className="q-body" ref={bodyRef} key={step.id}>
                <h2><span>{step.num}</span> {step.title}</h2>
                {step.body(ctx)}
            </article>}
            <footer className="q-nav">
                <button disabled={stepIdx === 0} onClick={() => goTo(stepIdx - 1)}>Previous</button>
                <span>{stepIdx + 1} of {STEPS.length}</span>
                <button disabled={stepIdx === STEPS.length - 1} onClick={() => goTo(stepIdx + 1)}>Next</button>
            </footer>
        </aside>

        <main className="q-figure">
            <div className="q-controls">
                <label>
                    <span>Prompt</span>
                    <select aria-label="Prompt" value={promptKey} onChange={e => setPromptKey(e.target.value)}>
                        {index.prompts.map(p => <option key={p.key} value={p.key}>{p.text.replace('\n', ' ⏎ ')}</option>)}
                    </select>
                </label>
                {step.layer !== 'none' && <div className="q-strip" role="group" aria-label="Layer">
                    <span>Layer {layer}, {index.layer_types[layer] === 'full_attention' ? 'full' : 'linear'}</span>
                    <div>
                        {index.layer_types.map((t, i) => <button key={i} aria-label={`Layer ${i}, ${t === 'full_attention' ? 'full attention' : 'linear attention'}`}
                            title={`Layer ${i}`} className={`${t === 'full_attention' ? 'full' : ''} ${i === layer ? 'on' : ''}`} onClick={() => chooseLayer(i)} />)}
                    </div>
                </div>}
                {step.tokenSlider && <div className="q-slider">
                    <button onClick={() => { if (tokenCursor >= T - 1) setTokenCursor(0); setPlaying(p => !p); }}>{playing ? 'Pause' : 'Play'}</button>
                    <input type="range" aria-label="Token" min={0} max={T - 1} value={tokenCursor} onChange={e => { setPlaying(false); setTokenCursor(+e.target.value); }} />
                    <span>after <code className="q-t">{showTok(trace.manifest.tokens[tokenCursor]?.text ?? '')}</code></span>
                </div>}
                <button className="q-viewbtn" onClick={() => {
                    let st = stRef.current;
                    if (!st) return;
                    st.frontView = !frontView;
                    st.pendingFit = true;
                    setFrontView(!frontView);
                    dirtyRef.current();
                }}>{frontView ? '3D view' : 'Front view'}</button>
            </div>
            <div className="q-stage">
                <canvas ref={canvasRef} className="q-canvas" onMouseDown={onDown} onWheel={onWheel} onContextMenu={e => e.preventDefault()} />
                {glError && <div className="q-msg">This page requires a browser with WebGL2.</div>}
                <div ref={overlayRef} className="q-overlay">
                    {labelBlocks.map(b => <div key={b.key} data-a="lbl" data-k={b.key} data-tokentop={scene?.tokenAxes.some(a => a.key === b.key && a.axis === 'top') ? '1' : undefined}
                        className="q-lbl" style={{ display: 'none' }}>
                        <div><b>{b.title}</b>{b.shapeText && <span>{b.shapeText}</span>}</div>
                    </div>)}
                    {scene?.ops.map((o, i) => <div key={`o${i}`} data-a="op" data-i={i} className={o.kind ? `q-note-${o.kind}` : 'q-op'} style={{ display: 'none' }}><span>{o.text}</span></div>)}
                    {scene?.tokenAxes.flatMap(ax => tokenTexts.map((t, i) => <div key={`${ax.key}${ax.axis}${i}`} data-a="tok" data-k={ax.key} data-i={i} data-axis={ax.axis}
                        className={`q-tk q-tk-${ax.axis}`} style={{ display: 'none' }}><span>{t}</span></div>))}
                    {isTower && index.layer_types.map((t, i) => {
                        let lens = trace.manifest.lens[i][0];
                        let show = step.lensTags || i % 4 === 3 || i === 0;
                        return show ? <div key={`l${i}`} data-a="tag" data-i={i} className="q-tag" style={{ display: 'none' }}>
                            <div>
                                <span className="q-tag-n">L{i}</span>
                                {step.lensTags && <code className="q-t">{showTok(lens.text)}</code>}
                                {step.lensTags && <span className="q-tag-p">{(lens.p * 100).toFixed(0)}%</span>}
                            </div>
                        </div> : null;
                    })}
                </div>
                <div ref={tipRef} className="q-tip" style={{ display: 'none' }} />
                {selected && selected.tensor && <Inspector block={selected} tokens={trace.manifest.tokens.map(t => t.text)} onClose={() => { setSelected(null); if (stRef.current) stRef.current.selectedKey = null; dirtyRef.current(); }} />}
            </div>
            <figcaption className="q-caption">
                <div className="q-legend" aria-hidden={false}>
                    {(step.legend === 'div' || step.legend === 'both') && <span className="q-scale"><i className="div" /><small>−</small><small>0</small><small>+</small></span>}
                    {(step.legend === 'seq' || step.legend === 'both') && <span className="q-scale"><i className="seq" /><small>0</small><small /><small>1</small></span>}
                </div>
                <p><b>Figure {stepIdx + 1}:</b> {step.caption(ctx)} Prompt: <code className="q-t">{trace.manifest.text.replace('\n', '⏎')}</code></p>
            </figcaption>
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
        let seq = block.t === 'a';
        let pos = seq ? [26, 31, 43] : [178, 51, 43];
        // Same colour scale as the 3D view.
        let s = seq && t.scale <= 1 && !sl ? 1 / 127 : t.scale / 127 / (t.p99 || 1);
        let color = (i: number) => {
            let v = Math.max(-1, Math.min(1, t.data[offset + i] * s));
            let a = Math.abs(v), tgt = v >= 0 ? pos : [33, 99, 171];
            return [241 + (tgt[0] - 241) * a, 239 + (tgt[1] - 239) * a, 236 + (tgt[2] - 236) * a];
        };
        let ctx = cv.getContext('2d')!;
        if (labelled) {
            // A token × token matrix, large enough to read which token relates to which.
            let dpr = 2, cw = Math.min(26, Math.floor(330 / cols)), gutter = 84, head = 66;
            cv.width = (gutter + cw * cols) * dpr;
            cv.height = (head + cw * rows) * dpr;
            cv.style.width = `${gutter + cw * cols}px`;
            ctx.scale(dpr, dpr);
            ctx.font = '11px ui-monospace, Menlo, Consolas, "Noto Sans CJK SC", monospace';
            ctx.fillStyle = '#22242a';
            let show = (x: string) => showTok(x).slice(0, 12);
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
    }, [block, cols, rows, t, offset, labelled, tokens, sl]);
    let shown = t.shape.join(' × ');
    let full = (t.full_shape ?? t.shape).join(' × ');
    return <section className="q-insp" aria-label="Block detail">
        <div className="q-insp-head">
            <h3>{block.title}</h3>
            <button onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="q-insp-canvas"><canvas ref={ref} className={labelled ? 'labelled' : ''} style={labelled ? undefined : { aspectRatio: `${cols} / ${Math.max(rows, cols / 10)}` }} /></div>
        <dl>
            <dt>Shape</dt><dd>{full}</dd>
            {sampled && <><dt>Shown</dt><dd>{shown}: exact values, every {t.stride!.join(' × ')}th</dd></>}
            {!sampled && !sl && <><dt>Shown</dt><dd>All stored values</dd></>}
            {sl && <><dt>Shown</dt><dd>{sl.note}</dd></>}
            <dt>Mean</dt><dd>{fmt(t.full_mean ?? t.mean)}</dd>
            <dt>Std</dt><dd>{fmt(t.full_std ?? t.std)}</dd>
            {t.full_min !== undefined ? <><dt>Range</dt><dd>{fmt(t.full_min)} to {fmt(t.full_max!)}</dd></> : <><dt>Max |x|</dt><dd>{fmt(t.scale)}</dd></>}
        </dl>
        <p className="q-note">{sampled ? 'Statistics cover the whole matrix. ' : sl && t.shape.length > 2 ? 'Statistics cover all heads. ' : ''}Stored at 8 bits, to within ±{fmt(t.scale / 254)}. The colour limit is the 99th percentile.</p>
    </section>;
};
