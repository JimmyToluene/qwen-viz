// Minimal types the reused llm-viz renderer needs. The original GPT layout and
// walkthrough were removed; the Qwen layout lives in src/qwen.
import { Mat4f } from "@/src/utils/matrix";
import { IBufferTex } from "@/src/utils/renderPhases";
import { Vec3, Vec4 } from "@/src/utils/vector";
import type { ICamera } from "./Camera";
import type { IRenderState } from "./render/modelRender";

export interface IBlkAccess {
    src: IBufferTex;
    channel: 'r' | 'g' | 'b';
    scale: number;
    mat: Mat4f; // first 4 floats: texX coefficients of cell x,y,z,1; next 4: texY
    disable?: boolean;
}

export interface IBlkDef {
    idx: number;
    t: 'w' | 'i' | 'a'; // weights; intermediate value; other (used for full-attention layers)
    x: number; y: number; z: number;
    dx: number; dy: number; dz: number;
    cx: number; cy: number; cz: number;
    access?: IBlkAccess;
    name: string;
    localMtx?: Mat4f;
    highlight: number;
    opacity: number;
    small: boolean;
    subs?: IBlkDef[];
}

export interface IModelLayout {
    cell: number;
    height: number;
    margin: number;
    cubes: IBlkDef[];
}

export const Colors = {
    Weights: new Vec4(0.3, 0.3, 1.0),
    Intermediates: new Vec4(0.4, 0.8, 0.4),
    Aggregates: new Vec4(1.0, 0.8, 0.3),
    Black: new Vec4(0, 0, 0),
};

export interface IProgramState {
    camera: ICamera;
    layout: IModelLayout;
    render: IRenderState;
    examples: { enabled: boolean; layout?: IModelLayout; offset: Vec3; blockRender: any }[];
}
