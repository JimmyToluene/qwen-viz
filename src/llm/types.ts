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
    t: 'w' | 'i' | 'a'; // weights; computed value; value in [0, 1]
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
    // Positive pole of each block type. 'w' and 'i' share the diverging scale; 'a' is a
    // one-hue ramp for values in [0, 1], such as attention weights and probabilities.
    Weights: new Vec4(0.70, 0.20, 0.17),
    Intermediates: new Vec4(0.70, 0.20, 0.17),
    Aggregates: new Vec4(0.10, 0.12, 0.17),
    Black: new Vec4(0, 0, 0),
};

export interface IProgramState {
    camera: ICamera;
    layout: IModelLayout;
    render: IRenderState;
    examples: { enabled: boolean; layout?: IModelLayout; offset: Vec3; blockRender: any }[];
}
