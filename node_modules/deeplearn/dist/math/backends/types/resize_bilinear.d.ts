import { NamedArrayMap } from '../../../util';
import { Array3D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface ResizeBilinear3DNode extends KernelNode {
    inputAndArgs: ResizeBilinear3DInputConfig;
    output: Array3D;
    gradient: (dy: Array3D, y: Array3D) => ResizeBilinear3DGradientInputArrays;
}
export interface ResizeBilinear3DInputConfig extends KernelInputConfig {
    inputs: ResizeBilinear3DInputArrays;
    args: {
        newShape2D: [number, number];
        alignCorners: boolean;
    };
}
export interface ResizeBilinear3DInputArrays extends NamedArrayMap {
    x: Array3D;
}
export interface ResizeBilinear3DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array3D;
}
