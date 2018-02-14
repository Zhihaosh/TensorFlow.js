import { NamedArrayMap } from '../../../util';
import { Array1D, Array2D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface Pad1DNode extends KernelNode {
    inputAndArgs: Pad1DInputConfig;
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => Pad1DGradientInputArrays;
}
export interface Pad1DInputConfig extends KernelInputConfig {
    inputs: Pad1DInputArrays;
    args: {
        paddings: [number, number];
        constantValue: number;
    };
}
export interface Pad1DInputArrays extends NamedArrayMap {
    x: Array1D;
}
export interface Pad1DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array1D;
}
export interface Pad2DNode extends KernelNode {
    inputAndArgs: Pad2DInputConfig;
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => Pad2DGradientInputArrays;
}
export interface Pad2DInputConfig extends KernelInputConfig {
    inputs: Pad2DInputArrays;
    args: {
        paddings: [[number, number], [number, number]];
        constantValue: number;
    };
}
export interface Pad2DInputArrays extends NamedArrayMap {
    x: Array2D;
}
export interface Pad2DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array2D;
}
