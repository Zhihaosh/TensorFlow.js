import { NamedArrayMap } from '../../../util';
import { Array1D, Array2D, Array3D, Array4D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface BatchNorm4DNode extends KernelNode {
    inputAndArgs: BatchNorm4DInputConfig;
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => BatchNorm4DGradientInputArrays;
}
export interface BatchNorm4DInputConfig extends KernelInputConfig {
    inputs: BatchNorm4DInputArrays;
    args: {
        varianceEpsilon: number;
    };
}
export interface BatchNorm4DInputArrays extends NamedArrayMap {
    x: Array4D;
    mean: Array4D | Array1D;
    variance: Array4D | Array1D;
    scale?: Array4D | Array1D;
    offset?: Array4D | Array1D;
}
export interface BatchNorm4DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array4D;
    mean: () => Array4D | Array1D;
    variance: () => Array4D | Array1D;
    scale?: () => Array4D | Array1D;
    offset?: () => Array4D | Array1D;
}
export interface BatchNorm3DNode extends KernelNode {
    inputAndArgs: BatchNorm3DInputConfig;
    output: Array3D;
    gradient: (dy: Array3D, y: Array3D) => BatchNorm3DGradientInputArrays;
}
export interface BatchNorm3DInputConfig extends KernelInputConfig {
    inputs: BatchNorm3DInputArrays;
    args: {
        varianceEpsilon: number;
    };
}
export interface BatchNorm3DInputArrays extends NamedArrayMap {
    x: Array3D;
    mean: Array3D | Array1D;
    variance: Array3D | Array1D;
    scale?: Array3D | Array1D;
    offset?: Array3D | Array1D;
}
export interface BatchNorm3DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array3D;
    mean: () => Array3D | Array1D;
    variance: () => Array3D | Array1D;
    scale?: () => Array3D | Array1D;
    offset?: () => Array3D | Array1D;
}
export interface BatchNorm2DNode extends KernelNode {
    inputAndArgs: BatchNorm2DInputConfig;
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => BatchNorm2DGradientInputArrays;
}
export interface BatchNorm2DInputConfig extends KernelInputConfig {
    inputs: BatchNorm2DInputArrays;
    args: {
        varianceEpsilon: number;
    };
}
export interface BatchNorm2DInputArrays extends NamedArrayMap {
    x: Array2D;
    mean: Array2D | Array1D;
    variance: Array2D | Array1D;
    scale?: Array2D | Array1D;
    offset?: Array2D | Array1D;
}
export interface BatchNorm2DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array2D;
    mean: () => Array2D | Array1D;
    variance: () => Array2D | Array1D;
    scale?: () => Array2D | Array1D;
    offset?: () => Array2D | Array1D;
}
