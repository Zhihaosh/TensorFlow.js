import { NamedArrayMap } from '../../../util';
import { Array1D, Array2D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface OneHotNode extends KernelNode {
    inputAndArgs: OneHotInputConfig;
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => OneHotGradientInputArrays;
}
export interface OneHotInputConfig extends KernelInputConfig {
    inputs: OneHotInputArrays;
    args: {
        depth: number;
        onValue: number;
        offValue: number;
    };
}
export interface OneHotInputArrays extends NamedArrayMap {
    indices: Array1D;
}
export interface OneHotGradientInputArrays extends TapeNodeInputGradientArrays {
    indices: () => Array1D;
}
