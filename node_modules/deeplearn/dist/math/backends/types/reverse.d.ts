import { NamedArrayMap } from '../../../util';
import { Array4D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface Reverse4DNode extends KernelNode {
    inputAndArgs: Reverse4DInputConfig;
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => Reverse4DGradientInputArrays;
}
export interface Reverse4DInputConfig extends KernelInputConfig {
    inputs: Reverse4DInputArrays;
    args: {
        axis: number[];
    };
}
export interface Reverse4DInputArrays extends NamedArrayMap {
    x: Array4D;
}
export interface Reverse4DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array4D;
}
