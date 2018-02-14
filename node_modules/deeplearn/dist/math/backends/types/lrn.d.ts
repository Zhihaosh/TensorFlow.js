import { NamedArrayMap } from '../../../util';
import { Array4D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface LRN4DNode extends KernelNode {
    inputAndArgs: LRN4DInputConfig;
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => LRN4DGradientInputArrays;
}
export interface LRN4DInputConfig extends KernelInputConfig {
    inputs: LRN4DInputArrays;
    args: {
        radius: number;
        bias: number;
        alpha: number;
        beta: number;
        normRegion: 'acrossChannels' | 'withinChannel';
    };
}
export interface LRN4DInputArrays extends NamedArrayMap {
    x: Array4D;
}
export interface LRN4DGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array4D;
}
