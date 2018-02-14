import { NamedArrayMap } from '../../../util';
import { NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface BinaryNode extends KernelNode {
    inputAndArgs: BinaryInputConfig;
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => BinaryInputGradientArrays;
}
export interface BinaryInputConfig extends KernelInputConfig {
    inputs: BinaryInputArrays;
}
export interface BinaryInputArrays extends NamedArrayMap {
    a: NDArray;
    b: NDArray;
}
export interface BinaryInputGradientArrays extends TapeNodeInputGradientArrays {
    a: () => NDArray;
    b: () => NDArray;
}
