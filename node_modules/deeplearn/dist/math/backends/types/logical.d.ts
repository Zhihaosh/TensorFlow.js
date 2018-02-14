import { NamedArrayMap } from '../../../util';
import { NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface DualInputArrays extends NamedArrayMap {
    a: NDArray;
    b: NDArray;
}
export interface DualGradientInputArrays extends TapeNodeInputGradientArrays {
    a: () => NDArray;
    b: () => NDArray;
}
export interface EqualNode extends KernelNode {
    inputAndArgs: EqualInputConfig;
    output: NDArray<'bool'>;
    gradient: (dy: NDArray<'bool'>, y: NDArray<'bool'>) => DualGradientInputArrays;
}
export interface EqualInputConfig extends KernelInputConfig {
    inputs: DualInputArrays;
}
export interface LogicalOrNode extends KernelNode {
    inputAndArgs: LogicalOrInputConfig;
    output: NDArray<'bool'>;
    gradient: (dy: NDArray<'bool'>, y: NDArray<'bool'>) => DualGradientInputArrays;
}
export interface LogicalOrInputConfig extends KernelInputConfig {
    inputs: DualInputArrays;
}
