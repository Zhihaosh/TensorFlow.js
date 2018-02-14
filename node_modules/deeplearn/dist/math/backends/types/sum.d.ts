import { NamedArrayMap } from '../../../util';
import { DataType, NDArray } from '../../ndarray';
import { SumTypes } from '../../types';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface SumNode<D extends DataType> extends KernelNode {
    inputAndArgs: SumInputConfig<D>;
    output: NDArray<SumTypes[D]>;
    gradient: (dy: NDArray<SumTypes[D]>, y: NDArray<SumTypes[D]>) => SumGradientInputArrays<D>;
}
export interface SumInputConfig<D extends DataType> extends KernelInputConfig {
    inputs: SumInputArrays<D>;
    args: {
        axes: number[];
    };
}
export interface SumInputArrays<D extends DataType> extends NamedArrayMap {
    x: NDArray<D>;
}
export interface SumGradientInputArrays<D extends DataType> extends TapeNodeInputGradientArrays {
    x: () => NDArray<D>;
}
