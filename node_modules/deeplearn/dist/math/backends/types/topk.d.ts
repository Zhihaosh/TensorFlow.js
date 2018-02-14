import { NamedArrayMap } from '../../../util';
import { Array1D, DataType, NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface TopKValuesNode<D extends DataType, T extends NDArray<D>> extends KernelNode {
    inputAndArgs: TopKValuesInputConfig<T>;
    output: Array1D<D>;
    gradient: (dy: Array1D<D>, y: Array1D<D>) => TopKValuesGradientInputArrays<T>;
}
export interface TopKValuesInputConfig<T extends NDArray> extends KernelInputConfig {
    inputs: TopKValuesInputArrays<T>;
    args: {
        k: number;
    };
}
export interface TopKValuesInputArrays<T extends NDArray> extends NamedArrayMap {
    x: T;
}
export interface TopKValuesGradientInputArrays<T extends NDArray> extends TapeNodeInputGradientArrays {
    x: () => T;
}
export interface TopKIndicesNode extends KernelNode {
    inputAndArgs: TopKIndicesInputConfig;
    output: Array1D<'int32'>;
    gradient: (dy: Array1D<'int32'>, y: Array1D<'int32'>) => TopKIndicesGradientInputArrays;
}
export interface TopKIndicesInputConfig extends KernelInputConfig {
    inputs: TopKIndicesInputArrays;
    args: {
        k: number;
    };
}
export interface TopKIndicesInputArrays extends NamedArrayMap {
    x: NDArray;
}
export interface TopKIndicesGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => NDArray;
}
