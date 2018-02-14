import { NamedArrayMap } from '../../../util';
import { DataType, NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface MinNode<D extends DataType> extends KernelNode {
    inputAndArgs: MinInputConfig<D>;
    output: NDArray<D>;
    gradient: (dy: NDArray<D>, y: NDArray<D>) => MinGradientInputArrays<D>;
}
export interface MinimumNode<D extends DataType> extends KernelNode {
    inputAndArgs: MinimumInputConfig<D>;
    output: NDArray<D>;
    gradient: (dy: NDArray<D>, y: NDArray<D>) => {
        a: () => NDArray<D>;
        b: () => NDArray<D>;
    };
}
export interface MinimumInputConfig<D extends DataType> extends KernelInputConfig {
    inputs: {
        a: NDArray<D>;
        b: NDArray<D>;
    };
}
export interface MinInputConfig<D extends DataType> extends KernelInputConfig {
    inputs: MinInputArrays<D>;
}
export interface MinInputArrays<D extends DataType> extends NamedArrayMap {
    x: NDArray<D>;
}
export interface MinGradientInputArrays<D extends DataType> extends TapeNodeInputGradientArrays {
    x: () => NDArray<D>;
}
export interface MaxNode<D extends DataType> extends KernelNode {
    inputAndArgs: MaxInputConfig<D>;
    output: NDArray<D>;
    gradient: (dy: NDArray<D>, y: NDArray<D>) => MaxGradientInputArrays<D>;
}
export interface MaximumNode<D extends DataType> extends KernelNode {
    inputAndArgs: MaximumInputConfig<D>;
    output: NDArray<D>;
    gradient: (dy: NDArray<D>, y: NDArray<D>) => {
        a: () => NDArray<D>;
        b: () => NDArray<D>;
    };
}
export interface MaximumInputConfig<D extends DataType> extends KernelInputConfig {
    inputs: {
        a: NDArray<D>;
        b: NDArray<D>;
    };
}
export interface MaxInputConfig<D extends DataType> extends KernelInputConfig {
    inputs: MaxInputArrays<D>;
}
export interface MaxInputArrays<D extends DataType> extends NamedArrayMap {
    x: NDArray<D>;
}
export interface MaxGradientInputArrays<D extends DataType> extends TapeNodeInputGradientArrays {
    x: () => NDArray<D>;
}
