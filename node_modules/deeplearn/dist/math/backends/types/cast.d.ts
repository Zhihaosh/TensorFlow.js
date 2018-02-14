import { DataType, NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode } from '../tape_types';
export interface CastNode extends KernelNode {
    inputAndArgs: CastInputConfig;
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        x: () => NDArray;
    };
}
export interface CastInputConfig extends KernelInputConfig {
    inputs: {
        x: NDArray;
    };
    args: {
        newDType: DataType;
    };
}
