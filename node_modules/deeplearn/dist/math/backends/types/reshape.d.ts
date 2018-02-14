import { NDArray } from '../../ndarray';
import { KernelInputConfig, KernelNode } from '../tape_types';
export interface ReshapeNode extends KernelNode {
    inputAndArgs: ReshapeInputConfig;
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        x: () => NDArray;
    };
}
export interface ReshapeInputConfig extends KernelInputConfig {
    inputs: {
        x: NDArray;
    };
    args: {
        newShape: number[];
    };
}
