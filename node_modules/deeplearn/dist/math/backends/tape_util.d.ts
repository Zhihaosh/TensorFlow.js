import { NDArray } from '../ndarray';
import { Tape, TapeNodeInputConfig } from './tape_types';
export declare function getFilteredNodesXToY(tape: Tape, xs: NDArray[], y: NDArray): Tape;
export declare function backpropagateGradients(arrayAccumulatedGradientMap: {
    [ndarrayId: number]: NDArray;
}, filteredTape: Tape): void;
export declare function computeInputs(tape: Tape): {
    [idx: string]: NDArray;
};
export declare type ScopeResultImmediate = void | NDArray | NDArray[] | {
    [key: string]: NDArray;
};
export declare type ScopeResult = ScopeResultImmediate | Promise<ScopeResultImmediate>;
export declare function extractNDArraysFromScopeResult(result: ScopeResultImmediate): NDArray[];
export declare function stripUndefinedInputsFromInputConfig(config: TapeNodeInputConfig): TapeNodeInputConfig;
