import { NamedArrayMap } from '../../util';
import { DataType, NDArray, Rank, Scalar } from '../ndarray';
import { MathBackend } from './backend';
import { KernelConfigRegistry } from './kernel_registry';
import { TapeNodeInputGradientArrays } from './tape_types';
import { ScopeResult, ScopeResultImmediate } from './tape_util';
export declare class BackendEngine {
    private backend;
    private safeMode;
    private nextTapeNodeId;
    private activeTape;
    private gradientScopeCount;
    private customGradientDepth;
    private activeScope;
    private scopeStack;
    private debugMode;
    constructor(backend: MathBackend, safeMode: boolean);
    enableDebugMode(): void;
    executeKernel<K extends keyof KernelConfigRegistry, C extends KernelConfigRegistry[K]['inputAndArgs']>(kernelName: K, config: C, grad?: KernelConfigRegistry[K]['gradient']): KernelConfigRegistry[K]['output'];
    customGradient<D extends DataType, R extends Rank>(f: () => {
        value: NDArray<D, R>;
        gradients: (dy: NDArray<'float32', R>, y: NDArray<D, R>) => TapeNodeInputGradientArrays;
    }, inputs: NamedArrayMap, name: string): NDArray<D, R>;
    gradients(f: () => Scalar, xs: NDArray[], returnValue: boolean): NDArray[] | {
        value: Scalar;
        gradients: NDArray[];
    };
    vjp<T extends NDArray>(f: () => T, xs: NDArray[], dy: T): NDArray[];
    private gradientWrt<T>(y, xs, dy?);
    scope<T extends ScopeResult>(name: string, scopeFn: (keep: <D1 extends DataType, T1 extends NDArray<D1>>(ndarray: T1) => T1, track: <D2 extends DataType, T2 extends NDArray<D2>>(ndarray: T2) => T2) => T, gradientsMode: boolean): T;
    startScope(gradientsMode: boolean): void;
    endScope(result: ScopeResultImmediate, gradientsMode: boolean): void;
    keep<T extends NDArray>(result: T): T;
    track<D extends DataType, T extends NDArray<D>>(result: T): T;
    getBackend(): MathBackend;
}
