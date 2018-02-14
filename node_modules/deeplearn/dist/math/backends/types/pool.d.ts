import { NamedArrayMap } from '../../../util';
import { Conv2DInfo } from '../../conv_util';
import { Array4D } from '../../ndarray';
import { KernelInputConfig, KernelNode, TapeNodeInputGradientArrays } from '../tape_types';
export interface PoolNode extends KernelNode {
    inputAndArgs: PoolInputConfig;
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => PoolGradientInputArrays;
}
export interface PoolInputConfig extends KernelInputConfig {
    inputs: PoolInputArrays;
    args: {
        convInfo: Conv2DInfo;
    };
}
export interface PoolInputArrays extends NamedArrayMap {
    x: Array4D;
}
export interface PoolGradientInputArrays extends TapeNodeInputGradientArrays {
    x: () => Array4D;
}
export interface PoolBackpropNode extends KernelNode {
    inputAndArgs: PoolInputConfig;
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => PoolBackpropGradientInputArrays;
}
export interface PoolBackpropInputConfig extends KernelInputConfig {
    inputs: PoolBackpropInputArrays;
    args: {
        convInfo: Conv2DInfo;
    };
}
export interface PoolBackpropInputArrays extends NamedArrayMap {
    dy: Array4D;
    x: Array4D;
}
export interface PoolBackpropGradientInputArrays extends TapeNodeInputGradientArrays {
    dy: () => Array4D;
    x: () => Array4D;
}
