import { Node, VariableNode } from '../../graph/graph';
import { SessionRuntime } from '../../graph/session';
import { SummedTensorArrayMap, TensorArrayMap } from '../../graph/tensor_array_map';
import { NDArrayMath } from '../../math/math';
import { DataType, Scalar } from '../../math/ndarray';
import { NamedVariableMap } from '../../util';
export declare abstract class Optimizer {
    protected learningRate: number;
    protected variableNodes: VariableNode[];
    protected specifiedVariableNodes: VariableNode[] | null;
    constructor(learningRate: number, specifiedVariableList?: Node[]);
    minimize<D extends DataType>(f: () => Scalar<D>, returnCost?: boolean): Scalar<D> | null;
    computeGradients<D extends DataType>(f: () => Scalar<D>): {
        value: Scalar<D>;
        gradients: NamedVariableMap;
    };
    abstract applyGradients(variableGradients: NamedVariableMap): void;
    beforeBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    afterExample(math: NDArrayMath, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    abstract afterBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    dispose(): void;
    protected variableGradients: TensorArrayMap;
    protected prevBatchSize: number;
    protected one: Scalar;
    protected cGraph: Scalar;
}
