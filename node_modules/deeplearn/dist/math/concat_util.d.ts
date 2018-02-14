export declare function assertParams(aShape: number[], bShape: number[], axis: number): void;
export declare function computeOutShape1D(x1Shape: number[], x2Shape: number[]): number[];
export declare function computeOutShape(x1Shape: number[], x2Shape: number[], axis: number): number[];
export declare function computeGradientSliceShapes2D(x1TensorShape: number[], yTensorShape: number[], axis: number): {
    x1Begin: [number, number];
    x1Size: [number, number];
    x2Begin: [number, number];
    x2Size: [number, number];
};
export declare function computeGradientSliceShapes3D(x1Shape: number[], yShape: number[], axis: number): {
    x1Begin: [number, number, number];
    x1Size: [number, number, number];
    x2Begin: [number, number, number];
    x2Size: [number, number, number];
};
export declare function computeGradientSliceShapes4D(x1TensorShape: number[], yTensorShape: number[], axis: number): {
    x1Begin: [number, number, number, number];
    x1Size: [number, number, number, number];
    x2Begin: [number, number, number, number];
    x2Size: [number, number, number, number];
};
