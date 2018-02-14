import { NDArrayMath } from './math';
import { RandNormalDataTypes } from './rand';
export declare enum DType {
    float32 = "float32",
    int32 = "int32",
    bool = "bool",
}
export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}
export declare type DataType = keyof DataTypeMap;
export interface RankMap<D extends DataType> {
    0: Scalar<D>;
    1: Array1D<D>;
    2: Array2D<D>;
    3: Array3D<D>;
    4: Array4D<D>;
    higher: NDArray<D, 'higher'>;
}
export declare type Rank = keyof RankMap<DataType>;
export interface NDArrayData<D extends DataType> {
    dataId?: number;
    values?: DataTypeMap[D];
}
export interface ShapeMap {
    0: number[];
    1: [number];
    2: [number, number];
    3: [number, number, number];
    4: [number, number, number, number];
    higher: number[];
}
export declare class NDArray<D extends DataType = DataType, R extends Rank = Rank> {
    private static nextId;
    private static nextDataId;
    id: number;
    dataId: number;
    shape: ShapeMap[R];
    size: number;
    dtype: D;
    rankType: R;
    strides: number[];
    protected math: NDArrayMath;
    protected constructor(shape: number[], dtype: D, values?: DataTypeMap[D], dataId?: number, math?: NDArrayMath);
    static ones<D extends DataType = DataType, R extends Rank = Rank>(shape: number[], dtype?: D): RankMap<D>[R];
    static zeros<D extends DataType = DataType, R extends Rank = Rank>(shape: number[], dtype?: D): RankMap<D>[R];
    static onesLike<T extends NDArray>(another: T): T;
    static zerosLike<T extends NDArray>(another: T): T;
    static like<T extends NDArray>(another: T): T;
    static make<D extends DataType = 'float32', R extends Rank = Rank>(shape: number[], data: NDArrayData<D>, dtype?: D, math?: NDArrayMath): RankMap<D>[R];
    static fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels?: number, math?: NDArrayMath): Array3D<'int32'>;
    reshape<R2 extends Rank>(newShape: number[]): RankMap<D>[R2];
    squeeze<T extends NDArray<D>>(axis?: number[]): T;
    flatten(): Array1D<D>;
    asScalar(): Scalar<D>;
    as1D(): Array1D<D>;
    as2D(rows: number, columns: number): Array2D<D>;
    as3D(rows: number, columns: number, depth: number): Array3D<D>;
    as4D(rows: number, columns: number, depth: number, depth2: number): Array4D<D>;
    asType<D2 extends DataType>(dtype: D2): NDArray<D2, R>;
    readonly rank: number;
    get(...locs: number[]): number;
    add(value: number, ...locs: number[]): void;
    set(value: number, ...locs: number[]): void;
    val(...locs: number[]): Promise<number>;
    locToIndex(locs: ShapeMap[R]): number;
    indexToLoc(index: number): ShapeMap[R];
    fill(value: number): void;
    getValues(): DataTypeMap[D];
    getValuesAsync(): Promise<DataTypeMap[D]>;
    data(): Promise<DataTypeMap[D]>;
    dataSync(): DataTypeMap[D];
    dispose(): void;
    equals(t: NDArray<D, R>): boolean;
    static rand<D extends DataType, R extends Rank>(shape: number[], randFunction: () => number, dtype?: D): RankMap<D>[R];
    static randNormal<D extends keyof RandNormalDataTypes, R extends Rank>(shape: number[], mean?: number, stdDev?: number, dtype?: D, seed?: number): RankMap<D>[R];
    static randTruncatedNormal<D extends keyof RandNormalDataTypes, R extends Rank>(shape: number[], mean?: number, stdDev?: number, dtype?: D, seed?: number): RankMap<D>[R];
    static randUniform<D extends DataType, R extends Rank>(shape: number[], a: number, b: number, dtype?: D): RankMap<D>[R];
    private isDisposed;
    private throwIfDisposed();
}
export declare class Scalar<D extends DataType = DataType> extends NDArray<D, '0'> {
    static new<D extends DataType = 'float32'>(value: number | boolean, dtype?: D): Scalar<D>;
    get(): number;
    val(): Promise<number>;
    add(value: number): void;
    asType<D2 extends DataType>(dtype: D2): Scalar<D2>;
    locToIndex(loc: number[]): number;
    indexToLoc(index: number): number[];
}
export declare class Array1D<D extends DataType = DataType> extends NDArray<D, '1'> {
    static new<D extends DataType = 'float32'>(values: DataTypeMap[D] | number[] | boolean[], dtype?: D): Array1D<D>;
    get(i: number): number;
    val(i: number): Promise<number>;
    add(value: number, i: number): void;
    locToIndex(loc: [number]): number;
    indexToLoc(index: number): [number];
    asType<D2 extends DataType>(dtype: D2): Array1D<D2>;
    static ones<D extends DataType = DataType>(shape: [number], dtype?: D): Array1D<D>;
    static zeros<D extends DataType = DataType>(shape: [number], dtype?: D): Array1D<D>;
    static randNormal<D extends keyof RandNormalDataTypes>(shape: [number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array1D<D>;
    static randTruncatedNormal<D extends keyof RandNormalDataTypes>(shape: [number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array1D<D>;
    static randUniform<D extends DataType>(shape: [number], a: number, b: number, dtype?: D): Array1D<D>;
}
export declare class Array2D<D extends DataType = DataType> extends NDArray<D, '2'> {
    constructor(shape: [number, number], dtype: D, values?: DataTypeMap[D], dataId?: number, math?: NDArrayMath);
    static new<D extends DataType = 'float32'>(shape: [number, number], values: DataTypeMap[D] | number[] | number[][] | boolean[] | boolean[][], dtype?: D): Array2D<D>;
    get(i: number, j: number): number;
    add(value: number, i: number, j: number): void;
    val(i: number, j: number): Promise<number>;
    locToIndex(locs: [number, number]): number;
    indexToLoc(index: number): [number, number];
    asType<D2 extends DataType>(dtype: D2): Array2D<D2>;
    static ones<D extends DataType = DataType>(shape: [number, number], dtype?: D): Array2D<D>;
    static zeros<D extends DataType = DataType>(shape: [number, number], dtype?: D): Array2D<D>;
    static randNormal<D extends keyof RandNormalDataTypes>(shape: [number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array2D<D>;
    static randTruncatedNormal<D extends keyof RandNormalDataTypes>(shape: [number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array2D<D>;
    static randUniform<D extends DataType>(shape: [number, number], a: number, b: number, dtype?: D): Array2D<D>;
}
export declare class Array3D<D extends DataType = DataType> extends NDArray<D, '3'> {
    constructor(shape: [number, number, number], dtype: D, values?: DataTypeMap[D], dataId?: number, math?: NDArrayMath);
    static new<D extends DataType = 'float32'>(shape: [number, number, number], values: DataTypeMap[D] | number[] | number[][][] | boolean[] | boolean[][][], dtype?: D): Array3D<D>;
    get(i: number, j: number, k: number): number;
    val(i: number, j: number, k: number): Promise<number>;
    add(value: number, i: number, j: number, k: number): void;
    locToIndex(locs: [number, number, number]): number;
    indexToLoc(index: number): [number, number, number];
    static ones<D extends DataType = DataType>(shape: [number, number, number], dtype?: D): Array3D<D>;
    asType<D2 extends DataType>(dtype: D2): Array3D<D2>;
    static zeros<D extends DataType = DataType>(shape: [number, number, number], dtype?: D): Array3D<D>;
    static randNormal<D extends keyof RandNormalDataTypes>(shape: [number, number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array3D<D>;
    static randTruncatedNormal<D extends keyof RandNormalDataTypes>(shape: [number, number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array3D<D>;
    static randUniform<D extends DataType>(shape: [number, number, number], a: number, b: number, dtype?: D): Array3D<D>;
}
export declare class Array4D<D extends DataType = DataType> extends NDArray<D, '4'> {
    constructor(shape: [number, number, number, number], dtype: D, values?: DataTypeMap[D], dataId?: number, math?: NDArrayMath);
    static new<D extends DataType = 'float32'>(shape: [number, number, number, number], values: DataTypeMap[D] | number[] | number[][][][] | boolean[] | boolean[][][][], dtype?: D): Array4D<D>;
    get(i: number, j: number, k: number, l: number): number;
    val(i: number, j: number, k: number, l: number): Promise<number>;
    add(value: number, i: number, j: number, k: number, l: number): void;
    locToIndex(locs: [number, number, number, number]): number;
    indexToLoc(index: number): [number, number, number, number];
    asType<D2 extends DataType>(dtype: D2): Array4D<D2>;
    static ones<D extends DataType = DataType>(shape: [number, number, number, number], dtype?: D): Array4D<D>;
    static zeros<D extends DataType = DataType>(shape: [number, number, number, number], dtype?: D): Array4D<D>;
    static randNormal<D extends keyof RandNormalDataTypes>(shape: [number, number, number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array4D<D>;
    static randTruncatedNormal<D extends keyof RandNormalDataTypes>(shape: [number, number, number, number], mean?: number, stdDev?: number, dtype?: D, seed?: number): Array4D<D>;
    static randUniform<D extends DataType>(shape: [number, number, number, number], a: number, b: number, dtype?: D): Array4D<D>;
}
export declare class Variable<D extends DataType = DataType, R extends Rank = Rank> extends NDArray<D, R> {
    trainable: boolean;
    private static nextVarId;
    name: string;
    private constructor();
    static variable<D extends DataType, R extends Rank>(initialValue: NDArray<D, R>, trainable?: boolean, name?: string, dtype?: D): Variable<D, R>;
    assign(newValue: NDArray<D, R>): void;
}
declare const variable: typeof Variable.variable;
export { variable };
