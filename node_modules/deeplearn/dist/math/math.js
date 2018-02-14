"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var util = require("../util");
var axis_util = require("./axis_util");
var backend_engine_1 = require("./backends/backend_engine");
var matmul_1 = require("./backends/types/matmul");
var broadcast_util = require("./broadcast_util");
var concat_util = require("./concat_util");
var conv_util = require("./conv_util");
var ndarray_1 = require("./ndarray");
var slice_util = require("./slice_util");
var NDArrayMath = (function () {
    function NDArrayMath(backend, safeMode) {
        this.registeredArrays = new Map();
        this.customBackend = false;
        this.registeredVariables = {};
        if (typeof backend === 'string') {
            this.backend = environment_1.ENV.getBackend(backend);
        }
        else {
            this.customBackend = true;
            this.backend = backend;
        }
        this.backendEngine = new backend_engine_1.BackendEngine(this.backend, safeMode);
    }
    NDArrayMath.prototype.time = function (query) {
        return this.backend.time(query);
    };
    NDArrayMath.prototype.getNumArrays = function () {
        return this.registeredArrays.size;
    };
    NDArrayMath.prototype.register = function (a) {
        var refCount = this.registeredArrays.has(a.dataId) ?
            this.registeredArrays.get(a.dataId) :
            0;
        if (refCount === 0) {
            this.backend.register(a.dataId, a.shape, a.dtype);
        }
        this.registeredArrays.set(a.dataId, refCount + 1);
        if (!(a instanceof ndarray_1.Variable)) {
            this.backendEngine.track(a);
        }
    };
    NDArrayMath.prototype.registerVariable = function (v) {
        if (this.registeredVariables[v.name] != null) {
            throw new Error("Variable with name " + v.name + " was already registered");
        }
        this.registeredVariables[v.name] = v;
    };
    NDArrayMath.prototype.writePixels = function (dataId, pixels, numChannels) {
        this.backend.writePixels(dataId, pixels, numChannels);
    };
    NDArrayMath.prototype.write = function (dataId, values) {
        this.backend.write(dataId, values);
    };
    NDArrayMath.prototype.readSync = function (dataId) {
        return this.backend.readSync(dataId);
    };
    NDArrayMath.prototype.read = function (dataId) {
        return this.backend.read(dataId);
    };
    NDArrayMath.prototype.enableDebugMode = function () {
        this.backendEngine.enableDebugMode();
        console.warn('Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
    };
    NDArrayMath.prototype.scope = function (scopeFn) {
        var gradientsMode = false;
        return this.backendEngine.scope('scope', scopeFn, gradientsMode);
    };
    NDArrayMath.prototype.gradientsScope = function (scopeFn) {
        var gradientsMode = true;
        return this.backendEngine.scope('gradientsScope', scopeFn, gradientsMode);
    };
    NDArrayMath.prototype.startScope = function () {
        var gradientsMode = false;
        this.backendEngine.startScope(gradientsMode);
    };
    NDArrayMath.prototype.endScope = function (result) {
        var gradientsMode = false;
        this.backendEngine.endScope(result, gradientsMode);
    };
    NDArrayMath.prototype.keep = function (result) {
        return this.backendEngine.keep(result);
    };
    NDArrayMath.prototype.track = function (result) {
        return result;
    };
    NDArrayMath.prototype.dispose = function () {
        if (this.customBackend) {
            this.backend.dispose();
        }
    };
    NDArrayMath.prototype.matMul = function (a, b, aOrientation, bOrientation) {
        var _this = this;
        if (aOrientation === void 0) { aOrientation = matmul_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = matmul_1.MatrixOrientation.REGULAR; }
        var innerShapeA = (aOrientation === matmul_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var innerShapeB = (bOrientation === matmul_1.MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];
        util.assert(a.rank === 2 && b.rank === 2, "Error in matMul: inputs must be rank 2, got ranks " + a.rank +
            (" and " + b.rank + "."));
        util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
            (innerShapeB + ") of NDArrays with shapes " + a.shape + " and ") +
            (b.shape + " and orientations " + matmul_1.MatrixOrientation[aOrientation]) +
            (" and " + matmul_1.MatrixOrientation[bOrientation] + " must match."));
        return this.backendEngine.executeKernel('MatMul', { inputs: { a: a, b: b }, args: { aOrientation: aOrientation, bOrientation: bOrientation } }, function (dy, y) {
            if (aOrientation === matmul_1.MatrixOrientation.TRANSPOSED ||
                bOrientation === matmul_1.MatrixOrientation.TRANSPOSED) {
                throw new Error("Backprop for transposed MatMul not yet implemented.");
            }
            return {
                a: function () { return _this.matMul(dy, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED); },
                b: function () { return _this.matMul(a, dy, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR); }
            };
        });
    };
    NDArrayMath.prototype.executeOp = function (name, f) {
        return f();
    };
    NDArrayMath.prototype.vectorTimesMatrix = function (v, matrix) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: first input must be rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: second input must be rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[0], "Error in vectorTimesMatrix: size of vector (" + v.size + ") " +
            ("must match first dimension of matrix (" + matrix.shape[0] + ")"));
        return this.matMul(v.as2D(1, -1), matrix).as1D();
    };
    NDArrayMath.prototype.matrixTimesVector = function (matrix, v) {
        util.assert(v.rank === 1, "Error in matrixTimesVector: second input must rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in matrixTimesVector: first input must be a rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[1], "Error in matrixTimesVector: size of first rank 1 input " + v.size + " " +
            "must match inner dimension of second rank 2 input, but got " +
            ("shape " + matrix.shape + "."));
        return this.matMul(matrix, v.as2D(-1, 1)).as1D();
    };
    NDArrayMath.prototype.dotProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in dotProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        util.assert(v1.size === v2.size, "Error in dotProduct: size of inputs (" + v1.size + ") and (" +
            (v2.size + ") must match."));
        return this.matMul(v1.as2D(1, -1), v2.as2D(-1, 1)).asScalar();
    };
    NDArrayMath.prototype.outerProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        return this.matMul(v1.as2D(-1, 1), v2.as2D(1, -1));
    };
    NDArrayMath.prototype.clone = function (x) {
        return this.backendEngine.executeKernel('Clone', { inputs: { x: x } });
    };
    NDArrayMath.prototype.reshape = function (x, newShape) {
        newShape = util.inferFromImplicitShape(newShape, x.size);
        util.assert(x.size === util.sizeFromShape(newShape), 'new shape and old shape must have the same number of elements.');
        var grad = function (dy, y) {
            return { x: function () { return dy.reshape(x.shape); } };
        };
        return this.backendEngine.executeKernel('Reshape', { inputs: { x: x }, args: { newShape: newShape } }, grad);
    };
    NDArrayMath.prototype.cast = function (x, newDType) {
        var grad = function (dy, y) {
            return { x: function () { return dy.reshape(dy.shape); } };
        };
        return this.backendEngine.executeKernel('Cast', { inputs: { x: x }, args: { newDType: newDType } }, grad);
    };
    NDArrayMath.prototype.slice1D = function (x, begin, size) {
        slice_util.assertParamsValid(x, [begin], [size]);
        return this.backendEngine.executeKernel('Slice1D', { inputs: { x: x }, args: { begin: begin, size: size } });
    };
    NDArrayMath.prototype.slice2D = function (x, begin, size) {
        slice_util.assertParamsValid(x, begin, size);
        return this.backendEngine.executeKernel('Slice2D', { inputs: { x: x }, args: { begin: begin, size: size } });
    };
    NDArrayMath.prototype.slice3D = function (x, begin, size) {
        slice_util.assertParamsValid(x, begin, size);
        return this.backendEngine.executeKernel('Slice3D', { inputs: { x: x }, args: { begin: begin, size: size } });
    };
    NDArrayMath.prototype.slice4D = function (x, begin, size) {
        slice_util.assertParamsValid(x, begin, size);
        return this.backendEngine.executeKernel('Slice4D', { inputs: { x: x }, args: { begin: begin, size: size } });
    };
    NDArrayMath.prototype.reverse1D = function (x) {
        util.assert(x.rank === 1, "Error in reverse1D: x must be rank 1 but got\n             rank " + x.rank + ".");
        var input4D = x.as4D(1, 1, 1, x.shape[0]);
        var res = this.reverse4D(input4D, [3]);
        return res.as1D();
    };
    NDArrayMath.prototype.reverse2D = function (x, axis) {
        util.assert(x.rank === 2, "Error in reverse2D: x must be rank 2 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(function (a) { return a + 2; });
        var input4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
        var res = this.reverse4D(input4D, axisCleaned);
        return res.as2D(res.shape[2], res.shape[3]);
    };
    NDArrayMath.prototype.reverse3D = function (x, axis) {
        util.assert(x.rank === 3, "Error in reverse3D: x must be rank 3 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(function (a) { return a + 1; });
        var input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        var res = this.reverse4D(input4D, axisCleaned);
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    };
    NDArrayMath.prototype.reverse4D = function (x, axis) {
        util.assert(x.rank === 4, "Error in reverse4D: x must be rank 4 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape);
        return this.backendEngine.executeKernel('Reverse4D', { inputs: { x: x }, args: { axis: axisCleaned } });
    };
    NDArrayMath.prototype.concat1D = function (a, b) {
        concat_util.assertParams(a.shape, b.shape, 0);
        return this.backendEngine.executeKernel('Concat1D', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.concat2D = function (a, b, axis) {
        concat_util.assertParams(a.shape, b.shape, axis);
        return this.backendEngine.executeKernel('Concat2D', { inputs: { a: a, b: b }, args: { axis: axis } });
    };
    NDArrayMath.prototype.concat3D = function (a, b, axis) {
        var _this = this;
        concat_util.assertParams(a.shape, b.shape, axis);
        var gradients = function (dy, y) {
            var _a = concat_util.computeGradientSliceShapes3D(a.shape, y.shape, axis), x1Begin = _a.x1Begin, x1Size = _a.x1Size, x2Begin = _a.x2Begin, x2Size = _a.x2Size;
            return {
                a: function () { return _this.slice3D(dy, x1Begin, x1Size); },
                b: function () { return _this.slice3D(dy, x2Begin, x2Size); }
            };
        };
        return this.backendEngine.executeKernel('Concat3D', { inputs: { a: a, b: b }, args: { axis: axis } }, gradients);
    };
    NDArrayMath.prototype.concat4D = function (a, b, axis) {
        concat_util.assertParams(a.shape, b.shape, axis);
        return this.backendEngine.executeKernel('Concat4D', { inputs: { a: a, b: b }, args: { axis: axis } });
    };
    NDArrayMath.prototype.logSumExp = function (input, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, input.shape);
        return this.executeOp('logSumExp', function () {
            var xMax = _this.max(input, axes, true);
            var a = _this.subtract(input, xMax);
            var b = _this.exp(a);
            var c = _this.sum(b, axes);
            var d = _this.log(c);
            var res = _this.add(xMax.reshape(d.shape), d);
            if (keepDims) {
                var newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
                return res.reshape(newShape);
            }
            return res;
        });
    };
    NDArrayMath.prototype.sum = function (x, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        return this.executeOp('sum', function () {
            return _this.customGradient(function () {
                var permutation = axis_util.getAxesPermutation(axes, x.rank);
                var reductionAxes = axes;
                var permutedX = x;
                if (permutation != null) {
                    permutedX = _this.transpose(x, permutation);
                    reductionAxes =
                        axis_util.getInnerMostAxes(reductionAxes.length, x.rank);
                }
                var value = _this.backendEngine.executeKernel('Sum', { inputs: { x: permutedX }, args: { axes: reductionAxes } });
                if (keepDims) {
                    var newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
                    value = value.reshape(newShape);
                }
                var gradients = function (dy) {
                    var expandedDyShape = x.shape.slice();
                    axes.forEach(function (axis) {
                        expandedDyShape[axis] = 1;
                    });
                    var expandedDy = dy.reshape(expandedDyShape);
                    var derX = function () {
                        return _this.multiply(expandedDy, ndarray_1.NDArray.ones(x.shape, 'float32'));
                    };
                    return { x: derX };
                };
                return { value: value, gradients: gradients };
            }, { x: x }, 'sum');
        });
    };
    NDArrayMath.prototype.mean = function (x, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var shapes = axis_util.computeOutAndReduceShapes(x.shape, axes);
        var reduceShape = shapes[1];
        var reduceSize = util.sizeFromShape(reduceShape);
        return this.executeOp('mean', function () {
            return _this.customGradient(function () {
                var reduceSizeScalar = ndarray_1.Scalar.new(reduceSize);
                var res = _this.divide(x, reduceSizeScalar);
                var value = _this.sum(res, axis, keepDims);
                var gradients = function (dy) {
                    var expandedDyShape = x.shape.slice();
                    axes.forEach(function (axis) {
                        expandedDyShape[axis] = 1;
                    });
                    var expandedDy = dy.reshape(expandedDyShape);
                    var derX = function () { return _this.divide(_this.multiply(expandedDy, ndarray_1.NDArray.ones(x.shape, 'float32')), reduceSizeScalar); };
                    return { x: derX };
                };
                return { value: value, gradients: gradients };
            }, { x: x }, 'mean');
        });
    };
    NDArrayMath.prototype.argMin = function (x, axis) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        return this.executeOp('argMin', function () {
            if (permutedAxes != null) {
                x = _this.transpose(x, permutedAxes);
                axes = axis_util.getInnerMostAxes(axes.length, x.rank);
            }
            return _this.backendEngine.executeKernel('ArgMin', { inputs: { x: x }, args: { axes: axes } });
        });
    };
    NDArrayMath.prototype.argMax = function (x, axis) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        return this.executeOp('argMax', function () {
            if (permutedAxes != null) {
                x = _this.transpose(x, permutedAxes);
                axes = axis_util.getInnerMostAxes(axes.length, x.rank);
            }
            return _this.backendEngine.executeKernel('ArgMax', { inputs: { x: x }, args: { axes: axes } });
        });
    };
    NDArrayMath.prototype.argMaxEquals = function (x1, x2) {
        var _this = this;
        util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
        return this.executeOp('argMaxEquals', function () { return _this.scope(function () {
            return _this.equal(_this.argMax(x1), _this.argMax(x2));
        }); });
    };
    NDArrayMath.prototype.equal = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('Equal', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.equalStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
        return this.equal(a, b);
    };
    NDArrayMath.prototype.notEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('NotEqual', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.notEqualStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in notEqualStrict: ');
        return this.notEqual(a, b);
    };
    NDArrayMath.prototype.lessEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('LessEqual', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.greater = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('Greater', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.greaterEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('GreaterEqual', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.logicalOr = function (a, b) {
        util.assert(a.dtype === 'bool' || b.dtype === 'bool', 'Error Array must be of type bool.');
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('LogicalOr', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.topK = function (x, k) {
        var _this = this;
        util.assert(k <= x.size, "Error in topK: k value (" + k + ") must be less than size of input " +
            ("ndarray, got shape " + x.shape + "."));
        var values;
        var indices;
        this.executeOp('topK', function () {
            values = _this.backendEngine.executeKernel('TopKValues', { inputs: { x: x }, args: { k: k } });
            indices = _this.backendEngine.executeKernel('TopKIndices', { inputs: { x: x }, args: { k: k } });
            return values;
        });
        var result = { values: values, indices: indices };
        return result;
    };
    NDArrayMath.prototype.min = function (x, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var origAxes = axis_util.parseAxisParam(axis, x.shape);
        var axes = origAxes;
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        return this.executeOp('min', function () {
            if (permutedAxes != null) {
                x = _this.transpose(x, permutedAxes);
                axes = axis_util.getInnerMostAxes(axes.length, x.rank);
            }
            var res = _this.backendEngine.executeKernel('Min', { inputs: { x: x }, args: { axes: axes } });
            if (keepDims) {
                var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
                return res.reshape(newShape);
            }
            return res;
        });
    };
    NDArrayMath.prototype.minimum = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('Minimum', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.max = function (x, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var origAxes = axis_util.parseAxisParam(axis, x.shape);
        var axes = origAxes;
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        return this.executeOp('max', function () {
            if (permutedAxes != null) {
                x = _this.transpose(x, permutedAxes);
                axes = axis_util.getInnerMostAxes(axes.length, x.rank);
            }
            var res = _this.backendEngine.executeKernel('Max', { inputs: { x: x }, args: { axes: axes } });
            if (keepDims) {
                var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
                return res.reshape(newShape);
            }
            return res;
        });
    };
    NDArrayMath.prototype.maximum = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return this.backendEngine.executeKernel('Maximum', { inputs: { a: a, b: b } });
    };
    NDArrayMath.prototype.softmax = function (logits, dim) {
        var _this = this;
        if (dim === void 0) { dim = -1; }
        if (dim === -1) {
            dim = logits.rank - 1;
        }
        if (dim !== logits.rank - 1) {
            throw Error('Softmax along a non-last dimension is not yet supported. ' +
                ("Logits was rank " + logits.rank + " and dim was " + dim));
        }
        var gradients = function (dy, y) {
            return {
                logits: function () {
                    var dyTimesY = _this.multiply(dy, y);
                    var keepDims = true;
                    return _this.subtract(dyTimesY, _this.multiply(_this.sum(dyTimesY, [dim], keepDims), y));
                }
            };
        };
        return this.executeOp('softmax', function () {
            return _this.customGradient(function () {
                var keepDims = true;
                var lse = _this.logSumExp(logits, [dim], keepDims);
                var logResult = _this.subtract(logits.asType('float32'), lse);
                var value = _this.exp(logResult);
                return { value: value, gradients: gradients };
            }, { logits: logits }, 'softmax');
        });
    };
    NDArrayMath.prototype.softmaxCrossEntropyWithLogits = function (labels, logits, dim) {
        var _this = this;
        if (dim === void 0) { dim = -1; }
        util.assertShapesMatch(labels.shape, logits.shape, 'Error in softmaxCrossEntropyWithLogits: ');
        if (dim === -1) {
            dim = logits.rank - 1;
        }
        if (dim !== logits.rank - 1) {
            throw Error("Softmax cross entropy along a non-last dimension is not yet " +
                ("supported. Labels / logits was rank " + logits.rank + " ") +
                ("and dim was " + dim));
        }
        return this.executeOp('softmaxCrossEntropyWithLogits', function () {
            return _this.customGradient(function () {
                var softmaxLogits = _this.softmax(logits, dim);
                var yPlusEps = _this.add(ndarray_1.Scalar.new(1e-5), softmaxLogits);
                var logOutput = _this.log(yPlusEps);
                var tarLogOutput = _this.multiply(labels, logOutput);
                var costVector = _this.neg(tarLogOutput);
                var value = _this.sum(costVector, [dim]);
                var gradients = function (dy, y) {
                    var dyShape = axis_util.expandShapeToKeepDim(dy.shape, [dim]);
                    return {
                        logits: function () { return _this.multiply(dy.reshape(dyShape), _this.subtract(softmaxLogits, labels.asType('float32'))); },
                        labels: function () { return _this.multiply(dy.reshape(dyShape), _this.subtract(labels, softmaxLogits)); }
                    };
                };
                return { value: value, gradients: gradients };
            }, { labels: labels, logits: logits }, 'softmaxCrossEntropyWithLogits');
        });
    };
    NDArrayMath.prototype.switchDim = function (a, newDim) {
        return this.transpose(a, newDim);
    };
    NDArrayMath.prototype.tile = function (x, reps) {
        util.assert(x.rank === reps.length, "Error in transpose: rank of input " + x.rank + " " +
            ("must match length of reps " + reps + "."));
        return this.backendEngine.executeKernel('Tile', { inputs: { x: x }, args: { reps: reps } });
    };
    NDArrayMath.prototype.pad1D = function (x, paddings, constantValue) {
        if (constantValue === void 0) { constantValue = 0; }
        util.assert(paddings.length === 2, 'Invalid number of paddings. Must be length of 2.');
        return this.backendEngine.executeKernel('Pad1D', { inputs: { x: x }, args: { paddings: paddings, constantValue: constantValue } });
    };
    NDArrayMath.prototype.pad2D = function (x, paddings, constantValue) {
        if (constantValue === void 0) { constantValue = 0; }
        util.assert(paddings.length === 2 && paddings[0].length === 2 &&
            paddings[1].length === 2, 'Invalid number of paddings. Must be length of 2 each.');
        return this.backendEngine.executeKernel('Pad2D', { inputs: { x: x }, args: { paddings: paddings, constantValue: constantValue } });
    };
    NDArrayMath.prototype.transpose = function (x, perm) {
        var _this = this;
        if (perm == null) {
            perm = x.shape.map(function (s, i) { return i; }).reverse();
        }
        var der = function (dy) {
            var undoPerm = axis_util.getUndoAxesPermutation(perm);
            var derX = function () { return _this.transpose(dy, undoPerm); };
            return { x: derX };
        };
        util.assert(x.rank === perm.length, "Error in transpose: rank of input " + x.rank + " " +
            ("must match length of perm " + perm + "."));
        return this.backendEngine.executeKernel('Transpose', { inputs: { x: x }, args: { perm: perm } }, der);
    };
    NDArrayMath.prototype.scalarPlusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarPlusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.add(c, a);
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.subtract(c, a);
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.subtract(a, c);
    };
    NDArrayMath.prototype.neg = function (x) {
        return this.backendEngine.executeKernel('Neg', { inputs: { x: x } });
    };
    NDArrayMath.prototype.add = function (a, b) {
        var _this = this;
        util.assertTypesMatch(a, b);
        var outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var der = function (dy, y) {
            var derA = function () {
                var res = dy;
                var reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
                if (reduceAxes.length > 0) {
                    res = _this.sum(res, reduceAxes);
                }
                return res.reshape(a.shape);
            };
            var derB = function () {
                var res = dy;
                var reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
                if (reduceAxes.length > 0) {
                    res = _this.sum(res, reduceAxes);
                }
                return res.reshape(b.shape);
            };
            return { a: derA, b: derB };
        };
        return this.backendEngine.executeKernel('Add', { inputs: { a: a, b: b } }, der);
    };
    NDArrayMath.prototype.addStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
        return this.add(a, b);
    };
    NDArrayMath.prototype.subtract = function (a, b) {
        var _this = this;
        util.assertTypesMatch(a, b);
        var outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var der = function (dy, y) {
            var derA = function () {
                var res = dy;
                var reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
                if (reduceAxes.length > 0) {
                    res = _this.sum(res, reduceAxes);
                }
                return res.reshape(a.shape);
            };
            var derB = function () {
                var res = dy;
                var reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
                if (reduceAxes.length > 0) {
                    res = _this.sum(res, reduceAxes);
                }
                return _this.neg(res).reshape(b.shape);
            };
            return { a: derA, b: derB };
        };
        return this.backendEngine.executeKernel('Sub', { inputs: { a: a, b: b } }, der);
    };
    NDArrayMath.prototype.pow = function (a, b) {
        var _this = this;
        util.assert(b.dtype === 'int32', 'only supports int32 data type for the exponent parameter.');
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var gradient = function (dy, y) {
            if (!util.arraysEqual(a.shape, b.shape)) {
                throw new Error("Gradient of pow not yet supported for broadcasted shapes.");
            }
            var derA = function () {
                return _this.scope(function () {
                    return _this.multiply(dy, _this.multiply(b.asType(a.dtype), _this.pow(a, _this.subtract(b, ndarray_1.Scalar.new(1, 'int32')))));
                });
            };
            var derB = function () {
                throw new Error("Backprop through exponent of math.pow not " +
                    "implemented yet.");
            };
            return { a: derA, b: derB };
        };
        return this.backendEngine.executeKernel('Pow', { inputs: { a: a, b: b } }, gradient);
    };
    NDArrayMath.prototype.powStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in powStrict: ');
        return this.pow(a, b);
    };
    NDArrayMath.prototype.sub = function (a, b) {
        return this.subtract(a, b);
    };
    NDArrayMath.prototype.subStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
        return this.subtract(a, b);
    };
    NDArrayMath.prototype.multiply = function (a, b) {
        var _this = this;
        util.assertTypesMatch(a, b);
        var outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var der = function (dy, y) {
            var derA = function () {
                var res = _this.multiply(dy, b.asType('float32'));
                var reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
                if (reduceAxes.length > 0) {
                    return _this.sum(res, reduceAxes).reshape(a.shape);
                }
                return res;
            };
            var derB = function () {
                var res = _this.multiply(dy, a.asType('float32'));
                var reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
                if (reduceAxes.length > 0) {
                    return _this.sum(res, reduceAxes).reshape(b.shape);
                }
                return res;
            };
            return { a: derA, b: derB };
        };
        return this.backendEngine.executeKernel('Mul', { inputs: { a: a, b: b } }, der);
    };
    NDArrayMath.prototype.elementWiseMul = function (a, b) {
        return this.multiplyStrict(a, b);
    };
    NDArrayMath.prototype.multiplyStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.divide = function (a, b) {
        var _this = this;
        var outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var der = function (dy, y) {
            var derA = function () {
                var res = _this.divide(dy, b.asType('float32'));
                var reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
                if (reduceAxes.length > 0) {
                    return _this.sum(res, reduceAxes).reshape(a.shape);
                }
                return res;
            };
            var derB = function () {
                var res = _this.multiply(dy, a.asType('float32'));
                var reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
                if (reduceAxes.length > 0) {
                    res = _this.sum(res, reduceAxes).reshape(b.shape);
                }
                return _this.neg(_this.divide(res, _this.square(b)));
            };
            return { a: derA, b: derB };
        };
        return this.backendEngine.executeKernel('Div', { inputs: { a: a, b: b } }, der);
    };
    NDArrayMath.prototype.divideStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
        return this.divide(a, b);
    };
    NDArrayMath.prototype.scalarDividedByArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarDividedByArray: first argument must be rank 0, but " +
            ("got NDArray of rank " + c.rank + "."));
        return this.divide(c, a);
    };
    NDArrayMath.prototype.arrayDividedByScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: second argument must be rank 0, " +
            ("but got NDArray of rank " + c.rank + "."));
        return this.divide(a, c);
    };
    NDArrayMath.prototype.ceil = function (x) {
        return this.backendEngine.executeKernel('Ceil', { inputs: { x: x } });
    };
    NDArrayMath.prototype.floor = function (x) {
        return this.backendEngine.executeKernel('Floor', { inputs: { x: x } });
    };
    NDArrayMath.prototype.exp = function (x) {
        return this.backendEngine.executeKernel('Exp', { inputs: { x: x } });
    };
    NDArrayMath.prototype.log = function (x) {
        return this.backendEngine.executeKernel('Log', { inputs: { x: x } });
    };
    NDArrayMath.prototype.sqrt = function (x) {
        return this.backendEngine.executeKernel('Sqrt', { inputs: { x: x } });
    };
    NDArrayMath.prototype.square = function (x) {
        var _this = this;
        return this.backendEngine.executeKernel('Square', { inputs: { x: x } }, function (dy, y) {
            return {
                x: function () { return _this.multiply(dy, _this.multiply(x.asType('float32'), ndarray_1.Scalar.new(2))); }
            };
        });
    };
    NDArrayMath.prototype.abs = function (x) {
        return this.backendEngine.executeKernel('Abs', { inputs: { x: x } });
    };
    NDArrayMath.prototype.clip = function (x, min, max) {
        util.assert((min <= max), "Error in clip: min (" + min + ") must be" +
            ("less than or equal to max (" + max + ")."));
        return this.backendEngine.executeKernel('Clip', { inputs: { x: x }, args: { min: min, max: max } });
    };
    NDArrayMath.prototype.relu = function (x) {
        var _this = this;
        return this.backendEngine.executeKernel('Relu', { inputs: { x: x } }, function (dy, y) {
            return {
                x: function () { return _this.multiply(dy, _this.step(x).asType('float32')); }
            };
        });
    };
    NDArrayMath.prototype.elu = function (x) {
        return this.backendEngine.executeKernel('Elu', { inputs: { x: x } });
    };
    NDArrayMath.prototype.eluDer = function (x) {
        return this.backendEngine.executeKernel('EluDer', { inputs: { x: x } });
    };
    NDArrayMath.prototype.selu = function (x) {
        return this.backendEngine.executeKernel('Selu', { inputs: { x: x } });
    };
    NDArrayMath.prototype.leakyRelu = function (x, alpha) {
        if (alpha === void 0) { alpha = 0.2; }
        return this.backendEngine.executeKernel('LeakyRelu', { inputs: { x: x }, args: { alpha: alpha } });
    };
    NDArrayMath.prototype.prelu = function (x, alpha) {
        return this.backendEngine.executeKernel('PReLU', { inputs: { x: x, alpha: alpha } });
    };
    NDArrayMath.prototype.preluDer = function (x, alpha) {
        return this.backendEngine.executeKernel('PReLUDer', { inputs: { x: x, alpha: alpha } });
    };
    NDArrayMath.prototype.sigmoid = function (x) {
        return this.backendEngine.executeKernel('Sigmoid', { inputs: { x: x } });
    };
    NDArrayMath.prototype.sin = function (x) {
        return this.backendEngine.executeKernel('Sin', { inputs: { x: x } });
    };
    NDArrayMath.prototype.cos = function (x) {
        return this.backendEngine.executeKernel('Cos', { inputs: { x: x } });
    };
    NDArrayMath.prototype.tan = function (x) {
        return this.backendEngine.executeKernel('Tan', { inputs: { x: x } });
    };
    NDArrayMath.prototype.asin = function (x) {
        return this.backendEngine.executeKernel('Asin', { inputs: { x: x } });
    };
    NDArrayMath.prototype.acos = function (x) {
        return this.backendEngine.executeKernel('Acos', { inputs: { x: x } });
    };
    NDArrayMath.prototype.atan = function (x) {
        return this.backendEngine.executeKernel('Atan', { inputs: { x: x } });
    };
    NDArrayMath.prototype.sinh = function (x) {
        return this.backendEngine.executeKernel('Sinh', { inputs: { x: x } });
    };
    NDArrayMath.prototype.cosh = function (x) {
        return this.backendEngine.executeKernel('Cosh', { inputs: { x: x } });
    };
    NDArrayMath.prototype.tanh = function (x) {
        return this.backendEngine.executeKernel('Tanh', { inputs: { x: x } });
    };
    NDArrayMath.prototype.step = function (x, alpha) {
        if (alpha === void 0) { alpha = 0.0; }
        return this.backendEngine.executeKernel('Step', { inputs: { x: x }, args: { alpha: alpha } });
    };
    NDArrayMath.prototype.scaledArrayAdd = function (c1, a, c2, b) {
        var _this = this;
        util.assert(c1.size === 1, "Error in scaledArrayAdd: first argument must rank 0, but got " +
            (" rank " + c1.rank + "."));
        util.assert(c2.size === 1, "Error in scaledArrayAdd: third argument must be rank 0, but got " +
            ("NDArray of rank " + c2.rank + "."));
        util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');
        return this.executeOp('scaledArrayAdd', function () {
            return _this.scope(function () {
                return _this.add(_this.multiply(c1, a), _this.multiply(c2, b));
            });
        });
    };
    NDArrayMath.prototype.scalarTimesArray = function (c, a) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: first argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.multiply(c, a);
    };
    NDArrayMath.prototype.elementWiseMulBroadcast = function (a, b) {
        util.assert(a.rank === 2, "Error in elementWiseMulBroadcast: first argument must be " +
            ("rank 2, but got rank " + a.rank + "."));
        util.assert(b.rank === 2, "Error in elementWiseMulBroadcast: second argument must be " +
            ("rank 2, but got rank " + b.rank + "."));
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.conv1d = function (input, filter, bias, stride, pad, dimRoundingMode) {
        var _this = this;
        var input3D = input;
        var reshapedTo3D = false;
        if (input.rank === 2) {
            reshapedTo3D = true;
            input3D = input.as3D(1, input.shape[0], input.shape[1]);
        }
        util.assert(input3D.rank === 3, "Error in conv1d: input must be rank 3, but got rank " + input3D.rank + ".");
        util.assert(filter.rank === 3, "Error in conv1d: filter must be rank 3, but got rank " +
            (filter.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv1d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv1d: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        util.assert(input3D.shape[2] === filter.shape[1], "Error in conv1d: depth of input (" + input3D.shape[2] + ") must match  " +
            ("input depth for filter " + filter.shape[1] + "."));
        var filter4D = filter.as4D(1, filter.shape[0], filter.shape[1], filter.shape[2]);
        var input4D = input3D.as4D(input3D.shape[0], 1, input3D.shape[1], input3D.shape[2]);
        var strides = [1, stride];
        return this.executeOp('Conv1D', function () {
            var res = _this.conv2d(input4D, filter4D, bias, strides, pad, dimRoundingMode);
            if (reshapedTo3D) {
                return res.as2D(res.shape[2], res.shape[3]);
            }
            return res.as3D(res.shape[0], res.shape[2], res.shape[3]);
        });
    };
    NDArrayMath.prototype.conv2d = function (x, filter, bias, strides, pad, dimRoundingMode) {
        var _this = this;
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in conv2d: input must be rank 4, but got rank " + x4D.rank + ".");
        util.assert(filter.rank === 4, "Error in conv2d: filter must be rank 4, but got rank " +
            (filter.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv2d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2d: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        util.assert(x4D.shape[3] === filter.shape[2], "Error in conv2d: depth of input (" + x4D.shape[3] + ") must match  " +
            ("input depth for filter " + filter.shape[2] + "."));
        var convInfo = conv_util.computeConv2DInfo(x4D.shape, filter.shape, strides, pad, dimRoundingMode);
        return this.executeOp('Conv2D', function () {
            var gradients = function (dy, y) {
                return {
                    x: function () { return _this.conv2dDerInput(x4D.shape, dy, filter, strides, pad); },
                    filter: function () {
                        return _this.conv2dDerFilter(x4D, dy, filter.shape, strides, pad);
                    },
                    bias: function () { return _this.conv2dDerBias(dy); }
                };
            };
            var res = _this.backendEngine.executeKernel('Conv2D', { inputs: { x: x4D, filter: filter, bias: bias }, args: { convInfo: convInfo } }, gradients);
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.conv2dDerInput = function (xShape, dy, filter, strides, pad, dimRoundingMode) {
        var _this = this;
        util.assert(xShape.length === dy.rank, "Length of inShape " +
            ("(" + xShape.length + ") and rank of dy (" + dy.rank + ") must match"));
        var xShape4D = xShape;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (dy.rank === 3) {
            reshapedTo4D = true;
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
            xShape4D = [1, xShape[0], xShape[1], xShape[2]];
        }
        var inDepth = xShape4D[3];
        var outDepth = dy4D.shape[3];
        util.assert(xShape4D.length === 4, "Error in conv2dDerInput: inShape must be length 4, but got length " +
            (xShape4D.length + "."));
        util.assert(dy4D.rank === 4, "Error in conv2dDerInput: dy must be rank 4, but got " +
            ("rank " + dy4D.rank));
        util.assert(filter.rank === 4, "Error in conv2dDerInput: filter must be rank 4, but got " +
            ("rank " + filter.rank));
        util.assert(inDepth === filter.shape[2], "Error in conv2dDerInput: depth of input (" + inDepth + ") must " +
            ("match input depth for filter " + filter.shape[2] + "."));
        util.assert(outDepth === filter.shape[3], "Error in conv2dDerInput: depth of output (" + outDepth + ") must" +
            ("match output depth for filter " + filter.shape[3] + "."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2dDerInput: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(xShape4D, filter.shape, strides, pad, dimRoundingMode);
        return this.executeOp('conv2dDerInput', function () {
            var res = _this.backendEngine.executeKernel('Conv2DDerInput', { inputs: { dy: dy4D, filter: filter }, args: { convInfo: convInfo } });
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.conv2dDerBias = function (dy) {
        var dy4D = dy;
        if (dy.rank === 3) {
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        return this.backendEngine.executeKernel('Conv2DDerBias', { inputs: { dy: dy4D } });
    };
    NDArrayMath.prototype.conv2dDerFilter = function (x, dy, filterShape, strides, pad, dimRoundingMode) {
        var x4D = x;
        if (x.rank === 3) {
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        var dy4D = dy;
        if (dy4D.rank === 3) {
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in conv2dDerFilter: input must be rank 4, but got shape " +
            (x4D.shape + "."));
        util.assert(dy4D.rank === 4, "Error in conv2dDerFilter: dy must be rank 4, but got shape " +
            (dy4D.shape + "."));
        util.assert(filterShape.length === 4, "Error in conv2dDerFilter: filterShape must be length 4, but got " +
            (filterShape + "."));
        util.assert(x4D.shape[3] === filterShape[2], "Error in conv2dDerFilter: depth of input " + x4D.shape[3] + ") must " +
            ("match input depth in filter (" + filterShape[2] + "."));
        util.assert(dy4D.shape[3] === filterShape[3], "Error in conv2dDerFilter: depth of dy (" + dy4D.shape[3] + ") must " +
            ("match output depth for filter (" + filterShape[3] + ")."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2dDerFilter: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(x4D.shape, filterShape, strides, pad, dimRoundingMode);
        return this.backendEngine.executeKernel('Conv2DDerFilter', { inputs: { x: x4D, dy: dy4D }, args: { convInfo: convInfo } });
    };
    NDArrayMath.prototype.conv2dTranspose = function (x, filter, outputShape, strides, pad, dimRoundingMode) {
        return this.conv2dDerInput(outputShape, x, filter, strides, pad, dimRoundingMode);
    };
    NDArrayMath.prototype.depthwiseConv2D = function (input, filter, strides, pad, rates, dimRoundingMode) {
        var _this = this;
        if (rates === void 0) { rates = [1, 1]; }
        var input4D = input;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
        }
        util.assert(input4D.rank === 4, "Error in depthwiseConv2D: input must be rank 4, but got " +
            ("rank " + input4D.rank + "."));
        util.assert(filter.rank === 4, "Error in depthwiseConv2D: filter must be rank 4, but got rank " +
            (filter.rank + "."));
        util.assert(input4D.shape[3] === filter.shape[2], "Error in depthwiseConv2D: number of input channels " +
            ("(" + input4D.shape[3] + ") must match the inChannels dimension in ") +
            ("filter " + filter.shape[2] + "."));
        rates = rates || [1, 1];
        var _a = parseTupleParam(rates), rateHeight = _a[0], rateWidth = _a[1];
        util.assert(rateHeight === 1 && rateWidth === 1, 'Error in depthwiseConv2D: rates greater than 1 are not yet ' +
            ("supported. Got rates '" + rates + "'"));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in depthwiseConv2D: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(input4D.shape, filter.shape, strides, pad, dimRoundingMode, true);
        return this.executeOp('depthwiseConv2D', function () {
            var res = _this.backendEngine.executeKernel('DepthwiseConv2D', { inputs: { x: input4D, filter: filter }, args: { convInfo: convInfo } });
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.maxPool = function (x, filterSize, strides, pad, dimRoundingMode) {
        var _this = this;
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in maxPool: input must be rank 4 but got rank " + x4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in maxPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad, dimRoundingMode);
        var gradients = function (dy, y) {
            return { x: function () { return _this.maxPoolBackprop(dy, x4D, filterSize, strides, pad); } };
        };
        return this.executeOp('maxPool', function () {
            var res = _this.backendEngine.executeKernel('MaxPool', { inputs: { x: x4D }, args: { convInfo: convInfo } }, gradients);
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.maxPoolBackprop = function (dy, input, filterSize, strides, pad, dimRoundingMode) {
        var _this = this;
        util.assert(input.rank === dy.rank, "Rank of input (" + input.rank + ") does not match rank of dy (" + dy.rank + ")");
        var input4D = input;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(dy4D.rank === 4, "Error in maxPoolBackprop: dy must be rank 4 but got rank " +
            (dy4D.rank + "."));
        util.assert(input4D.rank === 4, "Error in maxPoolBackprop: input must be rank 4 but got rank " +
            (input4D.rank + "."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in maxPoolBackprop: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad, dimRoundingMode);
        return this.executeOp('maxPoolBackprop', function () {
            var res = _this.backendEngine.executeKernel('MaxPoolBackprop', { inputs: { dy: dy4D, x: input4D }, args: { convInfo: convInfo } });
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.minPool = function (input, filterSize, strides, pad, dimRoundingMode) {
        var _this = this;
        var input4D = input;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
        }
        util.assert(input4D.rank === 4, "Error in minPool: x must be rank 4 but got rank " + input4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in minPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad, dimRoundingMode);
        return this.executeOp('minPool', function () {
            var res = _this.backendEngine.executeKernel('MinPool', { inputs: { x: input4D }, args: { convInfo: convInfo } });
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.avgPool = function (x, filterSize, strides, pad, dimRoundingMode) {
        var _this = this;
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in avgPool: x must be rank 4 but got rank " + x4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in avgPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad);
        var gradients = function (dy, y) {
            return { x: function () { return _this.avgPoolBackprop(dy, x4D, filterSize, strides, pad); } };
        };
        return this.executeOp('avgPool', function () {
            var res = _this.backendEngine.executeKernel('AvgPool', { inputs: { x: x4D }, args: { convInfo: convInfo } }, gradients);
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.avgPoolBackprop = function (dy, input, filterSize, strides, pad) {
        var _this = this;
        util.assert(input.rank === dy.rank, "Rank of input (" + input.rank + ") does not match rank of dy (" + dy.rank + ")");
        var input4D = input;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(dy4D.rank === 4, "Error in avgPoolBackprop: dy must be rank 4 but got rank " +
            (dy4D.rank + "."));
        util.assert(input4D.rank === 4, "Error in avgPoolBackprop: input must be rank 4 but got rank " +
            (input4D.rank + "."));
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
        return this.executeOp('avgPoolBackprop', function () {
            var res = _this.backendEngine.executeKernel('AvgPoolBackprop', { inputs: { dy: dy4D, x: input4D }, args: { convInfo: convInfo } });
            if (reshapedTo4D) {
                return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
            }
            return res;
        });
    };
    NDArrayMath.prototype.resizeBilinear3D = function (x, newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        util.assert(x.rank === 3, "Error in resizeBilinear3D: x must be rank 3 but got rank " + x.rank + ".");
        util.assert(newShape2D.length === 2, "Error in resizeBilinear3D: new shape must 2D, but got shape " +
            (newShape2D + "."));
        return this.backendEngine.executeKernel('ResizeBilinear3D', { inputs: { x: x }, args: { newShape2D: newShape2D, alignCorners: alignCorners } });
    };
    NDArrayMath.prototype.batchNormalization2D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 2, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 2 || mean.rank === 1, "Error in batchNormalization2D: mean must be rank 2 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 2 || variance.rank === 1, "Error in batchNormalization2D: variance must be rank 2 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 2 || scale.rank === 1, "Error in batchNormalization2D: scale must be rank 2 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 2 || offset.rank === 1, "Error in batchNormalization2D: offset must be rank 2 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.backendEngine.executeKernel('BatchNorm2D', { inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset }, args: { varianceEpsilon: varianceEpsilon } });
    };
    NDArrayMath.prototype.batchNormalization3D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 3 || mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 3 || variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 3 || scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 3 || offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.backendEngine.executeKernel('BatchNorm3D', { inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset }, args: { varianceEpsilon: varianceEpsilon } });
    };
    NDArrayMath.prototype.batchNormalization4D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 4, "Error in batchNormalization4D: x must be rank 4 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 4 || mean.rank === 1, "Error in batchNormalization4D: mean must be rank 4 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 4 || variance.rank === 1, "Error in batchNormalization4D: variance must be rank 4 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 4 || scale.rank === 1, "Error in batchNormalization4D: scale must be rank 4 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 4 || offset.rank === 1, "Error in batchNormalization4D: offset must be rank 4 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.backendEngine.executeKernel('BatchNorm4D', { inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset }, args: { varianceEpsilon: varianceEpsilon } });
    };
    NDArrayMath.prototype.localResponseNormalization3D = function (x, radius, bias, alpha, beta, normRegion) {
        if (radius === void 0) { radius = 5; }
        if (bias === void 0) { bias = 1; }
        if (alpha === void 0) { alpha = 1; }
        if (beta === void 0) { beta = 0.5; }
        if (normRegion === void 0) { normRegion = 'acrossChannels'; }
        util.assert(x.rank === 3, "Error in localResponseNormalization3D: x must be rank 3 but got\n         rank " + x.rank + ".");
        util.assert(util.isInt(radius), "Error in localResponseNormalization3D: radius must be an integer\n         but got radius " + radius + ".");
        var input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        var res = this.localResponseNormalization4D(input4D, radius, bias, alpha, beta, normRegion);
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    };
    NDArrayMath.prototype.localResponseNormalization4D = function (x, radius, bias, alpha, beta, normRegion) {
        if (radius === void 0) { radius = 5; }
        if (bias === void 0) { bias = 1; }
        if (alpha === void 0) { alpha = 1; }
        if (beta === void 0) { beta = 0.5; }
        if (normRegion === void 0) { normRegion = 'acrossChannels'; }
        util.assert(x.rank === 4, "Error in localResponseNormalization4D: x must be rank 4 but got\n         rank " + x.rank + ".");
        util.assert(util.isInt(radius), "Error in localResponseNormalization3D: radius must be an integer\n         but got radius " + radius + ".");
        return this.backendEngine.executeKernel('LRN4D', { inputs: { x: x }, args: { radius: radius, bias: bias, alpha: alpha, beta: beta, normRegion: normRegion } });
    };
    NDArrayMath.prototype.multiRNNCell = function (lstmCells, data, c, h) {
        var res = this.scope(function () {
            var input = data;
            var newStates = [];
            for (var i = 0; i < lstmCells.length; i++) {
                var output = lstmCells[i](input, c[i], h[i]);
                newStates.push(output[0]);
                newStates.push(output[1]);
                input = output[1];
            }
            return newStates;
        });
        var newC = [];
        var newH = [];
        for (var i = 0; i < res.length; i += 2) {
            newC.push(res[i]);
            newH.push(res[i + 1]);
        }
        return [newC, newH];
    };
    NDArrayMath.prototype.basicLSTMCell = function (forgetBias, lstmKernel, lstmBias, data, c, h) {
        var _this = this;
        var res = this.scope(function () {
            var combined = _this.concat2D(data, h, 1);
            var weighted = _this.matMul(combined, lstmKernel);
            var res = _this.add(weighted, lstmBias);
            var batchSize = res.shape[0];
            var sliceCols = res.shape[1] / 4;
            var sliceSize = [batchSize, sliceCols];
            var i = _this.slice2D(res, [0, 0], sliceSize);
            var j = _this.slice2D(res, [0, sliceCols], sliceSize);
            var f = _this.slice2D(res, [0, sliceCols * 2], sliceSize);
            var o = _this.slice2D(res, [0, sliceCols * 3], sliceSize);
            var newC = _this.addStrict(_this.multiplyStrict(c, _this.sigmoid(_this.add(forgetBias, f))), _this.multiplyStrict(_this.sigmoid(i), _this.tanh(j)));
            var newH = _this.multiplyStrict(_this.tanh(newC), _this.sigmoid(o));
            return [newC, newH];
        });
        return [res[0], res[1]];
    };
    NDArrayMath.prototype.multinomial = function (probabilities, numSamples, seed) {
        var _this = this;
        var numOutcomes = probabilities.size;
        if (numOutcomes < 2) {
            throw new Error("Error in multinomial: you need at least 2 outcomes, but got " +
                (numOutcomes + "."));
        }
        if (probabilities.rank > 2) {
            throw new Error("Rank of probabilities must be 1 or 2, but is " + probabilities.rank);
        }
        seed = seed || Math.random();
        var origRank = probabilities.rank;
        if (probabilities.rank === 1) {
            probabilities = probabilities.as2D(1, -1);
        }
        return this.executeOp('multinomial', function () {
            var res = _this.backendEngine.executeKernel('Multinomial', {
                inputs: { probs: probabilities },
                args: { numSamples: numSamples, seed: seed }
            });
            if (origRank === 1) {
                return res.as1D();
            }
            return res;
        });
    };
    NDArrayMath.prototype.oneHot = function (indices, depth, onValue, offValue) {
        if (onValue === void 0) { onValue = 1; }
        if (offValue === void 0) { offValue = 0; }
        if (depth < 2) {
            throw new Error("Error in oneHot: depth must be >=2, but it is " + depth);
        }
        return this.backendEngine.executeKernel('OneHot', { inputs: { indices: indices }, args: { depth: depth, onValue: onValue, offValue: offValue } });
    };
    NDArrayMath.prototype.moments = function (x, axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var result = this.scope(function () {
            var mean = _this.mean(x, axes, keepDims);
            var keepDimsShape = mean.shape;
            if (!keepDims) {
                keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
            }
            var devSquared = _this.square(_this.subtract(x.asType('float32'), mean.reshape(keepDimsShape)));
            var variance = _this.mean(devSquared, axes, keepDims);
            return { mean: mean, variance: variance };
        });
        return result;
    };
    NDArrayMath.prototype.norm = function (x, ord, axis, keepDims) {
        var _this = this;
        if (ord === void 0) { ord = 'euclidean'; }
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        return this.scope(function () {
            var norm = _this.normInternal(x, ord, axis);
            var keepDimsShape = norm.shape;
            if (keepDims) {
                var axes = axis_util.parseAxisParam(axis, x.shape);
                keepDimsShape = axis_util.expandShapeToKeepDim(norm.shape, axes);
            }
            return norm.reshape(keepDimsShape);
        });
    };
    NDArrayMath.prototype.normInternal = function (x, p, axis) {
        if (axis === void 0) { axis = null; }
        if (x.rank === 0) {
            return this.abs(x);
        }
        if (x.rank !== 1 && axis === null) {
            return this.normInternal(x.reshape([-1]), p, axis);
        }
        if (x.rank === 1 || typeof axis === 'number' ||
            axis instanceof Array && axis.length === 1) {
            if (p === 1) {
                return this.sum(this.abs(x), axis);
            }
            if (p === Infinity) {
                return this.max(this.abs(x), axis);
            }
            if (p === -Infinity) {
                return this.min(this.abs(x), axis);
            }
            if (p === 'euclidean' || p === 2) {
                return this.sqrt(this.sum(this.pow(this.abs(x), ndarray_1.Scalar.new(2, 'int32')), axis));
            }
            throw new Error("Error in norm: invalid ord value: " + p);
        }
        if (axis instanceof Array && axis.length === 2) {
            if (p === 1) {
                return this.max(this.sum(this.abs(x), axis[0]), axis[1] - 1);
            }
            if (p === Infinity) {
                return this.max(this.sum(this.abs(x), axis[1]), axis[0]);
            }
            if (p === -Infinity) {
                return this.min(this.sum(this.abs(x), axis[1]), axis[0]);
            }
            if (p === 'fro' || p === 'euclidean') {
                return this.sqrt(this.sum(this.pow(x, ndarray_1.Scalar.new(2, 'int32')), axis));
            }
            throw new Error("Error in norm: invalid ord value: " + p);
        }
        throw new Error("Error in norm: invalid axis: " + axis);
    };
    NDArrayMath.prototype.vjp = function (f, x, dy) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var vjp = this.backendEngine.vjp(f, xs, dy);
        if (x instanceof ndarray_1.NDArray) {
            return vjp[0];
        }
        else {
            return util.unflattenToNameArrayMap(keys, vjp);
        }
    };
    NDArrayMath.prototype.gradients = function (f, x) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var returnValue = false;
        var gradients = this.backendEngine.gradients(f, xs, returnValue);
        if (x instanceof ndarray_1.NDArray) {
            return gradients[0];
        }
        else {
            return util.unflattenToNameArrayMap(keys, gradients);
        }
    };
    NDArrayMath.prototype.variableGradients = function (f) {
        return this.valueAndGradients(f, this.registeredVariables);
    };
    NDArrayMath.prototype.valueAndGradients = function (f, x) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var returnValue = true;
        var valueAndGradients = this.backendEngine.gradients(f, xs, returnValue);
        var gradients;
        if (x instanceof ndarray_1.NDArray) {
            gradients = valueAndGradients.gradients[0];
        }
        else {
            gradients =
                util.unflattenToNameArrayMap(keys, valueAndGradients.gradients);
        }
        return { value: valueAndGradients.value, gradients: gradients };
    };
    NDArrayMath.prototype.customGradient = function (f, inputs, name) {
        return this.backendEngine.customGradient(f, inputs, name == null ? '' : name);
    };
    NDArrayMath.prototype.disposeData = function (dataId) {
        if (!this.registeredArrays.has(dataId)) {
            return;
        }
        var refCount = this.registeredArrays.get(dataId);
        if (refCount <= 1) {
            this.registeredArrays.delete(dataId);
            this.backend.disposeData(dataId);
        }
        else {
            this.registeredArrays.set(dataId, refCount - 1);
        }
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
function parseTupleParam(param) {
    return typeof param === 'number' ? [param, param] : param;
}
