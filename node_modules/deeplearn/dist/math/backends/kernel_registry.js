"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var ndarray_1 = require("../ndarray");
var KERNEL_METHODS = {
    MatMul: function (backend, config) {
        return backend.matMul(config.inputs.a, config.inputs.b, config.args.aOrientation, config.args.bOrientation);
    },
    Clone: function (backend, config) {
        return backend.clone(config.inputs.x);
    },
    Slice1D: function (backend, config) {
        return backend.slice1D(config.inputs.x, config.args.begin, config.args.size);
    },
    Slice2D: function (backend, config) {
        return backend.slice2D(config.inputs.x, config.args.begin, config.args.size);
    },
    Slice3D: function (backend, config) {
        return backend.slice3D(config.inputs.x, config.args.begin, config.args.size);
    },
    Slice4D: function (backend, config) {
        return backend.slice4D(config.inputs.x, config.args.begin, config.args.size);
    },
    Reverse4D: function (backend, config) {
        return backend.reverse4D(config.inputs.x, config.args.axis);
    },
    Concat1D: function (backend, config) {
        return backend.concat1D(config.inputs.a, config.inputs.b);
    },
    Concat2D: function (backend, config) {
        return backend.concat2D(config.inputs.a, config.inputs.b, config.args.axis);
    },
    Concat3D: function (backend, config) {
        return backend.concat3D(config.inputs.a, config.inputs.b, config.args.axis);
    },
    Concat4D: function (backend, config) {
        return backend.concat4D(config.inputs.a, config.inputs.b, config.args.axis);
    },
    Neg: function (backend, config) {
        return backend.neg(config.inputs.x);
    },
    Add: function (backend, config) {
        return backend.add(config.inputs.a, config.inputs.b);
    },
    Sub: function (backend, config) {
        return backend.subtract(config.inputs.a, config.inputs.b);
    },
    Mul: function (backend, config) {
        return backend.multiply(config.inputs.a, config.inputs.b);
    },
    Div: function (backend, config) {
        return backend.divide(config.inputs.a, config.inputs.b);
    },
    Sum: function (backend, config) {
        return backend.sum(config.inputs.x, config.args.axes);
    },
    ArgMax: function (backend, config) {
        return backend.argMax(config.inputs.x, config.args.axes);
    },
    ArgMin: function (backend, config) {
        return backend.argMin(config.inputs.x, config.args.axes);
    },
    Equal: function (backend, config) {
        return backend.equal(config.inputs.a, config.inputs.b);
    },
    NotEqual: function (backend, config) {
        return backend.notEqual(config.inputs.a, config.inputs.b);
    },
    LessEqual: function (backend, config) {
        return backend.lessEqual(config.inputs.a, config.inputs.b);
    },
    Greater: function (backend, config) {
        return backend.greater(config.inputs.a, config.inputs.b);
    },
    GreaterEqual: function (backend, config) {
        return backend.greaterEqual(config.inputs.a, config.inputs.b);
    },
    LogicalOr: function (backend, config) {
        return backend.logicalOr(config.inputs.a, config.inputs.b);
    },
    TopKValues: function (backend, config) {
        return backend.topKValues(config.inputs.x, config.args.k);
    },
    TopKIndices: function (backend, config) {
        return backend.topKIndices(config.inputs.x, config.args.k);
    },
    Min: function (backend, config) {
        return backend.min(config.inputs.x, config.args.axes);
    },
    Minimum: function (backend, config) {
        return backend.minimum(config.inputs.a, config.inputs.b);
    },
    Max: function (backend, config) {
        return backend.max(config.inputs.x, config.args.axes);
    },
    Maximum: function (backend, config) {
        return backend.maximum(config.inputs.a, config.inputs.b);
    },
    Ceil: function (backend, config) {
        return backend.ceil(config.inputs.x);
    },
    Floor: function (backend, config) {
        return backend.floor(config.inputs.x);
    },
    Pow: function (backend, config) {
        return backend.pow(config.inputs.a, config.inputs.b);
    },
    Exp: function (backend, config) {
        return backend.exp(config.inputs.x);
    },
    Log: function (backend, config) {
        return backend.log(config.inputs.x);
    },
    Sqrt: function (backend, config) {
        return backend.sqrt(config.inputs.x);
    },
    Square: function (backend, config) {
        return backend.square(config.inputs.x);
    },
    Relu: function (backend, config) {
        return backend.relu(config.inputs.x);
    },
    Reshape: function (backend, config) {
        var x = config.inputs.x;
        var newShape = config.args.newShape;
        return ndarray_1.NDArray.make(newShape, { dataId: x.dataId }, x.dtype);
    },
    Cast: function (backend, config) {
        var x = config.inputs.x;
        var newDType = config.args.newDType;
        if (!util.hasEncodingLoss(x.dtype, newDType)) {
            return ndarray_1.NDArray.make(x.shape, { dataId: x.dataId }, newDType);
        }
        if (newDType === 'int32') {
            return backend.int(x);
        }
        else if (newDType === 'bool') {
            return backend.notEqual(x, ndarray_1.Scalar.new(0, x.dtype));
        }
        else {
            throw new Error("Error in Cast: unknown dtype argument (" + newDType + ")");
        }
    },
    LeakyRelu: function (backend, config) {
        return backend.leakyRelu(config.inputs.x, config.args.alpha);
    },
    PReLU: function (backend, config) {
        return backend.prelu(config.inputs.x, config.inputs.alpha);
    },
    PReLUDer: function (backend, config) {
        return backend.preluDer(config.inputs.x, config.inputs.alpha);
    },
    Elu: function (backend, config) {
        return backend.elu(config.inputs.x);
    },
    EluDer: function (backend, config) {
        return backend.eluDer(config.inputs.x);
    },
    Selu: function (backend, config) {
        return backend.selu(config.inputs.x);
    },
    Abs: function (backend, config) {
        return backend.abs(config.inputs.x);
    },
    Sigmoid: function (backend, config) {
        return backend.sigmoid(config.inputs.x);
    },
    Step: function (backend, config) {
        return backend.step(config.inputs.x, config.args.alpha);
    },
    Sin: function (backend, config) {
        return backend.sin(config.inputs.x);
    },
    Cos: function (backend, config) {
        return backend.cos(config.inputs.x);
    },
    Tan: function (backend, config) {
        return backend.tan(config.inputs.x);
    },
    Asin: function (backend, config) {
        return backend.asin(config.inputs.x);
    },
    Acos: function (backend, config) {
        return backend.acos(config.inputs.x);
    },
    Atan: function (backend, config) {
        return backend.atan(config.inputs.x);
    },
    Sinh: function (backend, config) {
        return backend.sinh(config.inputs.x);
    },
    Cosh: function (backend, config) {
        return backend.cosh(config.inputs.x);
    },
    Tanh: function (backend, config) {
        return backend.tanh(config.inputs.x);
    },
    Clip: function (backend, config) {
        return backend.clip(config.inputs.x, config.args.min, config.args.max);
    },
    Tile: function (backend, config) {
        return backend.tile(config.inputs.x, config.args.reps);
    },
    Pad1D: function (backend, config) {
        return backend.pad1D(config.inputs.x, config.args.paddings, config.args.constantValue);
    },
    Pad2D: function (backend, config) {
        return backend.pad2D(config.inputs.x, config.args.paddings, config.args.constantValue);
    },
    Transpose: function (backend, config) {
        return backend.transpose(config.inputs.x, config.args.perm);
    },
    Conv2D: function (backend, config) {
        return backend.conv2d(config.inputs.x, config.inputs.filter, config.inputs.bias, config.args.convInfo);
    },
    Conv2DDerInput: function (backend, config) {
        return backend.conv2dDerInput(config.inputs.dy, config.inputs.filter, config.args.convInfo);
    },
    Conv2DDerFilter: function (backend, config) {
        return backend.conv2dDerFilter(config.inputs.x, config.inputs.dy, config.args.convInfo);
    },
    Conv2DDerBias: function (backend, config) {
        return backend.conv2dDerBias(config.inputs.dy);
    },
    DepthwiseConv2D: function (backend, config) {
        return backend.depthwiseConv2D(config.inputs.x, config.inputs.filter, config.args.convInfo);
    },
    MaxPool: function (backend, config) {
        return backend.maxPool(config.inputs.x, config.args.convInfo);
    },
    MaxPoolBackprop: function (backend, config) {
        return backend.maxPoolBackprop(config.inputs.dy, config.inputs.x, config.args.convInfo);
    },
    AvgPool: function (backend, config) {
        return backend.avgPool(config.inputs.x, config.args.convInfo);
    },
    AvgPoolBackprop: function (backend, config) {
        return backend.avgPoolBackprop(config.inputs.dy, config.inputs.x, config.args.convInfo);
    },
    MinPool: function (backend, config) {
        return backend.minPool(config.inputs.x, config.args.convInfo);
    },
    ResizeBilinear3D: function (backend, config) {
        return backend.resizeBilinear3D(config.inputs.x, config.args.newShape2D, config.args.alignCorners);
    },
    BatchNorm4D: function (backend, config) {
        return backend.batchNormalization4D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    },
    BatchNorm3D: function (backend, config) {
        return backend.batchNormalization3D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    },
    BatchNorm2D: function (backend, config) {
        return backend.batchNormalization2D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    },
    LRN4D: function (backend, config) {
        return backend.localResponseNormalization4D(config.inputs.x, config.args.radius, config.args.bias, config.args.alpha, config.args.beta, config.args.normRegion);
    },
    Multinomial: function (backend, config) {
        return backend.multinomial(config.inputs.probs, config.args.numSamples, config.args.seed);
    },
    OneHot: function (backend, config) {
        return backend.oneHot(config.inputs.indices, config.args.depth, config.args.onValue, config.args.offValue);
    }
};
function executeKernel(backend, kernelName, config) {
    return KERNEL_METHODS[kernelName](backend, config);
}
exports.executeKernel = executeKernel;
