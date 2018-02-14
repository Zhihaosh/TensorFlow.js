"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var concat_util = require("../../math/concat_util");
var op_1 = require("./op");
var Concat1D = (function (_super) {
    __extends(Concat1D, _super);
    function Concat1D(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        return _this;
    }
    Concat1D.prototype.feedForward = function (math, inferecenArrays) {
        var _this = this;
        var x1 = inferecenArrays.get(this.x1Tensor);
        var x2 = inferecenArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat1D(x1, x2);
            inferecenArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat1D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1Size = this.x1Tensor.shape[0];
        var x2Size = this.x2Tensor.shape[0];
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            var slice1Result = math.slice1D(dy, 0, x1Size);
            var slice2Result = math.slice1D(dy, x1Size, x2Size);
            gradientArrays.add(_this.x1Tensor, slice1Result);
            gradientArrays.add(_this.x2Tensor, slice2Result);
        });
    };
    return Concat1D;
}(op_1.Operation));
exports.Concat1D = Concat1D;
var Concat2D = (function (_super) {
    __extends(Concat2D, _super);
    function Concat2D(x1Tensor, x2Tensor, axis, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.axis = axis;
        _this.yTensor = yTensor;
        concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
        return _this;
    }
    Concat2D.prototype.feedForward = function (math, inferecenArrays) {
        var _this = this;
        var x1 = inferecenArrays.get(this.x1Tensor);
        var x2 = inferecenArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat2D(x1, x2, _this.axis);
            inferecenArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat2D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        var _a = concat_util.computeGradientSliceShapes2D(this.x1Tensor.shape, this.yTensor.shape, this.axis), x1Begin = _a.x1Begin, x1Size = _a.x1Size, x2Begin = _a.x2Begin, x2Size = _a.x2Size;
        math.scope(function (keep) {
            var slice1Result = math.slice2D(dy, x1Begin, x1Size);
            var slice2Result = math.slice2D(dy, x2Begin, x2Size);
            gradientArrays.add(_this.x1Tensor, slice1Result);
            gradientArrays.add(_this.x2Tensor, slice2Result);
        });
    };
    return Concat2D;
}(op_1.Operation));
exports.Concat2D = Concat2D;
var Concat3D = (function (_super) {
    __extends(Concat3D, _super);
    function Concat3D(x1Tensor, x2Tensor, axis, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.axis = axis;
        _this.yTensor = yTensor;
        concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
        return _this;
    }
    Concat3D.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat3D(x1, x2, _this.axis);
            inferenceArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat3D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        var _a = concat_util.computeGradientSliceShapes3D(this.x1Tensor.shape, this.yTensor.shape, this.axis), x1Begin = _a.x1Begin, x1Size = _a.x1Size, x2Begin = _a.x2Begin, x2Size = _a.x2Size;
        math.scope(function (keep) {
            var slice1Result = math.slice3D(dy, x1Begin, x1Size);
            var slice2Result = math.slice3D(dy, x2Begin, x2Size);
            gradientArrays.add(_this.x1Tensor, slice1Result);
            gradientArrays.add(_this.x2Tensor, slice2Result);
        });
    };
    return Concat3D;
}(op_1.Operation));
exports.Concat3D = Concat3D;
var Concat4D = (function (_super) {
    __extends(Concat4D, _super);
    function Concat4D(x1Tensor, x2Tensor, axis, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.axis = axis;
        _this.yTensor = yTensor;
        concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
        return _this;
    }
    Concat4D.prototype.feedForward = function (math, inferecenArrays) {
        var _this = this;
        var x1 = inferecenArrays.get(this.x1Tensor);
        var x2 = inferecenArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat4D(x1, x2, _this.axis);
            inferecenArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat4D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        var _a = concat_util.computeGradientSliceShapes4D(this.x1Tensor.shape, this.yTensor.shape, this.axis), x1Begin = _a.x1Begin, x1Size = _a.x1Size, x2Begin = _a.x2Begin, x2Size = _a.x2Size;
        math.scope(function (keep) {
            var slice1Result = math.slice4D(dy, x1Begin, x1Size);
            var slice2Result = math.slice4D(dy, x2Begin, x2Size);
            gradientArrays.add(_this.x1Tensor, slice1Result);
            gradientArrays.add(_this.x2Tensor, slice2Result);
        });
    };
    return Concat4D;
}(op_1.Operation));
exports.Concat4D = Concat4D;
