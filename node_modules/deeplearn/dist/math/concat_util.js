"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function assertParams(aShape, bShape, axis) {
    var aRank = aShape.length;
    var bRank = bShape.length;
    util.assert(aShape.length === bShape.length, "Error in concat" + aRank + "D: rank of x1 (" + aRank + ") and x2 (" + bRank + ") " +
        "must be the same.");
    util.assert(axis >= 0 && axis < aRank, "Error in concat" + aRank + "D: axis must be " +
        ("between 0 and " + (aRank - 1) + "."));
    for (var i = 0; i < aRank; i++) {
        util.assert((i === axis) || (aShape[i] === bShape[i]), "Error in concat" + aRank + "D: Shape (" + aShape + ") does not match " +
            ("(" + bShape + ") along the non-concatenated axis " + i + "."));
    }
}
exports.assertParams = assertParams;
function computeOutShape1D(x1Shape, x2Shape) {
    util.assert(x1Shape.length === 1 && x2Shape.length === 1, 'x1 and x2 should be 1d array.');
    var outputShape = x1Shape.slice();
    outputShape[0] += x2Shape[0];
    return outputShape;
}
exports.computeOutShape1D = computeOutShape1D;
function computeOutShape(x1Shape, x2Shape, axis) {
    util.assert(x1Shape.length === x2Shape.length, 'x1 and x2 should have the same rank.');
    var outputShape = x1Shape.slice();
    outputShape[axis] += x2Shape[axis];
    return outputShape;
}
exports.computeOutShape = computeOutShape;
function computeGradientSliceShapes2D(x1TensorShape, yTensorShape, axis) {
    var x1AxisSize = x1TensorShape[axis];
    var x1Begin = [0, 0];
    var x1Size = [yTensorShape[0], yTensorShape[1]];
    x1Size[axis] = x1AxisSize;
    var x2Begin = [0, 0];
    x2Begin[axis] = x1AxisSize;
    var x2Size = [yTensorShape[0], yTensorShape[1]];
    x2Size[axis] = yTensorShape[axis] - x1AxisSize;
    return { x1Begin: x1Begin, x1Size: x1Size, x2Begin: x2Begin, x2Size: x2Size };
}
exports.computeGradientSliceShapes2D = computeGradientSliceShapes2D;
function computeGradientSliceShapes3D(x1Shape, yShape, axis) {
    var x1AxisSize = x1Shape[axis];
    var x1Begin = [0, 0, 0];
    var x1Size = [yShape[0], yShape[1], yShape[2]];
    x1Size[axis] = x1AxisSize;
    var x2Begin = [0, 0, 0];
    x2Begin[axis] = x1AxisSize;
    var x2Size = [yShape[0], yShape[1], yShape[2]];
    x2Size[axis] = yShape[axis] - x1AxisSize;
    return { x1Begin: x1Begin, x1Size: x1Size, x2Begin: x2Begin, x2Size: x2Size };
}
exports.computeGradientSliceShapes3D = computeGradientSliceShapes3D;
function computeGradientSliceShapes4D(x1TensorShape, yTensorShape, axis) {
    var x1AxisSize = x1TensorShape[axis];
    var x1Begin = [0, 0, 0, 0];
    var x1Size = [yTensorShape[0], yTensorShape[1], yTensorShape[2], yTensorShape[3]];
    x1Size[axis] = x1AxisSize;
    var x2Begin = [0, 0, 0, 0];
    x2Begin[axis] = x1AxisSize;
    var x2Size = [yTensorShape[0], yTensorShape[1], yTensorShape[2], yTensorShape[3]];
    x2Size[axis] = yTensorShape[axis] - x1AxisSize;
    return { x1Begin: x1Begin, x1Size: x1Size, x2Begin: x2Begin, x2Size: x2Size };
}
exports.computeGradientSliceShapes4D = computeGradientSliceShapes4D;
