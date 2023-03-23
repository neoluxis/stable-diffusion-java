package ai.djl.code;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class ImageDecoder implements NoBatchifyTranslator<NDArray, Image> {

    @Override
    public NDList processInput(TranslatorContext ctx, NDArray input) throws Exception {
        input = input.div(0.18215);
        return new NDList(input);
    }

    @Override
    public Image processOutput(TranslatorContext ctx, NDList output) throws Exception {
        NDArray scaled = output.get(0).div(2).add(0.5).clip(0, 1);
        scaled = scaled.transpose(0, 2, 3, 1);
        scaled = scaled.mul(255).round().toType(DataType.INT8, true).get(0);
        return ImageFactory.getInstance().fromNDArray(scaled);
    }
}
