package ai.djl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.code.StableDiffusionModel;
import ai.djl.modality.cv.Image;
import ai.djl.translate.TranslateException;

public final class Main {

    private Main() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        StableDiffusionModel model = new StableDiffusionModel(Device.cpu());
        Image result = model.generateImageFromText("sexy girl", 50);
        saveImage(result, "generated", "build/output");
    }

    public static void saveImage(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }
}
