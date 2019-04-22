import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class Main {

    public static void main(String[] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        System.out.println(TensorFlow.version());

//        runExample();

        String simpleMlp = new ClassPathResource("doubleCalc.h5").getFile().getPath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);

        int valueToCalculateDoubleFrom = 7;

        // calculate double of 7
        INDArray features = Nd4j.zeros(1);
        features.putScalar(new int[]{0}, valueToCalculateDoubleFrom);

        // get the prediction
        double prediction = model.output(features).getDouble(0);
        System.out.println("" + prediction); //should be around 13.8
    }

    private static void runExample() {
        try (Graph g = new Graph()) {
            final String value = "Hello from " + TensorFlow.version();
            // Construct the computation graph with a single operation, a constant
            // named "MyConst" with a value "value".
            try (Tensor t = Tensor.create(value.getBytes(StandardCharsets.UTF_8))) {
                // The Java API doesn't yet include convenience functions for adding operations.
                g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
            }
            // Execute the "MyConst" operation in a Session.
            try (Session s = new Session(g);
                 Tensor output = s.runner().fetch("MyConst").run().get(0)) {
                System.out.println(new String(output.bytesValue(), StandardCharsets.UTF_8));
            }
        }
    }
}
