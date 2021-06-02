import tensorflow as tf
from xlam import *


class AutoML:
    def __init__(self, cfg):
        self.cfg = cfg

        with open(self.cfg["labels_dict"], "r") as f:
            self.labels = []
            for img_label in f.readlines():
                self.labels.append(img_label.replace("\n", ""))

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.cfg["model_name"])

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']

    def preprocess_slice(self, slice_img):

        slice_img = resize_image(img=slice_img,
                                 height=self.input_shape[1],
                                 width=self.input_shape[2],
                                 letterbox=False)

        return slice_img

    def slice_wagon_img(self, img, slice_w=1000, stride=0.5):

        slices = []
        for x in range(100):

            if int(x * slice_w * stride + slice_w) > img.shape[1]:
                break

            slices.append(img[:, int(x * slice_w * stride): int(x * slice_w * stride + slice_w), :])

        return slices

    def predict(self, batch, top_k=None):

        self.interpreter.resize_tensor_input(self.input_details[0]['index'], [len(batch), 224, 224, 3])
        self.interpreter.allocate_tensors()

        self.input_data = np.array(batch, dtype=np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        output = output / 255.
        results_batch = []

        for predicts in output:
            results = []
            if top_k:
                indexes = predicts.argsort()[::-1][:top_k]
                for i in indexes:
                    results.append({'class': self.labels[i], 'prob': round(float(predicts[i]), 2)})

            else:
                pred = np.array([x.argmax() for x in predicts])[0]
                prob = float(np.max(predicts))

                results.append({'class': self.labels[pred], 'prob': round(prob, 2)})

            results_batch.append(results)

        return results_batch
