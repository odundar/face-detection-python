# Deep Learning Models

In this face detection applications, I used models from Intel's Open Model Zoo and from `deepinsight` github repository.

## Open Model Zoo Face & Age-Gender Detection Models

There are numerous PoC type of DL models by intel in the following repository:

https://github.com/opencv/open_model_zoo

I have used face detection and age-gender detection models respectively:

- https://github.com/opencv/open_model_zoo/tree/master/intel_models/face-detection-retail-0004
- https://github.com/opencv/open_model_zoo/tree/master/intel_models/age-gender-recognition-retail-0013

These models can be downloaded by OpenVINO(TM) Toolkit model downloader using below commands:

```bash
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name age-gender-recognition-retail-0013 --output_dir openvino_models/
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-retail-0004 --output_dir openvino_models/
```

You can use the models from their downloaded directory. Each has FP16 version as well. They are ready to be used with Inference Engine no extra work required.

Please also check the descriptions from the `description/face-detection-retail-0004.md` file to understand their input and outputs.

## MTCNN Face & Age-Gender Detection Models

### Face Detection Model

Source of Face Detection Model:

- https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection

Get Models and convert them using Model Optimizer.

Models are being trained using MxNet* framework therefore we will check guidance for conversion from MxNet conversion guideline:

- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet.html

```bash
git clone https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection.git

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_mxnet.py --input_model mxnet_mtcnn_face_detection/model/det1-0001.params --input_symbol mxnet_mtcnn_face_detection/model/det1-symbol.json --input_shape [1,3,12,12] --output_dir FP32/ --reverse_input_channels

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_mxnet.py --input_model mxnet_mtcnn_face_detection/model/det2-0001.params --input_symbol mxnet_mtcnn_face_detection/model/det2-symbol.json --input_shape [1,3,24,24] --output_dir FP32/ --reverse_input_channels

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_mxnet.py --input_model mxnet_mtcnn_face_detection/model/det3-0001.params --input_symbol mxnet_mtcnn_face_detection/model/det3-symbol.json --input_shape [1,3,48,48] --output_dir FP32/ --reverse_input_channels
```

You can also change FP32 to FP16 and make other corrections. 

### Age Gender Model

- https://github.com/deepinsight/insightface/tree/master/gender-age

Pre-trained Model URL:

https://www.dropbox.com/s/2xq8mcao6z14e3u/gamodel-r50.zip?dl=0

You can convert models using Model Optimizer to make them ready to be used with OpenVINO(TM)

Deepinsight models are being trained using MxNet* framework therefore we will check guidance for conversion from MxNet conversion guideline:

- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet.html

```bash
unzip gamodel-r50.zip

cd gamodel-r50

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_mxnet.py --input_model model-0000.params --input_shape [1,3,112,112] --output_dir FP32 --data_type FP32 --scale 0.0399 --mean_values [127.5,127.5,127.5]
```

Important part here is the --input_shape, --mean_values and --scale parameters, you should investigate what they should be from the model descriptions page. Otherwise you will get faulty output.

# Important Notes for Model Conversion

Important factor while converting models to IR representations is that, all layers should be supported by Intel(R) Disribution of OpenVINO(TM) Model Optimizer otherwise certain errors will be occured.

If layers not supported, you should follow some additional development process and extend OpenVINO(TM) Toolkit Model Optimizer and add layer primitive implementation to use it in the runtime.

Many of the layer implementations are supported for MxNet*, Tensorflow*, Caffe*, ONNX* and Kaldi* 

Below are the guidelines to follow from documentation to learn about the model conversion process.

- Here is the general guideline for conversion process:

    - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Prepare_Trained_Model.html
    - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html

- Here you can find the list of supported layers for each framework:

    - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

If there are custom layers being used you need to add those layers with using defined steps in OpenVINO(TM) Toolkit documentation.

- Adding Custom Layers: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html