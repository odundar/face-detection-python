# Face Detection Application with Deepinsight InsightFace Deep Learning Models

This is a basic Face Detection application using Intel(R) Distribution of OpenVINO(TM) Toolkit and Age-Gender Model from https://github.com/deepinsight/insightface.

I aim to show how OpenVINO(TM) Toolkit pre-trained model and a custom model can be used together with OpenVINO(TM) Toolkit Inference Engine Python API. 

# Convert Models to OpenVINO(TM) IR Files

In this basic tutorial, we will use Intel's pre-trained face detection model and age gender model from the deepinsight repository.
 
Source: https://github.com/deepinsight/insightface

- Models in deepinsight repo are licensed under MIT License.

Good thing about deepinsight models that, you can reuse them and retrain as your need since all the details about models are already stored in the github page.

- OpenVINO(TM) Toolkit pre-trained models are licensed under Apache Version 2.0.

## Download Pre-trained Models with Model Downloader

#### Face Detection Model of OpenVINO(TM) Toolkit:

Download Model:

```bash
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 -o /home/intel/openvino_models/
```

You can see the details of this model from the model zoo, which are required to know about input and output shapes.

`https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-adas-0001/description/face-detection-adas-0001.md`

#### Gender Age Model:

https://github.com/deepinsight/insightface/tree/master/gender-age

Pre-trained Model URL:

https://www.dropbox.com/s/2xq8mcao6z14e3u/gamodel-r50.zip?dl=0

## Convert Custom Models 

### Important Notes for Model Conversion

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

Deepinsight models are being trained using MxNet* framework therefore we will check guidance for conversion from MxNet conversion guideline:

- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet.html

### Model Optimizer

### Age-Gender Model Conversion

```bash
unzip gamodel-r50.zip

cd gamodel-r50

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_mxnet.py --input_model model-0000.params --input_shape [1,3,112,112] --output_dir FP32 --data_type FP32 --scale 0.0399 --mean_values [127.5,127.5,127.5]

```

Important part here is the --input_shape, --mean_values and --scale parameters, you should investigate what they should be from the model descriptions page. Otherwise you will get faulty output.

# Run Application

Please see the Python code file `face_detection_openvino.py`, which shows how to load converted models, and start using them.

Running and consuming deep learning models is easy, only thing you should do that the post-processing output vectors as they supposed to be. 

Therefore, when you know the conversion of those layers to meaningful values then you are ready to go. 

Please see the source code for detailed explanation of usage.