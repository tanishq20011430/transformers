<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

These models can be applied on:

* 📝 Text, for tasks like text classification, information extraction, question answering, summarization, translation, and text generation, in over 100 languages.
* 🖼️ Images, for tasks like image classification, object detection, and segmentation.
* 🗣️ Audio, for tasks like speech recognition and audio classification.

Transformer models can also perform tasks on **several modalities combined**, such as table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our [model hub](https://huggingface.co/models). At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other.

## Online demos

You can test most of our models directly on their pages from the [model hub](https://huggingface.co/models). We also offer [private model hosting, versioning, & an inference API](https://huggingface.co/pricing) for public and private models.

Here are a few examples:

In Natural Language Processing:
- [Masked word completion with BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Named Entity Recognition with Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Text generation with Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Natural Language Inference with RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Summarization with BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Question answering with DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Translation with T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

In Computer Vision:
- [Image classification with ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Object Detection with DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Semantic Segmentation with SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Panoptic Segmentation with Mask2Former](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [Depth Estimation with Depth Anything](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [Video Classification with VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Universal Segmentation with OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

In Audio:
- [Automatic Speech Recognition with Whisper](https://huggingface.co/openai/whisper-large-v3)
- [Keyword Spotting with Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Audio Classification with Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

In Multimodal tasks:
- [Table Question Answering with TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Visual Question Answering with ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Image captioning with LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Zero-shot Image Classification with SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Document Question Answering with LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Zero-shot Video Classification with X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [Zero-shot Object Detection with OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [Zero-shot Image Segmentation with CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [Automatic Mask Generation with SAM](https://huggingface.co/docs/transformers/model_doc/sam)


## 100 projects using Transformers

Transformers is more than a toolkit to use pretrained models: it's a community of projects built around it and the
Hugging Face Hub. We want Transformers to enable developers, researchers, students, professors, engineers, and anyone
else to build their dream projects.

In order to celebrate the 100,000 stars of transformers, we have decided to put the spotlight on the
community, and we have created the [awesome-transformers](./awesome-transformers.md) page which lists 100
incredible projects built in the vicinity of transformers.

If you own or use a project that you believe should be part of the list, please open a PR to add it!

## Serious about AI in your organisation? Build faster with the Hugging Face Enterprise Hub.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Quick tour

To immediately use a model on a given input (text, image, audio, ...), we provide the `pipeline` API. Pipelines group together a pretrained model with the preprocessing that was used during that model's training. Here is how to quickly use a pipeline to classify positive versus negative texts:

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

The second line of code downloads and caches the pretrained model used by the pipeline, while the third evaluates it on the given text. Here, the answer is "positive" with a confidence of 99.97%.

Many tasks have a pre-trained `pipeline` ready to go, in NLP but also in computer vision and speech. For example, we can easily extract detected objects in an image:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

Here, we get a list of objects detected in the image, with a box surrounding the object and a confidence score. Here is the original image on the left, with the predictions displayed on the right:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

You can learn more about the tasks supported by the `pipeline` API in [this tutorial](https://huggingface.co/docs/transformers/task_summary).

In addition to `pipeline`, to download and use any of the pretrained models on your given task, all it takes is three lines of code. Here is the PyTorch version:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

And here is the equivalent code for TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

The tokenizer is responsible for all the preprocessing the pretrained model expects and can be called directly on a single string (as in the above examples) or a list. It will output a dictionary that you can use in downstream code or simply directly pass to your model using the ** argument unpacking operator.

The model itself is a regular [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) or a [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (depending on your backend) which you can use as usual. [This tutorial](https://huggingface.co/docs/transformers/training) explains how to integrate such a model into a classic PyTorch or TensorFlow training loop, or how to use our `Trainer` API to quickly fine-tune on a new dataset.

## Why should I use transformers?

1. Easy-to-use state-of-the-art models:
    - High performance on natural language understanding & generation, computer vision, and audio tasks.
    - Low barrier to entry for educators and practitioners.
    - Few user-facing abstractions with just three classes to learn.
    - A unified API for using all our pretrained models.

1. Lower compute costs, smaller carbon footprint:
    - Researchers can share trained models instead of always retraining.
    - Practitioners can reduce compute time and production costs.
    - Dozens of architectures with over 400,000 pretrained models across all modalities.

1. Choose the right framework for every part of a model's lifetime:
    - Train state-of-the-art models in 3 lines of code.
    - Move a single model between TF2.0/PyTorch/JAX frameworks at will.
    - Seamlessly pick the right framework for training, evaluation, and production.

1. Easily customize a model or an example to your needs:
    - We provide examples for each architecture to reproduce the results published by its original authors.
    - Model internals are exposed as consistently as possible.
    - Model files can be used independently of the library for quick experiments.

## Why shouldn't I use transformers?

- This library is not a modular toolbox of building blocks for neural nets. The code in the model files is not refactored with additional abstractions on purpose, so that researchers can quickly iterate on each of the models without diving into additional abstractions/files.
- The training API is not intended to work on any model but is optimized to work with the models provided by the library. For generic machine learning loops, you should use another library (possibly, [Accelerate](https://huggingface.co/docs/accelerate)).
- While we strive to present as many use cases as possible, the scripts in our [examples folder](https://github.com/huggingface/transformers/tree/main/examples) are just that: examples. It is expected that they won't work out-of-the-box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs.

## Installation

### With pip

This repository is tested on Python 3.9+, Flax 0.4.1+, PyTorch 1.11+, and TensorFlow 2.6+.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install at least one of Flax, PyTorch, or TensorFlow.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/), [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or [Flax](https://github.com/google/flax#quick-install) and [Jax](https://github.com/google/jax#installation) installation pages regarding the specific installation command for your platform.

When one of those backends has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you must [install the library from source](https://huggingface.co/docs/transformers/installation#installing-from-source).

### With conda

🤗 Transformers can be installed using conda as follows:

```shell script
conda install conda-forge::transformers
```

> **_NOTE:_** Installing `transformers` from the `huggingface` channel is deprecated.

Follow the installation pages of Flax, PyTorch or TensorFlow to see how to install them with conda.

> **_NOTE:_**  On Windows, you may be prompted to activate Developer Mode in order to benefit from caching. If this is not an option for you, please let us know in [this issue](https://github.com/huggingface/huggingface_hub/issues/1062).

## Model architectures

**[All the model checkpoints](https://huggingface.co/models)** provided by 🤗 Transformers are seamlessly integrated from the huggingface.co [model hub](https://huggingface.co/models), where they are uploaded directly by [users](https://huggingface.co/users) and [organizations](https://huggingface.co/organizations).

Current number of checkpoints: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers currently provides the following architectures: see [here](https://huggingface.co/docs/transformers/model_summary) for a high-level summary of each them.

To check if each model has an implementation in Flax, PyTorch or TensorFlow, or has an associated tokenizer backed by the 🤗 Tokenizers library, refer to [this table](https://huggingface.co/docs/transformers/index#supported-frameworks).

These implementations have been tested on several datasets (see the example scripts) and should match the performance of the original implementations. You can find more details on performance in the Examples section of the [documentation](https://github.com/huggingface/transformers/tree/main/examples).


## Learn more

| Section | Description |
|-|-|
| [Documentation](https://huggingface.co/docs/transformers/) | Full API documentation and tutorials |
| [Task summary](https://huggingface.co/docs/transformers/task_summary) | Tasks supported by 🤗 Transformers |
| [Preprocessing tutorial](https://huggingface.co/docs/transformers/preprocessing) | Using the `Tokenizer` class to prepare data for the models |
| [Training and fine-tuning](https://huggingface.co/docs/transformers/training) | Using the models provided by 🤗 Transformers in a PyTorch/TensorFlow training loop and the `Trainer` API |
| [Quick tour: Fine-tuning/usage scripts](https://github.com/huggingface/transformers/tree/main/examples) | Example scripts for fine-tuning models on a wide range of tasks |
| [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing) | Upload and share your fine-tuned models with the community |

## Citation

We now have a [paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) you can cite for the 🤗 Transformers library:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```


### Automated Update - Sat Feb  1 06:21:58 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:26:25 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:37:42 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:42:56 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:54:42 UTC 2025 🚀


### Automated Update - Sat Feb  1 12:13:24 UTC 2025 🚀


### Automated Update - Sun Feb  2 00:41:33 UTC 2025 🚀


### Automated Update - Sun Feb  2 12:13:11 UTC 2025 🚀


### Automated Update - Mon Feb  3 00:40:30 UTC 2025 🚀


### Automated Update - Mon Feb  3 12:15:19 UTC 2025 🚀


### Automated Update - Tue Feb  4 00:39:22 UTC 2025 🚀


### Automated Update - Tue Feb  4 12:16:11 UTC 2025 🚀


### Automated Update - Wed Feb  5 00:39:28 UTC 2025 🚀


### Automated Update - Wed Feb  5 12:16:11 UTC 2025 🚀


### Automated Update - Thu Feb  6 00:39:57 UTC 2025 🚀


### Automated Update - Thu Feb  6 12:16:16 UTC 2025 🚀


### Automated Update - Fri Feb  7 00:39:43 UTC 2025 🚀


### Automated Update - Fri Feb  7 12:15:33 UTC 2025 🚀


### Automated Update - Sat Feb  8 00:38:39 UTC 2025 🚀


### Automated Update - Sat Feb  8 12:13:53 UTC 2025 🚀


### Automated Update - Sun Feb  9 00:42:37 UTC 2025 🚀


### Automated Update - Sun Feb  9 12:13:42 UTC 2025 🚀


### Automated Update - Mon Feb 10 00:40:54 UTC 2025 🚀


### Automated Update - Mon Feb 10 12:15:37 UTC 2025 🚀


### Automated Update - Tue Feb 11 00:40:00 UTC 2025 🚀


### Automated Update - Tue Feb 11 12:16:00 UTC 2025 🚀


### Automated Update - Wed Feb 12 00:39:42 UTC 2025 🚀


### Automated Update - Wed Feb 12 12:15:49 UTC 2025 🚀


### Automated Update - Thu Feb 13 00:40:04 UTC 2025 🚀


### Automated Update - Thu Feb 13 12:15:43 UTC 2025 🚀


### Automated Update - Fri Feb 14 00:39:37 UTC 2025 🚀


### Automated Update - Fri Feb 14 12:15:42 UTC 2025 🚀


### Automated Update - Sat Feb 15 00:39:04 UTC 2025 🚀


### Automated Update - Sat Feb 15 12:13:49 UTC 2025 🚀


### Automated Update - Sun Feb 16 00:43:40 UTC 2025 🚀


### Automated Update - Sun Feb 16 12:16:21 UTC 2025 🚀


### Automated Update - Mon Feb 17 00:42:38 UTC 2025 🚀


### Automated Update - Mon Feb 17 12:16:06 UTC 2025 🚀


### Automated Update - Tue Feb 18 00:39:23 UTC 2025 🚀


### Automated Update - Tue Feb 18 12:16:00 UTC 2025 🚀


### Automated Update - Wed Feb 19 00:39:50 UTC 2025 🚀


### Automated Update - Wed Feb 19 12:15:28 UTC 2025 🚀


### Automated Update - Thu Feb 20 00:40:31 UTC 2025 🚀


### Automated Update - Thu Feb 20 12:16:14 UTC 2025 🚀


### Automated Update - Fri Feb 21 00:40:19 UTC 2025 🚀


### Automated Update - Fri Feb 21 12:15:29 UTC 2025 🚀


### Automated Update - Sat Feb 22 00:38:59 UTC 2025 🚀


### Automated Update - Sat Feb 22 12:13:16 UTC 2025 🚀


### Automated Update - Sun Feb 23 00:43:24 UTC 2025 🚀


### Automated Update - Sun Feb 23 12:13:42 UTC 2025 🚀


### Automated Update - Mon Feb 24 00:42:05 UTC 2025 🚀


### Automated Update - Mon Feb 24 12:16:16 UTC 2025 🚀


### Automated Update - Tue Feb 25 00:40:51 UTC 2025 🚀


### Automated Update - Tue Feb 25 12:16:04 UTC 2025 🚀


### Automated Update - Wed Feb 26 00:40:39 UTC 2025 🚀


### Automated Update - Wed Feb 26 12:16:35 UTC 2025 🚀


### Automated Update - Thu Feb 27 00:41:06 UTC 2025 🚀


### Automated Update - Thu Feb 27 12:16:12 UTC 2025 🚀


### Automated Update - Fri Feb 28 00:41:18 UTC 2025 🚀


### Automated Update - Fri Feb 28 12:15:33 UTC 2025 🚀


### Automated Update - Sat Mar  1 00:44:24 UTC 2025 🚀


### Automated Update - Sat Mar  1 12:14:08 UTC 2025 🚀


### Automated Update - Sun Mar  2 00:44:15 UTC 2025 🚀


### Automated Update - Sun Mar  2 12:13:43 UTC 2025 🚀


### Automated Update - Mon Mar  3 00:42:39 UTC 2025 🚀


### Automated Update - Mon Mar  3 12:16:51 UTC 2025 🚀


### Automated Update - Tue Mar  4 00:41:31 UTC 2025 🚀


### Automated Update - Tue Mar  4 12:16:15 UTC 2025 🚀


### Automated Update - Wed Mar  5 00:41:43 UTC 2025 🚀


### Automated Update - Wed Mar  5 12:16:29 UTC 2025 🚀


### Automated Update - Thu Mar  6 00:41:25 UTC 2025 🚀


### Automated Update - Thu Mar  6 12:15:55 UTC 2025 🚀


### Automated Update - Fri Mar  7 00:42:00 UTC 2025 🚀


### Automated Update - Fri Mar  7 12:15:37 UTC 2025 🚀


### Automated Update - Sat Mar  8 00:32:58 UTC 2025 🚀


### Automated Update - Sat Mar  8 12:11:41 UTC 2025 🚀


### Automated Update - Sun Mar  9 00:36:50 UTC 2025 🚀


### Automated Update - Sun Mar  9 12:11:42 UTC 2025 🚀


### Automated Update - Mon Mar 10 00:36:07 UTC 2025 🚀


### Automated Update - Mon Mar 10 12:26:59 UTC 2025 🚀


### Automated Update - Tue Mar 11 00:43:13 UTC 2025 🚀


### Automated Update - Tue Mar 11 12:16:46 UTC 2025 🚀


### Automated Update - Wed Mar 12 00:41:23 UTC 2025 🚀


### Automated Update - Wed Mar 12 12:16:12 UTC 2025 🚀


### Automated Update - Thu Mar 13 00:42:08 UTC 2025 🚀


### Automated Update - Thu Mar 13 12:16:35 UTC 2025 🚀


### Automated Update - Fri Mar 14 00:41:44 UTC 2025 🚀


### Automated Update - Fri Mar 14 12:15:56 UTC 2025 🚀


### Automated Update - Sat Mar 15 00:41:02 UTC 2025 🚀


### Automated Update - Sat Mar 15 12:14:09 UTC 2025 🚀


### Automated Update - Sun Mar 16 00:45:49 UTC 2025 🚀


### Automated Update - Sun Mar 16 12:14:24 UTC 2025 🚀


### Automated Update - Mon Mar 17 00:43:47 UTC 2025 🚀


### Automated Update - Mon Mar 17 12:16:50 UTC 2025 🚀


### Automated Update - Tue Mar 18 00:41:56 UTC 2025 🚀


### Automated Update - Tue Mar 18 12:16:37 UTC 2025 🚀


### Automated Update - Wed Mar 19 00:42:23 UTC 2025 🚀


### Automated Update - Wed Mar 19 12:16:35 UTC 2025 🚀


### Automated Update - Thu Mar 20 00:41:44 UTC 2025 🚀


### Automated Update - Thu Mar 20 12:17:25 UTC 2025 🚀


### Automated Update - Fri Mar 21 00:42:34 UTC 2025 🚀


### Automated Update - Fri Mar 21 12:16:05 UTC 2025 🚀


### Automated Update - Sat Mar 22 00:41:30 UTC 2025 🚀


### Automated Update - Sat Mar 22 12:14:25 UTC 2025 🚀


### Automated Update - Sun Mar 23 00:46:09 UTC 2025 🚀


### Automated Update - Sun Mar 23 12:14:48 UTC 2025 🚀


### Automated Update - Mon Mar 24 00:44:09 UTC 2025 🚀


### Automated Update - Mon Mar 24 12:17:38 UTC 2025 🚀


### Automated Update - Tue Mar 25 00:42:55 UTC 2025 🚀


### Automated Update - Tue Mar 25 12:17:02 UTC 2025 🚀


### Automated Update - Wed Mar 26 00:42:42 UTC 2025 🚀


### Automated Update - Wed Mar 26 12:16:58 UTC 2025 🚀


### Automated Update - Thu Mar 27 00:42:51 UTC 2025 🚀


### Automated Update - Thu Mar 27 12:17:14 UTC 2025 🚀


### Automated Update - Fri Mar 28 00:42:36 UTC 2025 🚀


### Automated Update - Fri Mar 28 12:16:29 UTC 2025 🚀


### Automated Update - Sat Mar 29 00:41:59 UTC 2025 🚀


### Automated Update - Sat Mar 29 12:14:39 UTC 2025 🚀


### Automated Update - Sun Mar 30 00:46:38 UTC 2025 🚀


### Automated Update - Sun Mar 30 12:15:08 UTC 2025 🚀


### Automated Update - Mon Mar 31 00:45:57 UTC 2025 🚀


### Automated Update - Mon Mar 31 12:17:17 UTC 2025 🚀


### Automated Update - Tue Apr  1 00:50:33 UTC 2025 🚀


### Automated Update - Tue Apr  1 12:17:24 UTC 2025 🚀


### Automated Update - Wed Apr  2 00:43:21 UTC 2025 🚀


### Automated Update - Wed Apr  2 12:16:59 UTC 2025 🚀


### Automated Update - Thu Apr  3 00:42:38 UTC 2025 🚀


### Automated Update - Thu Apr  3 12:16:51 UTC 2025 🚀


### Automated Update - Fri Apr  4 00:42:41 UTC 2025 🚀


### Automated Update - Fri Apr  4 12:16:36 UTC 2025 🚀


### Automated Update - Sat Apr  5 00:42:05 UTC 2025 🚀


### Automated Update - Sat Apr  5 12:15:04 UTC 2025 🚀


### Automated Update - Sun Apr  6 00:46:31 UTC 2025 🚀


### Automated Update - Sun Apr  6 12:14:58 UTC 2025 🚀


### Automated Update - Mon Apr  7 00:44:52 UTC 2025 🚀


### Automated Update - Mon Apr  7 12:17:25 UTC 2025 🚀


### Automated Update - Tue Apr  8 00:42:45 UTC 2025 🚀


### Automated Update - Tue Apr  8 12:17:12 UTC 2025 🚀


### Automated Update - Wed Apr  9 01:27:59 UTC 2025 🚀


### Automated Update - Wed Apr  9 12:16:45 UTC 2025 🚀


### Automated Update - Thu Apr 10 00:43:20 UTC 2025 🚀


### Automated Update - Thu Apr 10 12:17:17 UTC 2025 🚀


### Automated Update - Fri Apr 11 00:43:51 UTC 2025 🚀


### Automated Update - Fri Apr 11 12:17:06 UTC 2025 🚀


### Automated Update - Sat Apr 12 00:42:36 UTC 2025 🚀


### Automated Update - Sat Apr 12 12:14:52 UTC 2025 🚀


### Automated Update - Sun Apr 13 02:10:06 UTC 2025 🚀


### Automated Update - Sun Apr 13 12:15:26 UTC 2025 🚀


### Automated Update - Mon Apr 14 00:46:09 UTC 2025 🚀


### Automated Update - Mon Apr 14 12:16:56 UTC 2025 🚀


### Automated Update - Tue Apr 15 00:44:27 UTC 2025 🚀


### Automated Update - Tue Apr 15 12:17:13 UTC 2025 🚀


### Automated Update - Wed Apr 16 00:44:54 UTC 2025 🚀


### Automated Update - Wed Apr 16 12:17:23 UTC 2025 🚀


### Automated Update - Thu Apr 17 00:43:30 UTC 2025 🚀


### Automated Update - Thu Apr 17 12:17:05 UTC 2025 🚀


### Automated Update - Fri Apr 18 00:43:28 UTC 2025 🚀


### Automated Update - Fri Apr 18 12:16:14 UTC 2025 🚀


### Automated Update - Sat Apr 19 00:42:06 UTC 2025 🚀


### Automated Update - Sat Apr 19 12:15:02 UTC 2025 🚀


### Automated Update - Sun Apr 20 00:48:22 UTC 2025 🚀


### Automated Update - Sun Apr 20 12:14:53 UTC 2025 🚀


### Automated Update - Mon Apr 21 00:46:54 UTC 2025 🚀


### Automated Update - Mon Apr 21 12:16:51 UTC 2025 🚀


### Automated Update - Tue Apr 22 00:44:28 UTC 2025 🚀


### Automated Update - Tue Apr 22 12:17:18 UTC 2025 🚀


### Automated Update - Wed Apr 23 00:44:06 UTC 2025 🚀


### Automated Update - Wed Apr 23 12:17:32 UTC 2025 🚀


### Automated Update - Thu Apr 24 00:44:15 UTC 2025 🚀


### Automated Update - Thu Apr 24 12:18:11 UTC 2025 🚀


### Automated Update - Fri Apr 25 00:44:48 UTC 2025 🚀


### Automated Update - Fri Apr 25 12:17:30 UTC 2025 🚀


### Automated Update - Sat Apr 26 00:43:18 UTC 2025 🚀


### Automated Update - Sat Apr 26 12:15:01 UTC 2025 🚀


### Automated Update - Sun Apr 27 00:48:13 UTC 2025 🚀


### Automated Update - Sun Apr 27 12:15:01 UTC 2025 🚀


### Automated Update - Mon Apr 28 00:46:30 UTC 2025 🚀


### Automated Update - Mon Apr 28 12:17:31 UTC 2025 🚀


### Automated Update - Tue Apr 29 00:44:25 UTC 2025 🚀


### Automated Update - Tue Apr 29 12:18:55 UTC 2025 🚀


### Automated Update - Wed Apr 30 00:45:14 UTC 2025 🚀


### Automated Update - Wed Apr 30 12:17:00 UTC 2025 🚀


### Automated Update - Thu May  1 00:51:20 UTC 2025 🚀


### Automated Update - Thu May  1 12:16:46 UTC 2025 🚀


### Automated Update - Fri May  2 00:44:56 UTC 2025 🚀


### Automated Update - Fri May  2 12:17:33 UTC 2025 🚀


### Automated Update - Sat May  3 00:43:56 UTC 2025 🚀


### Automated Update - Sat May  3 12:15:06 UTC 2025 🚀


### Automated Update - Sun May  4 00:51:23 UTC 2025 🚀


### Automated Update - Sun May  4 12:15:37 UTC 2025 🚀


### Automated Update - Mon May  5 00:48:34 UTC 2025 🚀


### Automated Update - Mon May  5 12:17:46 UTC 2025 🚀


### Automated Update - Tue May  6 00:45:07 UTC 2025 🚀


### Automated Update - Tue May  6 12:19:39 UTC 2025 🚀


### Automated Update - Wed May  7 00:45:16 UTC 2025 🚀


### Automated Update - Wed May  7 12:18:31 UTC 2025 🚀


### Automated Update - Thu May  8 00:45:42 UTC 2025 🚀


### Automated Update - Thu May  8 12:17:37 UTC 2025 🚀


### Automated Update - Fri May  9 00:45:18 UTC 2025 🚀


### Automated Update - Fri May  9 12:17:08 UTC 2025 🚀


### Automated Update - Sat May 10 00:43:25 UTC 2025 🚀


### Automated Update - Sat May 10 12:15:07 UTC 2025 🚀


### Automated Update - Sun May 11 00:50:15 UTC 2025 🚀


### Automated Update - Sun May 11 12:15:21 UTC 2025 🚀


### Automated Update - Mon May 12 00:49:14 UTC 2025 🚀


### Automated Update - Mon May 12 12:18:24 UTC 2025 🚀


### Automated Update - Tue May 13 00:45:58 UTC 2025 🚀


### Automated Update - Tue May 13 12:19:02 UTC 2025 🚀


### Automated Update - Wed May 14 00:45:52 UTC 2025 🚀


### Automated Update - Wed May 14 12:17:53 UTC 2025 🚀


### Automated Update - Thu May 15 00:44:52 UTC 2025 🚀


### Automated Update - Thu May 15 12:18:25 UTC 2025 🚀


### Automated Update - Fri May 16 00:46:52 UTC 2025 🚀


### Automated Update - Fri May 16 12:18:42 UTC 2025 🚀


### Automated Update - Sat May 17 00:44:56 UTC 2025 🚀


### Automated Update - Sat May 17 12:15:49 UTC 2025 🚀


### Automated Update - Sun May 18 00:50:54 UTC 2025 🚀


### Automated Update - Sun May 18 12:16:02 UTC 2025 🚀


### Automated Update - Mon May 19 00:49:47 UTC 2025 🚀


### Automated Update - Mon May 19 12:18:32 UTC 2025 🚀


### Automated Update - Tue May 20 00:47:26 UTC 2025 🚀


### Automated Update - Tue May 20 12:18:35 UTC 2025 🚀


### Automated Update - Wed May 21 00:46:47 UTC 2025 🚀


### Automated Update - Wed May 21 12:18:34 UTC 2025 🚀


### Automated Update - Thu May 22 00:45:55 UTC 2025 🚀


### Automated Update - Thu May 22 12:19:09 UTC 2025 🚀


### Automated Update - Fri May 23 00:45:58 UTC 2025 🚀


### Automated Update - Fri May 23 12:17:49 UTC 2025 🚀


### Automated Update - Sat May 24 00:43:59 UTC 2025 🚀


### Automated Update - Sat May 24 12:15:52 UTC 2025 🚀


### Automated Update - Sun May 25 00:52:17 UTC 2025 🚀


### Automated Update - Sun May 25 12:15:44 UTC 2025 🚀


### Automated Update - Mon May 26 00:48:39 UTC 2025 🚀


### Automated Update - Mon May 26 12:17:30 UTC 2025 🚀


### Automated Update - Tue May 27 00:45:39 UTC 2025 🚀


### Automated Update - Tue May 27 12:18:29 UTC 2025 🚀


### Automated Update - Wed May 28 00:46:36 UTC 2025 🚀


### Automated Update - Wed May 28 12:18:45 UTC 2025 🚀


### Automated Update - Thu May 29 00:46:34 UTC 2025 🚀


### Automated Update - Thu May 29 12:18:10 UTC 2025 🚀


### Automated Update - Fri May 30 00:46:00 UTC 2025 🚀


### Automated Update - Fri May 30 12:17:37 UTC 2025 🚀


### Automated Update - Sat May 31 00:44:55 UTC 2025 🚀


### Automated Update - Sat May 31 12:15:43 UTC 2025 🚀


### Automated Update - Sun Jun  1 00:58:29 UTC 2025 🚀


### Automated Update - Sun Jun  1 12:16:21 UTC 2025 🚀


### Automated Update - Mon Jun  2 00:50:02 UTC 2025 🚀


### Automated Update - Mon Jun  2 12:18:18 UTC 2025 🚀


### Automated Update - Tue Jun  3 00:48:11 UTC 2025 🚀


### Automated Update - Tue Jun  3 12:18:56 UTC 2025 🚀


### Automated Update - Wed Jun  4 00:47:25 UTC 2025 🚀


### Automated Update - Wed Jun  4 12:18:38 UTC 2025 🚀


### Automated Update - Thu Jun  5 00:46:51 UTC 2025 🚀


### Automated Update - Thu Jun  5 12:18:49 UTC 2025 🚀


### Automated Update - Fri Jun  6 00:46:01 UTC 2025 🚀


### Automated Update - Fri Jun  6 12:18:00 UTC 2025 🚀


### Automated Update - Sat Jun  7 00:46:14 UTC 2025 🚀


### Automated Update - Sat Jun  7 12:16:12 UTC 2025 🚀


### Automated Update - Sun Jun  8 00:53:13 UTC 2025 🚀


### Automated Update - Sun Jun  8 12:15:59 UTC 2025 🚀


### Automated Update - Mon Jun  9 00:51:32 UTC 2025 🚀


### Automated Update - Mon Jun  9 12:18:49 UTC 2025 🚀


### Automated Update - Tue Jun 10 00:47:32 UTC 2025 🚀


### Automated Update - Tue Jun 10 12:19:38 UTC 2025 🚀


### Automated Update - Wed Jun 11 00:47:30 UTC 2025 🚀


### Automated Update - Wed Jun 11 12:18:53 UTC 2025 🚀


### Automated Update - Thu Jun 12 00:46:56 UTC 2025 🚀


### Automated Update - Thu Jun 12 12:18:29 UTC 2025 🚀


### Automated Update - Fri Jun 13 00:47:47 UTC 2025 🚀


### Automated Update - Fri Jun 13 12:18:29 UTC 2025 🚀


### Automated Update - Sat Jun 14 00:45:38 UTC 2025 🚀


### Automated Update - Sat Jun 14 12:15:46 UTC 2025 🚀


### Automated Update - Sun Jun 15 00:53:45 UTC 2025 🚀


### Automated Update - Sun Jun 15 12:16:34 UTC 2025 🚀


### Automated Update - Mon Jun 16 00:50:24 UTC 2025 🚀


### Automated Update - Mon Jun 16 12:18:55 UTC 2025 🚀


### Automated Update - Tue Jun 17 00:47:40 UTC 2025 🚀


### Automated Update - Tue Jun 17 12:19:19 UTC 2025 🚀


### Automated Update - Wed Jun 18 00:47:37 UTC 2025 🚀


### Automated Update - Wed Jun 18 12:18:55 UTC 2025 🚀


### Automated Update - Thu Jun 19 00:48:19 UTC 2025 🚀


### Automated Update - Thu Jun 19 12:18:59 UTC 2025 🚀


### Automated Update - Fri Jun 20 00:47:32 UTC 2025 🚀


### Automated Update - Fri Jun 20 12:18:23 UTC 2025 🚀


### Automated Update - Sat Jun 21 00:46:19 UTC 2025 🚀


### Automated Update - Sat Jun 21 12:15:38 UTC 2025 🚀


### Automated Update - Sun Jun 22 00:53:37 UTC 2025 🚀


### Automated Update - Sun Jun 22 12:16:04 UTC 2025 🚀


### Automated Update - Mon Jun 23 00:52:20 UTC 2025 🚀


### Automated Update - Mon Jun 23 12:19:59 UTC 2025 🚀


### Automated Update - Tue Jun 24 00:48:24 UTC 2025 🚀


### Automated Update - Tue Jun 24 12:19:05 UTC 2025 🚀


### Automated Update - Wed Jun 25 00:48:58 UTC 2025 🚀


### Automated Update - Wed Jun 25 12:19:06 UTC 2025 🚀


### Automated Update - Thu Jun 26 00:48:09 UTC 2025 🚀


### Automated Update - Thu Jun 26 12:18:39 UTC 2025 🚀


### Automated Update - Fri Jun 27 00:49:23 UTC 2025 🚀


### Automated Update - Fri Jun 27 12:18:44 UTC 2025 🚀


### Automated Update - Sat Jun 28 00:46:03 UTC 2025 🚀


### Automated Update - Sat Jun 28 12:16:13 UTC 2025 🚀


### Automated Update - Sun Jun 29 00:54:44 UTC 2025 🚀


### Automated Update - Sun Jun 29 12:16:35 UTC 2025 🚀


### Automated Update - Mon Jun 30 00:52:46 UTC 2025 🚀


### Automated Update - Mon Jun 30 12:18:47 UTC 2025 🚀


### Automated Update - Tue Jul  1 00:55:18 UTC 2025 🚀


### Automated Update - Tue Jul  1 12:18:56 UTC 2025 🚀


### Automated Update - Wed Jul  2 00:48:46 UTC 2025 🚀


### Automated Update - Wed Jul  2 12:18:37 UTC 2025 🚀


### Automated Update - Thu Jul  3 00:48:01 UTC 2025 🚀


### Automated Update - Thu Jul  3 12:18:54 UTC 2025 🚀


### Automated Update - Fri Jul  4 00:47:59 UTC 2025 🚀


### Automated Update - Fri Jul  4 12:18:23 UTC 2025 🚀


### Automated Update - Sat Jul  5 00:45:27 UTC 2025 🚀


### Automated Update - Sat Jul  5 12:16:13 UTC 2025 🚀


### Automated Update - Sun Jul  6 00:54:01 UTC 2025 🚀


### Automated Update - Sun Jul  6 12:16:54 UTC 2025 🚀


### Automated Update - Mon Jul  7 00:53:01 UTC 2025 🚀


### Automated Update - Mon Jul  7 12:18:55 UTC 2025 🚀


### Automated Update - Tue Jul  8 00:48:37 UTC 2025 🚀


### Automated Update - Tue Jul  8 12:19:40 UTC 2025 🚀


### Automated Update - Wed Jul  9 00:50:04 UTC 2025 🚀


### Automated Update - Wed Jul  9 12:19:14 UTC 2025 🚀


### Automated Update - Thu Jul 10 00:49:36 UTC 2025 🚀


### Automated Update - Thu Jul 10 12:19:07 UTC 2025 🚀


### Automated Update - Fri Jul 11 00:50:22 UTC 2025 🚀


### Automated Update - Fri Jul 11 12:18:36 UTC 2025 🚀


### Automated Update - Sat Jul 12 00:51:32 UTC 2025 🚀


### Automated Update - Sat Jul 12 12:16:50 UTC 2025 🚀


### Automated Update - Sun Jul 13 00:55:49 UTC 2025 🚀


### Automated Update - Sun Jul 13 12:17:12 UTC 2025 🚀


### Automated Update - Mon Jul 14 00:53:37 UTC 2025 🚀


### Automated Update - Mon Jul 14 12:19:27 UTC 2025 🚀


### Automated Update - Tue Jul 15 00:52:40 UTC 2025 🚀


### Automated Update - Tue Jul 15 12:19:57 UTC 2025 🚀


### Automated Update - Wed Jul 16 00:51:31 UTC 2025 🚀


### Automated Update - Wed Jul 16 12:19:48 UTC 2025 🚀


### Automated Update - Thu Jul 17 00:51:58 UTC 2025 🚀


### Automated Update - Thu Jul 17 12:19:51 UTC 2025 🚀


### Automated Update - Fri Jul 18 00:51:18 UTC 2025 🚀


### Automated Update - Fri Jul 18 12:20:04 UTC 2025 🚀


### Automated Update - Sat Jul 19 00:49:28 UTC 2025 🚀


### Automated Update - Sat Jul 19 12:17:04 UTC 2025 🚀


### Automated Update - Sun Jul 20 00:57:18 UTC 2025 🚀


### Automated Update - Sun Jul 20 12:17:36 UTC 2025 🚀


### Automated Update - Mon Jul 21 00:54:58 UTC 2025 🚀


### Automated Update - Mon Jul 21 12:20:28 UTC 2025 🚀


### Automated Update - Tue Jul 22 00:52:12 UTC 2025 🚀


### Automated Update - Tue Jul 22 12:20:16 UTC 2025 🚀


### Automated Update - Wed Jul 23 00:52:11 UTC 2025 🚀


### Automated Update - Wed Jul 23 12:19:59 UTC 2025 🚀


### Automated Update - Thu Jul 24 00:51:50 UTC 2025 🚀


### Automated Update - Thu Jul 24 12:20:10 UTC 2025 🚀


### Automated Update - Fri Jul 25 00:51:38 UTC 2025 🚀


### Automated Update - Fri Jul 25 12:19:22 UTC 2025 🚀


### Automated Update - Sat Jul 26 00:50:02 UTC 2025 🚀


### Automated Update - Sat Jul 26 12:17:17 UTC 2025 🚀


### Automated Update - Sun Jul 27 00:56:54 UTC 2025 🚀


### Automated Update - Sun Jul 27 12:18:03 UTC 2025 🚀


### Automated Update - Mon Jul 28 00:55:45 UTC 2025 🚀


### Automated Update - Mon Jul 28 12:20:11 UTC 2025 🚀


### Automated Update - Tue Jul 29 00:57:35 UTC 2025 🚀


### Automated Update - Tue Jul 29 12:20:42 UTC 2025 🚀


### Automated Update - Wed Jul 30 00:52:38 UTC 2025 🚀


### Automated Update - Wed Jul 30 12:20:41 UTC 2025 🚀


### Automated Update - Thu Jul 31 00:53:08 UTC 2025 🚀


### Automated Update - Thu Jul 31 12:18:42 UTC 2025 🚀


### Automated Update - Fri Aug  1 00:58:42 UTC 2025 🚀


### Automated Update - Fri Aug  1 12:19:53 UTC 2025 🚀


### Automated Update - Sat Aug  2 00:49:46 UTC 2025 🚀


### Automated Update - Sat Aug  2 12:18:13 UTC 2025 🚀


### Automated Update - Sun Aug  3 00:58:02 UTC 2025 🚀


### Automated Update - Sun Aug  3 12:18:23 UTC 2025 🚀


### Automated Update - Mon Aug  4 00:57:34 UTC 2025 🚀


### Automated Update - Mon Aug  4 12:20:46 UTC 2025 🚀


### Automated Update - Tue Aug  5 00:53:57 UTC 2025 🚀


### Automated Update - Tue Aug  5 12:21:22 UTC 2025 🚀


### Automated Update - Wed Aug  6 00:53:35 UTC 2025 🚀


### Automated Update - Wed Aug  6 12:21:11 UTC 2025 🚀


### Automated Update - Thu Aug  7 00:53:35 UTC 2025 🚀


### Automated Update - Thu Aug  7 12:21:25 UTC 2025 🚀


### Automated Update - Fri Aug  8 00:53:07 UTC 2025 🚀


### Automated Update - Fri Aug  8 12:20:45 UTC 2025 🚀


### Automated Update - Sat Aug  9 00:47:15 UTC 2025 🚀


### Automated Update - Sat Aug  9 12:17:24 UTC 2025 🚀


### Automated Update - Sun Aug 10 00:56:04 UTC 2025 🚀


### Automated Update - Sun Aug 10 12:18:14 UTC 2025 🚀


### Automated Update - Mon Aug 11 00:54:11 UTC 2025 🚀


### Automated Update - Mon Aug 11 12:20:16 UTC 2025 🚀


### Automated Update - Tue Aug 12 00:48:10 UTC 2025 🚀


### Automated Update - Tue Aug 12 12:19:33 UTC 2025 🚀


### Automated Update - Wed Aug 13 00:49:07 UTC 2025 🚀


### Automated Update - Wed Aug 13 12:19:32 UTC 2025 🚀


### Automated Update - Thu Aug 14 00:49:09 UTC 2025 🚀


### Automated Update - Thu Aug 14 12:19:56 UTC 2025 🚀


### Automated Update - Fri Aug 15 00:49:56 UTC 2025 🚀


### Automated Update - Fri Aug 15 12:18:31 UTC 2025 🚀


### Automated Update - Sat Aug 16 00:45:23 UTC 2025 🚀


### Automated Update - Sat Aug 16 12:16:42 UTC 2025 🚀


### Automated Update - Sun Aug 17 00:53:51 UTC 2025 🚀


### Automated Update - Sun Aug 17 12:17:12 UTC 2025 🚀


### Automated Update - Mon Aug 18 00:53:14 UTC 2025 🚀


### Automated Update - Mon Aug 18 12:19:49 UTC 2025 🚀


### Automated Update - Tue Aug 19 00:47:06 UTC 2025 🚀


### Automated Update - Tue Aug 19 12:19:00 UTC 2025 🚀


### Automated Update - Wed Aug 20 00:44:41 UTC 2025 🚀


### Automated Update - Wed Aug 20 12:18:37 UTC 2025 🚀


### Automated Update - Thu Aug 21 00:43:49 UTC 2025 🚀


### Automated Update - Thu Aug 21 12:18:32 UTC 2025 🚀


### Automated Update - Fri Aug 22 00:45:43 UTC 2025 🚀


### Automated Update - Fri Aug 22 12:18:17 UTC 2025 🚀


### Automated Update - Sat Aug 23 00:43:13 UTC 2025 🚀


### Automated Update - Sat Aug 23 12:15:50 UTC 2025 🚀


### Automated Update - Sun Aug 24 00:52:16 UTC 2025 🚀


### Automated Update - Sun Aug 24 12:16:35 UTC 2025 🚀


### Automated Update - Mon Aug 25 00:48:01 UTC 2025 🚀


### Automated Update - Mon Aug 25 12:18:50 UTC 2025 🚀


### Automated Update - Tue Aug 26 00:45:30 UTC 2025 🚀


### Automated Update - Tue Aug 26 12:19:24 UTC 2025 🚀


### Automated Update - Wed Aug 27 00:45:04 UTC 2025 🚀


### Automated Update - Wed Aug 27 12:18:33 UTC 2025 🚀


### Automated Update - Thu Aug 28 00:43:58 UTC 2025 🚀


### Automated Update - Thu Aug 28 12:18:10 UTC 2025 🚀


### Automated Update - Fri Aug 29 00:44:58 UTC 2025 🚀


### Automated Update - Fri Aug 29 12:17:27 UTC 2025 🚀


### Automated Update - Sat Aug 30 00:41:54 UTC 2025 🚀


### Automated Update - Sat Aug 30 12:16:01 UTC 2025 🚀


### Automated Update - Sun Aug 31 00:47:51 UTC 2025 🚀


### Automated Update - Sun Aug 31 12:15:52 UTC 2025 🚀


### Automated Update - Mon Sep  1 00:54:57 UTC 2025 🚀


### Automated Update - Mon Sep  1 12:18:42 UTC 2025 🚀


### Automated Update - Tue Sep  2 00:44:27 UTC 2025 🚀


### Automated Update - Tue Sep  2 12:18:27 UTC 2025 🚀


### Automated Update - Wed Sep  3 00:41:29 UTC 2025 🚀


### Automated Update - Wed Sep  3 12:18:16 UTC 2025 🚀


### Automated Update - Thu Sep  4 00:42:26 UTC 2025 🚀


### Automated Update - Thu Sep  4 12:18:15 UTC 2025 🚀


### Automated Update - Fri Sep  5 00:42:57 UTC 2025 🚀


### Automated Update - Fri Sep  5 12:16:58 UTC 2025 🚀


### Automated Update - Sat Sep  6 00:41:56 UTC 2025 🚀


### Automated Update - Sat Sep  6 12:15:24 UTC 2025 🚀


### Automated Update - Sun Sep  7 00:47:50 UTC 2025 🚀


### Automated Update - Sun Sep  7 12:15:31 UTC 2025 🚀


### Automated Update - Mon Sep  8 00:46:17 UTC 2025 🚀


### Automated Update - Mon Sep  8 12:19:46 UTC 2025 🚀


### Automated Update - Tue Sep  9 00:43:21 UTC 2025 🚀


### Automated Update - Tue Sep  9 12:19:52 UTC 2025 🚀


### Automated Update - Wed Sep 10 00:42:54 UTC 2025 🚀


### Automated Update - Wed Sep 10 12:18:07 UTC 2025 🚀


### Automated Update - Thu Sep 11 00:43:21 UTC 2025 🚀


### Automated Update - Thu Sep 11 12:17:33 UTC 2025 🚀


### Automated Update - Fri Sep 12 00:42:12 UTC 2025 🚀


### Automated Update - Fri Sep 12 12:18:16 UTC 2025 🚀


### Automated Update - Sat Sep 13 00:40:08 UTC 2025 🚀


### Automated Update - Sat Sep 13 12:15:09 UTC 2025 🚀


### Automated Update - Sun Sep 14 00:46:05 UTC 2025 🚀


### Automated Update - Sun Sep 14 12:15:32 UTC 2025 🚀


### Automated Update - Mon Sep 15 00:46:30 UTC 2025 🚀


### Automated Update - Mon Sep 15 12:18:31 UTC 2025 🚀


### Automated Update - Tue Sep 16 00:42:08 UTC 2025 🚀


### Automated Update - Tue Sep 16 12:17:58 UTC 2025 🚀


### Automated Update - Wed Sep 17 00:42:22 UTC 2025 🚀


### Automated Update - Wed Sep 17 12:18:26 UTC 2025 🚀


### Automated Update - Thu Sep 18 00:41:43 UTC 2025 🚀


### Automated Update - Thu Sep 18 12:17:31 UTC 2025 🚀


### Automated Update - Fri Sep 19 00:43:56 UTC 2025 🚀


### Automated Update - Fri Sep 19 12:18:06 UTC 2025 🚀


### Automated Update - Sat Sep 20 00:41:19 UTC 2025 🚀


### Automated Update - Sat Sep 20 12:16:06 UTC 2025 🚀


### Automated Update - Sun Sep 21 00:48:15 UTC 2025 🚀


### Automated Update - Sun Sep 21 12:15:40 UTC 2025 🚀


### Automated Update - Mon Sep 22 00:47:03 UTC 2025 🚀


### Automated Update - Mon Sep 22 12:19:00 UTC 2025 🚀


### Automated Update - Tue Sep 23 00:42:49 UTC 2025 🚀


### Automated Update - Tue Sep 23 12:18:07 UTC 2025 🚀


### Automated Update - Wed Sep 24 00:43:23 UTC 2025 🚀


### Automated Update - Wed Sep 24 12:18:40 UTC 2025 🚀


### Automated Update - Thu Sep 25 00:43:28 UTC 2025 🚀


### Automated Update - Thu Sep 25 12:19:12 UTC 2025 🚀


### Automated Update - Fri Sep 26 00:42:51 UTC 2025 🚀


### Automated Update - Fri Sep 26 12:18:23 UTC 2025 🚀


### Automated Update - Sat Sep 27 00:41:29 UTC 2025 🚀


### Automated Update - Sat Sep 27 12:15:52 UTC 2025 🚀


### Automated Update - Sun Sep 28 00:48:48 UTC 2025 🚀


### Automated Update - Sun Sep 28 12:15:41 UTC 2025 🚀


### Automated Update - Mon Sep 29 00:45:15 UTC 2025 🚀


### Automated Update - Mon Sep 29 12:18:52 UTC 2025 🚀


### Automated Update - Tue Sep 30 00:43:49 UTC 2025 🚀


### Automated Update - Tue Sep 30 12:19:11 UTC 2025 🚀


### Automated Update - Wed Oct  1 00:50:49 UTC 2025 🚀


### Automated Update - Wed Oct  1 12:18:54 UTC 2025 🚀


### Automated Update - Thu Oct  2 00:42:30 UTC 2025 🚀


### Automated Update - Thu Oct  2 12:17:14 UTC 2025 🚀


### Automated Update - Fri Oct  3 00:42:18 UTC 2025 🚀


### Automated Update - Fri Oct  3 12:17:14 UTC 2025 🚀


### Automated Update - Sat Oct  4 00:40:12 UTC 2025 🚀


### Automated Update - Sat Oct  4 12:15:05 UTC 2025 🚀


### Automated Update - Sun Oct  5 00:48:09 UTC 2025 🚀


### Automated Update - Sun Oct  5 12:15:22 UTC 2025 🚀


### Automated Update - Mon Oct  6 00:44:12 UTC 2025 🚀


### Automated Update - Mon Oct  6 12:18:17 UTC 2025 🚀


### Automated Update - Tue Oct  7 00:43:09 UTC 2025 🚀


### Automated Update - Tue Oct  7 12:18:53 UTC 2025 🚀


### Automated Update - Wed Oct  8 00:42:30 UTC 2025 🚀


### Automated Update - Wed Oct  8 12:19:07 UTC 2025 🚀


### Automated Update - Thu Oct  9 00:43:24 UTC 2025 🚀


### Automated Update - Thu Oct  9 12:18:18 UTC 2025 🚀


### Automated Update - Fri Oct 10 00:43:18 UTC 2025 🚀


### Automated Update - Fri Oct 10 12:19:19 UTC 2025 🚀


### Automated Update - Sat Oct 11 00:40:49 UTC 2025 🚀


### Automated Update - Sat Oct 11 12:15:48 UTC 2025 🚀


### Automated Update - Sun Oct 12 00:45:40 UTC 2025 🚀


### Automated Update - Sun Oct 12 12:15:55 UTC 2025 🚀


### Automated Update - Mon Oct 13 00:47:02 UTC 2025 🚀


### Automated Update - Mon Oct 13 12:18:49 UTC 2025 🚀


### Automated Update - Tue Oct 14 00:43:06 UTC 2025 🚀


### Automated Update - Tue Oct 14 12:19:41 UTC 2025 🚀


### Automated Update - Wed Oct 15 00:45:04 UTC 2025 🚀


### Automated Update - Wed Oct 15 12:20:00 UTC 2025 🚀


### Automated Update - Thu Oct 16 00:44:50 UTC 2025 🚀


### Automated Update - Thu Oct 16 12:19:06 UTC 2025 🚀


### Automated Update - Fri Oct 17 00:43:49 UTC 2025 🚀


### Automated Update - Fri Oct 17 12:18:23 UTC 2025 🚀


### Automated Update - Sat Oct 18 00:41:28 UTC 2025 🚀


### Automated Update - Sat Oct 18 12:16:16 UTC 2025 🚀


### Automated Update - Sun Oct 19 00:50:17 UTC 2025 🚀


### Automated Update - Sun Oct 19 12:16:22 UTC 2025 🚀


### Automated Update - Mon Oct 20 00:49:05 UTC 2025 🚀


### Automated Update - Mon Oct 20 12:18:52 UTC 2025 🚀


### Automated Update - Tue Oct 21 00:45:00 UTC 2025 🚀


### Automated Update - Tue Oct 21 12:19:24 UTC 2025 🚀


### Automated Update - Wed Oct 22 00:46:26 UTC 2025 🚀


### Automated Update - Wed Oct 22 12:19:12 UTC 2025 🚀


### Automated Update - Thu Oct 23 00:45:03 UTC 2025 🚀


### Automated Update - Thu Oct 23 12:19:34 UTC 2025 🚀


### Automated Update - Fri Oct 24 00:42:13 UTC 2025 🚀


### Automated Update - Fri Oct 24 12:19:33 UTC 2025 🚀


### Automated Update - Sat Oct 25 00:43:23 UTC 2025 🚀


### Automated Update - Sat Oct 25 12:15:54 UTC 2025 🚀


### Automated Update - Sun Oct 26 00:48:22 UTC 2025 🚀


### Automated Update - Sun Oct 26 12:16:47 UTC 2025 🚀


### Automated Update - Mon Oct 27 00:50:51 UTC 2025 🚀


### Automated Update - Mon Oct 27 12:19:19 UTC 2025 🚀


### Automated Update - Tue Oct 28 00:43:43 UTC 2025 🚀


### Automated Update - Tue Oct 28 12:18:38 UTC 2025 🚀


### Automated Update - Wed Oct 29 00:47:42 UTC 2025 🚀


### Automated Update - Wed Oct 29 12:19:26 UTC 2025 🚀


### Automated Update - Thu Oct 30 00:47:04 UTC 2025 🚀


### Automated Update - Thu Oct 30 12:19:17 UTC 2025 🚀


### Automated Update - Fri Oct 31 00:44:56 UTC 2025 🚀


### Automated Update - Fri Oct 31 12:19:20 UTC 2025 🚀


### Automated Update - Sat Nov  1 00:49:20 UTC 2025 🚀


### Automated Update - Sat Nov  1 12:16:00 UTC 2025 🚀


### Automated Update - Sun Nov  2 00:49:56 UTC 2025 🚀


### Automated Update - Sun Nov  2 12:16:07 UTC 2025 🚀


### Automated Update - Mon Nov  3 00:49:30 UTC 2025 🚀


### Automated Update - Mon Nov  3 12:19:25 UTC 2025 🚀


### Automated Update - Tue Nov  4 00:45:31 UTC 2025 🚀


### Automated Update - Tue Nov  4 12:20:05 UTC 2025 🚀


### Automated Update - Wed Nov  5 00:47:59 UTC 2025 🚀


### Automated Update - Wed Nov  5 12:19:12 UTC 2025 🚀


### Automated Update - Thu Nov  6 00:46:29 UTC 2025 🚀


### Automated Update - Thu Nov  6 12:19:26 UTC 2025 🚀


### Automated Update - Fri Nov  7 00:46:05 UTC 2025 🚀


### Automated Update - Fri Nov  7 12:18:33 UTC 2025 🚀


### Automated Update - Sat Nov  8 00:43:50 UTC 2025 🚀


### Automated Update - Sat Nov  8 12:16:46 UTC 2025 🚀


### Automated Update - Sun Nov  9 00:49:45 UTC 2025 🚀


### Automated Update - Sun Nov  9 12:16:08 UTC 2025 🚀


### Automated Update - Mon Nov 10 00:49:55 UTC 2025 🚀


### Automated Update - Mon Nov 10 12:19:18 UTC 2025 🚀


### Automated Update - Tue Nov 11 00:47:46 UTC 2025 🚀


### Automated Update - Tue Nov 11 12:19:03 UTC 2025 🚀


### Automated Update - Wed Nov 12 00:47:14 UTC 2025 🚀


### Automated Update - Wed Nov 12 12:19:36 UTC 2025 🚀


### Automated Update - Thu Nov 13 00:47:19 UTC 2025 🚀


### Automated Update - Thu Nov 13 12:19:38 UTC 2025 🚀


### Automated Update - Fri Nov 14 00:46:54 UTC 2025 🚀


### Automated Update - Fri Nov 14 12:19:34 UTC 2025 🚀


### Automated Update - Sat Nov 15 00:45:09 UTC 2025 🚀


### Automated Update - Sat Nov 15 12:16:51 UTC 2025 🚀


### Automated Update - Sun Nov 16 00:51:06 UTC 2025 🚀


### Automated Update - Sun Nov 16 12:17:04 UTC 2025 🚀


### Automated Update - Mon Nov 17 00:48:43 UTC 2025 🚀


### Automated Update - Mon Nov 17 12:19:19 UTC 2025 🚀


### Automated Update - Tue Nov 18 00:45:50 UTC 2025 🚀


### Automated Update - Tue Nov 18 12:20:07 UTC 2025 🚀


### Automated Update - Wed Nov 19 00:47:01 UTC 2025 🚀


### Automated Update - Wed Nov 19 12:19:18 UTC 2025 🚀


### Automated Update - Thu Nov 20 00:45:41 UTC 2025 🚀


### Automated Update - Thu Nov 20 12:19:24 UTC 2025 🚀


### Automated Update - Fri Nov 21 00:46:02 UTC 2025 🚀


### Automated Update - Fri Nov 21 12:18:28 UTC 2025 🚀


### Automated Update - Sat Nov 22 00:44:19 UTC 2025 🚀


### Automated Update - Sat Nov 22 12:16:35 UTC 2025 🚀


### Automated Update - Sun Nov 23 00:54:21 UTC 2025 🚀


### Automated Update - Sun Nov 23 12:15:59 UTC 2025 🚀


### Automated Update - Mon Nov 24 00:51:27 UTC 2025 🚀


### Automated Update - Mon Nov 24 12:19:34 UTC 2025 🚀


### Automated Update - Tue Nov 25 00:45:21 UTC 2025 🚀


### Automated Update - Tue Nov 25 12:20:28 UTC 2025 🚀


### Automated Update - Wed Nov 26 00:47:21 UTC 2025 🚀


### Automated Update - Wed Nov 26 12:20:49 UTC 2025 🚀


### Automated Update - Thu Nov 27 00:45:52 UTC 2025 🚀


### Automated Update - Thu Nov 27 12:20:16 UTC 2025 🚀


### Automated Update - Fri Nov 28 00:45:09 UTC 2025 🚀


### Automated Update - Fri Nov 28 12:19:24 UTC 2025 🚀


### Automated Update - Sat Nov 29 00:45:12 UTC 2025 🚀


### Automated Update - Sat Nov 29 12:16:43 UTC 2025 🚀


### Automated Update - Sun Nov 30 00:54:31 UTC 2025 🚀


### Automated Update - Sun Nov 30 12:17:21 UTC 2025 🚀


### Automated Update - Mon Dec  1 00:58:18 UTC 2025 🚀


### Automated Update - Mon Dec  1 12:20:54 UTC 2025 🚀
