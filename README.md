# Embarrassingly Parallel Image Classification: Inferring Land Use from Aerial Imagery

## Introduction

Deep neural networks (DNNs) are extraordinarily versatile artificial intelligence models that have entered wide use over the last five years. Common use cases for DNNs include:

- Determining whether an uploaded video, audio, or text file contains inappropriate content
- Inferring a user's intent from their spoken or typed input
- Identifying objects or persons in a still image
- Translating speech or text between languages

Unfortunately, DNNs are also among the most time- and resource-intensive machine learning models. Whereas linear regression model results can be computed in negligible time, applying a DNN to a single file of interest may take tens or hundreds of milliseconds -- a processing rate on the order of 1,000 files per minute. Many business use cases require faster throughput. Fortunately, DNNs can be applied in parallel and scalable fashion when evaluation is performed on Spark clusters.

This guide repository demonstrates how trained DNNs produced with two common deep learning frameworks, Microsoft's Cognitive Toolkit and Google's Tensorflow, can be operationalized on Spark to score a large image set. Files stored on Azure Data Lake Store, Microsoft's HDFS-based cloud storage resource, are processed in parallel by workers on the Spark cluster. The guide follows a specific example use case, described below.

## Land use classification from aerial imagery

In this guide, we develop a classifier that can predict how a parcel of land has been used -- developed, cultivated, forested, barren, etc. -- from an aerial image. We apply the classifier to track changes in land use, like urban expansion and deforestation, over time. Classifying aerial images has many other applications in industry and government, including:
- Monitoring crop performance in agriculture
- Quantifying the impact of climate change on natural resources
- Property value estimation and feature tracking for marketing purposes
- Geopolitical surveillance
- Enforcing tax codes (cf. [identification of home pools in Greece](http://www.nytimes.com/2010/05/02/world/europe/02evasion.html))

Sample images and ground-truth labels are fortunately available in abundance for this use case. We use aerial imagery provided by the U.S. [National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/), and land use labels from the [National Land Cover Database](https://www.mrlc.gov/). NLCD labels are published roughly every five years, while NAIP data are collected more frequently: a trained land use classifier can be used to infer land use at all aerial imaging timepoints.

## Model training and validation

We applied transfer learning to retrain the final layers of existing Tensorflow (ResNet) and CNTK (AlexNet) models for classification of 1-meter resolution NAIP aerial images of 224 meter x 224 meter regions selected from across the United States. It is possible for regions of this size to have multiple land uses (NLCD labels are provided at 30-meter resolution): to avoid confusion, we used only regions with relatively homogeneous land use for model training and validation. We created a balanced training and validation sets containing aerial images in six major land use categories (Developed, Cultivated, Forest, Shrub, Barren, and Herbaceous) from non-neighboring counties and collection years. Our retrained models achieved a classification accuracy of ~80% on these six categories, with the majority of errors occurring between different types of undeveloped land. By further grouping our images into "Developed", "Cultivated," and "Undeveloped" classes, we were able to further improve the classification accuracy to roughly 95% in our validation set.

## Model application: predicting land use in 2016

The trained models were applied to aerial images tiling Middlesex County, MA (home of Microsoft's Boston-area office).


## Contributing and Adapting

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

The code in this repository is shared under the [MIT and Apache licenses](./LICENSE) included in this directory. Some Tensorflow scripts have been adapted from the [Tensorflow Models repository's slim](https://github.com/tensorflow/models/tree/master/slim) subdirectory (indicated where applicable). Cognitive Toolkit (CNTK) scripts for network definition and training have been adapted from the [CIFAR-10 Image Classification](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/ResNet/Python) example.