# Application Title Goes Here
<!-- If you'd like to use a logo instead uncomment this code and remove the text above this line

  ![Logo](http://blog.ebuystorm.com/aboutme/avatar.png)

-->

Created by: [Jingkun Chen](http://blog.ebuystorm.com/aboutme.html).



## Description
**Adversarial Convolutional Networks with Weak Domain-Transfer for Multi-Sequence Cardiac MR Images Segmentation** Analysis and modeling of the ventricles and myocardium are important in the diagnostic and treatment of heart diseases. Manual delineation of those tissues in cardiac MR (CMR) scans is laborious and time-consuming. The ambiguity of the boundaries makes the segmentation task rather challenging. Furthermore, the annotations on some modalities such as Late Gadolinium Enhancement (LGE) MRI, are often not available. We propose an end-to-end segmentation framework based on convolutional neural network (CNN) and adversarial learning. A dilated residual U-shape network is used as a segmentor to generate the prediction mask; meanwhile, a CNN is utilized as a discriminator model to judge the segmentation quality. To leverage the available annotations across modalities per patient, a new loss function named weak domain-transfer loss is introduced to the pipeline. The proposed model is evaluated on the public dataset released by the challenge organizer in MICCAI 2019, which consists of 45 sets of multi-sequence CMR images. We demonstrate that the proposed adversarial pipeline outperforms baseline deep-learning methods.

## Installation
train data:
train_data.npy train_gt.npy
test data:
test_4_data.npy test_4_gt.npy test_5_data.npy test_5_gt.npy



## Usage

Run the :

```erb
python3 run_train.py train_data.npy train_gt.npy weight test_4_data.npy test_4_gt.npy test_5_data.npy test_5_gt.npy --output_test_eval eval.log --use_cnn True
```


## Configuration

This block of text should explain how to configure your application:

`rails generate my_example_gem:install`


## Information

Screenshots of your application below:

The whole model:
![Screenshot 1](http://blog.ebuystorm.com/file/image/miccai_2019_segmentor.png)

The generator:
![Screenshot 2](http://blog.ebuystorm.com/file/image/miccai_2019_unet_224.png)

The gen loss:
![Screenshot 3](http://blog.ebuystorm.com/file/image/miccai_2019_gen_loss.png)

The adv loss:
![Screenshot 4](http://blog.ebuystorm.com/file/image/miccai_2019_adv_loss.png)

The result:
![Screenshot 5](http://blog.ebuystorm.com/file/image/miccai_2019_result.png)
## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request


## License

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
