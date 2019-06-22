def num_patches(output_img_dim=(224, 224, 4), sub_patch_dim=(56, 56)):
    """
    Creates non-overlaping patches to feed to the PATCH GAN
    (Section 2.2.2 in paper)
    The paper provides 3 options.
    Pixel GAN = 1x1 patches (aka each pixel)
    PatchGAN = nxn patches (non-overlaping blocks of the image)
    ImageGAN = im_size x im_size (full image)

    Ex: 4x4 image with patch_size of 2 means 4 non-overlaping patches

    :param output_img_dim:
    :param sub_patch_dim:
    :return:
    """
    # num of non-overlaping patches
    nb_non_overlaping_patches = (output_img_dim[0] / sub_patch_dim[0]) * (output_img_dim[1] / sub_patch_dim[1])

    # dimensions for the patch discriminator
    patch_disc_img_dim = ( sub_patch_dim[0], sub_patch_dim[1], output_img_dim[2])

    return int(nb_non_overlaping_patches), patch_disc_img_dim