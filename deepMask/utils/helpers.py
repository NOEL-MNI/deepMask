import os, random, string
from ants import apply_ants_transform_to_image, image_write, read_transform


def write_nifti(input, id, output_dir, type):
    output_fname = os.path.join(output_dir, id + "_" + type + ".nii.gz")
    image_write(input, output_fname)


def find_logger_basefilename(logger):
    """Finds the logger base filename(s) currently there is only one"""
    log_file = None
    handler = logger.handlers[0]
    log_file = handler.baseFilename
    return log_file


def random_case_id():
    letters = "".join(random.choices(string.ascii_letters, k=16))
    digits = "".join(random.choices(string.digits, k=16))
    x = letters[:3].lower() + "_" + digits[:4]
    return x


def apply_transform(image_to_xfm, reference_image, transform, invert_xfrm=True):
    xfrm = read_transform(transform)
    if invert_xfrm:
        xfrm = xfrm.invert()
    image_xfmd = apply_ants_transform_to_image(
        image=image_to_xfm,
        reference=reference_image,
        transform=xfrm,
        interpolation="nearestneighbor",
    )
    return image_xfmd
