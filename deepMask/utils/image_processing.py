import ants
import os
import logging
import time
# import matplotlib as mpl
# mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
# import zipfile
if os.environ.get("BRAIN_MASKING") == "cpu":
    from antspynet.utilities import brain_extraction
from .deepmask import *
from .helpers import *
from matplotlib.backends.backend_pdf import PdfPages
# read pngs to save as pdf
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logfile = './logs.log'
logfile = os.path.join("/tmp", str(random_case_id()) + ".log")
# create a file handler
try:
    os.remove(logfile)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

handler = logging.FileHandler(logfile)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())
os.environ["ANTS_RANDOM_SEED"] = "666"


class noelImageProcessor:
    def __init__(
        self,
        id,
        t1=None,
        t2=None,
        output_suffix="_brain_final.nii.gz",
        output_dir=None,
        template=None,
        transform="Affine",
        usen3=False,
        args=None,
        model=None,
        QC=None,
        preprocess=True,
    ):
        super(noelImageProcessor, self).__init__()
        self._id = id
        self._t1file = t1
        self._t2file = t2
        self._outsuffix = output_suffix
        self._outputdir = output_dir
        self._template = template
        self._transform = transform
        self._usen3 = usen3
        self._args = args
        self._model = model
        self._QC = QC
        self._preprocess = preprocess
        self._dpi = 300

    def __load_nifti_file(self):
        # load nifti data to memory
        logger.info("loading nifti files")
        print("loading nifti files")
        self._mni = self._template
        if self._t1file == None and self._t2file == None:
            logger.warn("Please load the data first", "The data is invalid/missing")
            return

        if self._t1file != None and self._t2file != None:
            self._t1 = ants.image_read(self._t1file)
            self._t2 = ants.image_read(self._t2file)
            self._icbm152 = ants.image_read(self._mni)

    def __register_to_MNI_space(self):
        logger.info("registration to MNI template space")
        print("registration to MNI template space")
        if self._t1file != None and self._t2file != None:
            self._t1_reg = ants.registration(
                fixed=self._icbm152,
                moving=self._t1,
                type_of_transform=self._transform,
            )
            self._t2_reg = ants.registration(
                fixed=self._t1_reg["warpedmovout"],
                moving=self._t2,
                type_of_transform=self._transform,
            )
            # create directory to store transforms
            xfmdir = os.path.join(self._outputdir, "transforms")
            if not os.path.exists(xfmdir):
                os.makedirs(xfmdir)
            # write forward transforms to xfmdir
            ants.write_transform(
                ants.read_transform(self._t1_reg["fwdtransforms"][0]),
                os.path.join(xfmdir, self._id + "_t1-native-to-MNI152.mat"),
            )
            ants.write_transform(
                ants.read_transform(self._t2_reg["fwdtransforms"][0]),
                os.path.join(xfmdir, self._id + "_t2-native-to-MNI152.mat"),
            )
            # self._t2_reg = ants.apply_transforms(fixed = self._t1_reg['warpedmovout'], moving = self._t2, transformlist = self._t1_reg['fwdtransforms'])
            # ants.image_write( self._t1_reg['warpedmovout'], self._t1regfile)
            # ants.image_write( self._t2_reg, self._t2regfile)

    def __bias_correction(self):
        logger.info(
            "performing {} bias correction".format("N3" if self._usen3 else "N4")
        )
        print("performing {} bias correction".format("N3" if self._usen3 else "N4"))
        if self._t1file != None and self._t2file != None:
            if self._usen3:
                self._t1_n4 = (
                    ants.iMath(
                        ants.n3_bias_field_correction(
                            self._t1_reg["warpedmovout"], downsample_factor=4
                        ),
                        "Normalize",
                    )
                    * 100
                )
                self._t2_n4 = (
                    ants.iMath(
                        ants.n3_bias_field_correction(
                            self._t2_reg["warpedmovout"], downsample_factor=4
                        ),
                        "Normalize",
                    )
                    * 100
                )
            else:
                self._t1_n4 = (
                    ants.iMath(
                        ants.n4_bias_field_correction(self._t1_reg["warpedmovout"]),
                        "Normalize",
                    )
                    * 100
                )
                self._t2_n4 = (
                    ants.iMath(
                        ants.n4_bias_field_correction(self._t2_reg["warpedmovout"]),
                        "Normalize",
                    )
                    * 100
                )
            self._t1regfile = os.path.join(
                self._outputdir, self._id + "_t1_final.nii.gz"
            )
            self._t2regfile = os.path.join(
                self._outputdir, self._id + "_t2_final.nii.gz"
            )
            ants.image_write(self._t1_n4, self._t1regfile)
            ants.image_write(self._t2_n4, self._t2regfile)

    def __skull_stripping(self):
        # specify the output filenames for brain extracted images
        self._t1brainfile = os.path.join(
            self._outputdir, self._id + "_t1" + self._outsuffix
        )
        self._t2brainfile = os.path.join(
            self._outputdir, self._id + "_t2" + self._outsuffix
        )
        if os.environ.get("BRAIN_MASKING") == "cpu":
            logger.info("performing brain extraction using ANTsPyNet")
            print("performing brain extraction using ANTsPyNet")
            if self._preprocess:
                prob = brain_extraction(self._t1_n4, modality="t1")
                # mask can be obtained as:
                mask = ants.threshold_image(
                    prob,
                    low_thresh=0.6,
                    high_thresh=1.0,
                    inval=1,
                    outval=0,
                    binary=True,
                )
                self._mask = self._t1_n4.new_image_like(mask.numpy())
                ants.image_write(self._t1_n4 * self._mask, self._t1brainfile)
                ants.image_write(self._t2_n4 * self._mask, self._t2brainfile)
            else:
                prob = brain_extraction(self._t1, modality="t1")
                # mask can be obtained as:
                mask = ants.threshold_image(
                    prob,
                    low_thresh=0.6,
                    high_thresh=1.0,
                    inval=1,
                    outval=0,
                    binary=True,
                )
                self._mask = self._t1.new_image_like(mask.numpy())
                # extract the brain and normalize between 0 and 100
                ants.image_write(
                    ants.iMath(self._t1 * self._mask, "Normalize") * 100,
                    self._t1brainfile,
                )
                ants.image_write(
                    ants.iMath(self._t2 * self._mask, "Normalize") * 100,
                    self._t2brainfile,
                )
        else:
            logger.info("performing brain extraction using deepMask")
            print("performing brain extraction using deepMask")
            if self._t1file != None and self._t2file != None:
                if self._preprocess:
                    mask = deepMask(
                        self._args,
                        self._model,
                        self._id,
                        self._t1_n4.numpy(),
                        self._t2_n4.numpy(),
                        self._t1regfile,
                        self._t2regfile,
                    )
                    self._mask = self._t1_n4.new_image_like(mask)
                    ants.image_write(self._t1_n4 * self._mask, self._t1brainfile)
                    ants.image_write(self._t2_n4 * self._mask, self._t2brainfile)
                else:
                    mask = deepMask(
                        self._args,
                        self._model,
                        self._id,
                        self._t1.numpy(),
                        self._t2.numpy(),
                        self._t1file,
                        self._t2file,
                    )
                    self._mask = self._t1.new_image_like(mask)
                    ants.image_write(self._t1 * self._mask, self._t1brainfile)
                    ants.image_write(self._t2 * self._mask, self._t2brainfile)

    def __apply_transforms(self):
        logger.info(
            "apply transforms to project outputs back to the native input space"
        )
        print("apply transforms to project outputs back to the native input space")
        self._t1_native = apply_transform(
            self._mask, self._t1, self._t1_reg["fwdtransforms"][0], invert_xfrm=True
        )
        self._t2_native = apply_transform(
            self._mask, self._t2, self._t2_reg["fwdtransforms"][0], invert_xfrm=True
        )

        mask_suffix = "_brain_mask_native.nii.gz"
        # write skull-stripped versions of the brain mask in native space
        ants.image_write(
            self._t1_native, self._t1brainfile.replace(self._outsuffix, mask_suffix)
        )
        ants.image_write(
            self._t2_native, self._t2brainfile.replace(self._outsuffix, mask_suffix)
        )

    def __generate_QC_maps(self):
        logger.info("generating QC report")
        qcdir = os.path.join(self._args.tmpdir, "qc")
        if not os.path.exists(qcdir):
            os.makedirs(qcdir)
        if self._t1file != None and self._t2file != None:
            self._icbm152.plot(
                overlay=self._t1,
                overlay_alpha=0.5,
                axis=2,
                ncol=8,
                nslices=32,
                title="T1w - Before Registration",
                filename=os.path.join(qcdir, "000_t1_before_registration.png"),
                dpi=self._dpi,
            )
            self._icbm152.plot(
                overlay=self._t1_reg["warpedmovout"],
                overlay_alpha=0.5,
                axis=2,
                ncol=8,
                nslices=32,
                title="T1w - After Registration",
                filename=os.path.join(qcdir, "001_t1_after_registration.png"),
                dpi=self._dpi,
            )
            self._icbm152.plot(
                overlay=self._t2,
                overlay_alpha=0.5,
                axis=2,
                ncol=8,
                nslices=32,
                title="T2w - Before Registration",
                filename=os.path.join(qcdir, "002_t2_before_registration.png"),
                dpi=self._dpi,
            )
            self._icbm152.plot(
                overlay=self._t2_reg,
                overlay_alpha=0.5,
                axis=2,
                ncol=8,
                nslices=32,
                title="T2w - After Registration",
                filename=os.path.join(qcdir, "003_t2_after_registration.png"),
                dpi=self._dpi,
            )

            ants.plot(
                self._t1_reg["warpedmovout"],
                axis=2,
                ncol=8,
                nslices=32,
                cmap="jet",
                title="T1w - Before Bias Correction",
                filename=os.path.join(qcdir, "004_t1_before_bias_correction.png"),
                dpi=self._dpi,
            )
            ants.plot(
                self._t1_n4,
                axis=2,
                ncol=8,
                nslices=32,
                cmap="jet",
                title="T1w - After Bias Correction",
                filename=os.path.join(qcdir, "005_t1_after_bias_correction.png"),
                dpi=self._dpi,
            )
            ants.plot(
                self._t2_reg,
                axis=2,
                ncol=8,
                nslices=32,
                cmap="jet",
                title="T2w - Before Bias Correction",
                filename=os.path.join(qcdir, "006_t2_before_bias_correction.png"),
                dpi=self._dpi,
            )
            ants.plot(
                self._t2_n4,
                axis=2,
                ncol=8,
                nslices=32,
                cmap="jet",
                title="T2w - After Bias Correction",
                filename=os.path.join(qcdir, "007_t2_after_bias_correction.png"),
                dpi=self._dpi,
            )

            self._t1_n4.plot(
                overlay=self._mask,
                overlay_alpha=0.5,
                axis=2,
                ncol=8,
                nslices=32,
                title="Brain Masking",
                filename=os.path.join(qcdir, "008_brain_masking.png"),
                dpi=self._dpi,
            )

            with PdfPages(
                os.path.join(self._outputdir, self._id + "_QC_report.pdf")
            ) as pdf:
                for i in sorted(os.listdir(qcdir)):
                    if i.endswith(".png"):
                        plt.figure()
                        img = Image.open(os.path.join(qcdir, i))
                        plt.imshow(img)
                        plt.axis("off")
                        pdf.savefig(dpi=self._dpi)
                        plt.close()
                        os.remove(os.path.join(qcdir, i))

    def __organize_and_cleanup(self):
        logger.info("moving intermediate files to args.tmpdir, reorganizing the rest")
        print("moving intermediate files to args.tmpdir, reorganizing the rest")

        _move_suffix = {
            "_denseCrf3dProbMapClass1.nii.gz",
            "_denseCrf3dProbMapClass0.nii.gz",
            "_vnet_maskpred.nii.gz",
        }
        _rename_suffix = "_denseCrf3dSegmMap.nii.gz"
        # _final_suffix = "_final.nii.gz"
        _native_suffix = "_native.nii.gz"

        for file in os.listdir(self._outputdir):
            if file.endswith(_rename_suffix):
                src = os.path.join(self._outputdir, file)
                dst = os.path.join(
                    self._outputdir,
                    file.replace(_rename_suffix, "_brain_mask_final.nii.gz"),
                )
                os.renames(src, dst)
            if file.endswith(_native_suffix):
                src = os.path.join(self._outputdir, file)
                dst = os.path.join(self._outputdir, "native", file)
                os.renames(src, dst)
            # if file.endswith(_final_suffix):
            #     src = os.path.join(self._outputdir, file)
            #     dst = os.path.join(self._outputdir, "final", file)
            #     os.renames(src, dst)
            for _suffix in _move_suffix:
                if file.endswith(_suffix):
                    src = os.path.join(self._outputdir, file)
                    dst = os.path.join(self._args.tmpdir, "deepMask", file)
                    os.renames(src, dst)

    # def __create_zip_archive(self):
    #     print("creating a zip archive")
    #     logger.info("creating a zip archive")
    #     zip_archive = zipfile.ZipFile(
    #         os.path.join(self._outputdir, self._id + "_archive.zip"), "w"
    #     )
    #     for folder, _, files in os.walk(self._outputdir):
    #         for file in files:
    #             if file.endswith(".nii.gz"):
    #                 zip_archive.write(
    #                     os.path.join(folder, file),
    #                     file,
    #                     compress_type=zipfile.ZIP_DEFLATED,
    #                 )
    #     zip_archive.close()

    def pipeline(self):
        start = time.time()
        if self._t1file.lower().endswith((".nii.gz", ".nii")):
            self.__load_nifti_file()
        else:
            raise Exception("Sorry, only NIfTI format is currently supported")

        if self._preprocess:
            self.__register_to_MNI_space()
            self.__bias_correction()
            self.__skull_stripping()
            self.__apply_transforms()
        else:
            print(
                "Skipping image preprocessing, presumably images are co-registered and bias-corrected"
            )
            self.__skull_stripping()

        if self._QC:
            self.__generate_QC_maps()
        # self.__create_zip_archive()

        self.__organize_and_cleanup()
        end = time.time()
        print(
            "pipeline processing time elapsed: {} seconds".format(
                np.round(end - start, 1)
            )
        )
        logger.info(
            "pipeline processing time elapsed: {} seconds".format(
                np.round(end - start, 1)
            )
        )
        logger.info("*********************************************")
