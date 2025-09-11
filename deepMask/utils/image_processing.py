import logging
import multiprocessing
import os
import time

import ants

# import matplotlib as mpl
# mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# import zipfile
if os.environ.get("BRAIN_MASKING") == "cpu":
    from antspynet.utilities import brain_extraction
from matplotlib.backends.backend_pdf import PdfPages

from .deepmask import deepMask
from .helpers import apply_transform, random_case_id

# Import BIDS metadata utilities (try relative, package, and legacy locations)
try:
    from ...utils.bids_metadata import (
        generate_bids_metadata_for_outputs,
        parse_subject_session,
    )
except Exception:
    try:
        from utils.bids_metadata import (
            generate_bids_metadata_for_outputs,
            parse_subject_session,
        )
    except Exception:
        try:
            from bids_metadata import (
                generate_bids_metadata_for_outputs,
                parse_subject_session,
            )
        except Exception:
            print(
                "Warning: Could not import BIDS metadata utilities. Metadata generation will be skipped."
            )
            generate_bids_metadata_for_outputs = None
            parse_subject_session = None
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
    print(f"Error: {e.filename} - {e.strerror}.")

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
        output_suffix="_brain.nii.gz",
        output_dir=None,
        template=None,
        transform="Affine",
        usen3=False,
        args=None,
        model=None,
        QC=None,
        preprocess=True,
    ):
        super().__init__()
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
        if self._t1file is None and self._t2file is None:
            logger.warning("Please load the data first", "The data is invalid/missing")
            return

        if self._t1file is not None and self._t2file is not None:
            self._t1 = ants.image_read(self._t1file)
            self._t2 = ants.image_read(self._t2file)
            self._icbm152 = ants.image_read(self._mni)

    def __register_to_MNI_space(self):
        logger.info("registration to MNI template space")
        print("registration to MNI template space")
        if self._t1file is not None and self._t2file is not None:
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
            # create directory to store transforms in BIDS-compliant location
            # transforms should be at session level in 'xfm' directory, not inside 'anat'
            if "_ses-" in self._id:
                # For sessions: go up from anat/ to session level and create xfm/
                session_dir = os.path.dirname(self._outputdir)  # go up from anat/
                xfmdir = os.path.join(session_dir, "xfm")
            else:
                # For no sessions: go up from anat/ to subject level and create xfm/
                subject_dir = os.path.dirname(self._outputdir)  # go up from anat/
                xfmdir = os.path.join(subject_dir, "xfm")

            if not os.path.exists(xfmdir):
                os.makedirs(xfmdir)
            # write forward transforms to xfmdir with BIDS BEP014 naming
            ants.write_transform(
                ants.read_transform(self._t1_reg["fwdtransforms"][0]),
                os.path.join(
                    xfmdir,
                    self._id + "_from-T1w_to-MNI152_mode-image_xfm.mat",
                ),
            )
            ants.write_transform(
                ants.read_transform(self._t2_reg["fwdtransforms"][0]),
                os.path.join(
                    xfmdir,
                    self._id + "_from-FLAIR_to-MNI152_mode-image_xfm.mat",
                ),
            )
            # self._t2_reg = ants.apply_transforms(fixed = self._t1_reg['warpedmovout'], moving = self._t2, transformlist = self._t1_reg['fwdtransforms'])
            # ants.image_write( self._t1_reg['warpedmovout'], self._t1regfile)
            # ants.image_write( self._t2_reg, self._t2regfile)

    def __bias_correction(self):
        logger.info(
            "performing {} bias correction".format("N3" if self._usen3 else "N4")
        )
        print("performing {} bias correction".format("N3" if self._usen3 else "N4"))
        if self._t1file is not None and self._t2file is not None:
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
                self._outputdir, self._id + "_space-MNI152_T1w.nii.gz"
            )
            self._t2regfile = os.path.join(
                self._outputdir, self._id + "_space-MNI152_FLAIR.nii.gz"
            )
            ants.image_write(self._t1_n4, self._t1regfile)
            ants.image_write(self._t2_n4, self._t2regfile)

    def __skull_stripping(self):
        # specify the output filenames for brain extracted images
        self._t1brainfile = os.path.join(
            self._outputdir, self._id + "_space-MNI152_T1w" + self._outsuffix
        )
        self._t2brainfile = os.path.join(
            self._outputdir, self._id + "_space-MNI152_FLAIR" + self._outsuffix
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
            if self._t1file is not None and self._t2file is not None:
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
                    self._t1_brain = self._t1_n4 * self._mask
                    self._t2_brain = self._t2_n4 * self._mask
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
                    self._t1_brain = self._t1 * self._mask
                    self._t2_brain = self._t2 * self._mask
                ants.image_write(self._t1_brain, self._t1brainfile)
                ants.image_write(self._t2_brain, self._t2brainfile)

        # Generate BIDS metadata for the output files
        self.__generate_bids_metadata()

    def __generate_bids_metadata(self):
        """Generate BIDS-compliant metadata for preprocessed output files."""
        logger.info("generating BIDS metadata for preprocessed outputs")
        print("generating BIDS metadata for preprocessed outputs")

        try:
            if (
                generate_bids_metadata_for_outputs is not None
                and parse_subject_session is not None
            ):
                # Parse subject and session from ID
                subject_id, session_id = parse_subject_session(self._id)

                # Determine processing steps based on what was actually done
                processing_steps = []
                if self._preprocess:
                    processing_steps.extend(
                        [
                            "Registration to MNI152 template space using ANTs",
                            "N3 bias field correction using ANTs",
                        ]
                    )

                # Always include brain extraction since it's always performed
                if os.environ.get("BRAIN_MASKING") == "cpu":
                    processing_steps.append("Brain extraction using ANTsPyNet")
                else:
                    processing_steps.append(
                        "Brain extraction using deepMask neural network"
                    )

                if self._preprocess:
                    processing_steps.append("Intensity normalization")
                else:
                    processing_steps.append(
                        "Intensity normalization (min-max scaling to 0-100)"
                    )

                # Generate metadata for both output files
                generate_bids_metadata_for_outputs(
                    subject_id=subject_id,
                    session_id=session_id,
                    output_dir=self._outputdir,
                    original_t1_file=self._t1file,
                    original_t2_file=self._t2file,
                    processing_steps=processing_steps,
                    space="MNI152" if self._preprocess else "native",
                )

                logger.info("BIDS metadata generation completed successfully")
                print("BIDS metadata generation completed successfully")

            else:
                logger.warning(
                    "BIDS metadata utilities not available - skipping metadata generation"
                )
                print(
                    "Warning: BIDS metadata utilities not available - skipping metadata generation"
                )

        except Exception as e:
            logger.warning(f"Error generating BIDS metadata: {e}")
            print(f"Warning: Error generating BIDS metadata: {e}")

    def __apply_transforms(self):
        logger.info(
            "apply transforms to project outputs back to the native input space"
        )
        print("apply transforms to project outputs back to the native input space")
        self._t1_native = apply_transform(
            self._t1_brain, self._t1, self._t1_reg["fwdtransforms"][0], invert_xfrm=True
        )
        self._t2_native = apply_transform(
            self._t2_brain, self._t2, self._t2_reg["fwdtransforms"][0], invert_xfrm=True
        )

        # write skull-stripped versions of the brain in native space
        ants.image_write(
            self._t1_native, self._t1brainfile.replace("MNI152", "orig")
        )
        ants.image_write(
            self._t2_native, self._t2brainfile.replace("MNI152", "orig")
        )

    def __generate_QC_maps(self):
        logger.info("generating QC report")
        qcdir = os.path.join(self._args.tmpdir, "qc")
        if not os.path.exists(qcdir):
            os.makedirs(qcdir)
        if self._t1file is not None and self._t2file is not None:
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
        }

        for file in os.listdir(self._outputdir):
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
        print(f"pipeline processing time elapsed: {np.round(end - start, 1)} seconds")
        logger.info(
            f"pipeline processing time elapsed: {np.round(end - start, 1)} seconds"
        )
        logger.info("*********************************************")
