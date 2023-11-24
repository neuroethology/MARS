import tensorflow as tf
import numpy as np
from skimage.transform import resize
# from PIL import Image
import json
import multiprocessing as mp
import cv2
import progressbar
import platform
from pathlib import Path

import os, sys
import pickle
import logging

def get_macOS_version_info():
    if platform.system() != "Darwin":
        return (0, 0, 0)
    version = list(map(int, platform.mac_ver()[0].split('.')))
    for ix in range(len(version), 3):
        version.append(0)
    return tuple(version)

major, minor, _ = get_macOS_version_info()
if major == 10 and minor >= 15:
    root_logger = logging.getLogger()
    saved_level = root_logger.level
    root_logger.setLevel(logging.ERROR)
    import coremltools as cmt
    root_logger.setLevel(saved_level)
    use_coreml = True
else:
    use_coreml = False

    
class ImportGraphDetection():
    """ Convenience class for setting up the detector and using it."""
    def __init__(self, quant_model):
        self.use_ml_model = False

        if use_coreml:
            # we're running on a Mac with at least MacOS Catalina, so use MLModel
            # First, try to read an existing mlmodel
            quant_model_stem, _ = os.path.splitext(quant_model)
            self.ml_model_path = quant_model_stem + '.mlmodel'
            if os.path.exists(self.ml_model_path):
                # Use an existing .mlmodel file
                # self.ml_model = cmt.models.MLModel(self.ml_model_path)
                self.ml_model = None
                self.use_ml_model = True
                print(f"ImportGraphDetection: Using mlmodel from {self.ml_model_path}")
            else:
                # Generate a .mlmodel using coremltools.converters.convert
                print(f"ImportGraphDetection: Building and saving {self.ml_model_path} from {quant_model}")
                mlmodel = cmt.converters.convert(quant_model)
                # We can't quantize down to fp16 to reduce model size.
                self.ml_model = cmt.models.neural_network.quantization_utils.quantize_weights(mlmodel, 16)
                self.ml_model.save(self.ml_model_path)
                self.use_ml_model = True

        else:   # use the TensorFlow model
            # Read the graph protocol buffer (.pb) file and parse it to retrieve the unserialized graph definition.
            # print("ImportGraphDetection: Running with TensorFlow (no GPU on Mac)")
            with tf.io.gfile.GFile(quant_model, 'rb') as f:
                self.graph_def = tf.compat.v1.GraphDef()
                self.graph_def.ParseFromString(f.read())

            # Load the graph definition (stored in graph_def) into a live graph.
            self.graph = tf.Graph()
            with self.graph.as_default():
                tf.import_graph_def(self.graph_def, name="")

            # Configure the settings for our Tensorflow session.
            sess_config = tf.compat.v1.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                # gpu_options=tf.GPUOptions(allow_growth=True))
                gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=.30))

            # Create the Session we will use to execute the model.
            self.sess = tf.compat.v1.Session(graph=self.graph, config=sess_config)

            # Give object access to the input and output nodes.
            self.input_op = self.graph.get_operation_by_name('images')
            self.input_tensor = self.input_op.outputs[0]
            self.output_op_loc = self.graph.get_operation_by_name('predicted_locations')
            self.output_tensor_loc = self.output_op_loc.outputs[0]
            self.output_op_conf = self.graph.get_operation_by_name('Multibox/Sigmoid')
            self.output_tensor_conf = self.output_op_conf.outputs[0]

    def run(self, input_image):
        ''' This method is what actually runs an image through the Multibox network.'''
        if self.use_ml_model:
            if not self.ml_model:
                # delay loading the model until we're in the correct thread
                self.ml_model = cmt.models.MLModel(self.ml_model_path)
            predictions = self.ml_model.predict({'images': input_image})
            return predictions['predicted_locations'], predictions['Multibox/Sigmoid']
        return self.sess.run([self.output_tensor_loc, self.output_tensor_conf], {self.input_tensor: input_image})


class ImportGraphPose():
    """ Convenience class for setting up the pose estimator and using it."""
    def __init__(self, quant_model):
        quant_model_stem, _ = os.path.splitext(quant_model)
        self.use_ml_model = False

        if use_coreml: # on MacOS Catalina; use Mac-native GPU if possible
            self.ml_model_path = quant_model_stem + '.mlmodel'
            if os.path.exists(self.ml_model_path):
                # Use an existing .mlmodel file
                # self.ml_model = cmt.models.MLModel(self.ml_model_path)
                self.ml_model = None
                self.use_ml_model = True
                print(f"ImportGraphPose: Using mlmodel from {self.ml_model_path}")
            else:
                # Generate a .mlmodel using coremltools.models.nn.convert
                # We can't quantize down to fp16 to reduce model size.
                print(f"ImportGraphPose: Building and saving {self.ml_model_path} from {quant_model}")
                self.ml_model = cmt.converters.convert(quant_model)
                self.ml_model.save(self.ml_model_path)
                self.use_ml_model = True

        else:   # use TensorFlow model
            # Read the graph protbuf (.pb) file and parse it to retrieve the unserialized graph definition.
            # print("ImportGraphPose: Running with TensorFlow (no GPU on Mac)")
            with tf.io.gfile.GFile(quant_model, 'rb') as f:
                self.graph_def = tf.compat.v1.GraphDef()
                self.graph_def.ParseFromString(f.read())

            # Load the graph definition (stored in graph_def) into a live graph.
            self.graph = tf.Graph()
            with self.graph.as_default():
                tf.import_graph_def(self.graph_def, name="")

            # Create the tf.session we will use to execute the model.
            sess_config = tf.compat.v1.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                # gpu_options=tf.GPUOptions(allow_growth=True))
                gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=.35))

            self.sess = tf.compat.v1.Session(graph=self.graph, config=sess_config)

            # access to input and output nodes
            self.input_op = self.graph.get_operation_by_name('images')
            self.input_tensor = self.input_op.outputs[0]
            self.output_op_heatmaps = self.graph.get_operation_by_name('output_heatmaps')
            self.output_tensor_heatmaps = self.output_op_heatmaps.outputs[0]

    def run(self, cropped_images):
        """ This method is what actually runs an image through the stacked hourglass network."""
        if self.use_ml_model:
            if not self.ml_model:
                # delay loading the model until we're in the correct thread
                self.ml_model = cmt.models.MLModel(self.ml_model_path)
            predictions = self.ml_model.predict({'images': cropped_images})
            # wrap return value in list to match TF behavior
            return [predictions['output_heatmaps']]
        return self.sess.run([self.output_tensor_heatmaps], {self.input_tensor: cropped_images})


def get_median_frame(cap):
    # Randomly select 25 frames
    frameIds = np.round(cap.NUM_FRAMES * np.random.uniform(size=25))

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        frame = cap.getFrame(fid)
        frames.append(frame)
    cap.seek(0) # reset to first frame

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return medianFrame


def pre_process_image(image, medianFrame, IM_TOP_H, IM_TOP_W, DET_IM_SIZE):
    """ Takes a u8int image and prepares it for detection. """

    if medianFrame: # this is empty if bgSubtract is false
        norm_image = cv2.divide(image.astype(np.float32), medianFrame.astype(np.float32))
        norm_image = np.minimum(norm_image, 8.0)
    else:
        norm_image = image.astype(np.float32)

    # Resize the image to the size the detector takes.
    prep_image = cv2.resize(norm_image, (DET_IM_SIZE, DET_IM_SIZE), interpolation=cv2.INTER_NEAREST)

    # Convert image to float and shift the image values from [0, 256] to [-1, 1].
    if medianFrame:
        prep_image = np.divide(np.add(prep_image, -4.).astype(np.float32), 4.)
    else:
        prep_image = (prep_image - 128) / 128

    # Convert to RGB if necessary.
    if len(prep_image.shape) < 3:
        prep_image = cv2.cvtColor(prep_image, cv2.COLOR_GRAY2RGB)

    # Flatten the array.
    prep_image = prep_image.ravel()
    # Add an additional dimension to stack images on.
    return np.expand_dims(prep_image, 0)


def post_process_detection(locations, confidences):
    """ Takes Multibox predictions and their confidences scores, chooses the best one, and returns a stretched version.
    """
    # pred_locs: [x1,y1,x2,y2] in normalized coordinates
    pred_locs = np.clip(locations[0], 0., 1.)

    # First, we want to filter the proposals that are not in the square.
    filtered_bboxes = [[0.,0.,0.,0.]]
    filtered_confs = [0]
    best_conf = 0.
    for bbox, conf in zip(pred_locs, confidences[0]):
        if conf > .005 and conf > best_conf:
            best_conf = conf
            filtered_bboxes[0] = bbox
            filtered_confs[0] = conf
        """
        print(f"bbox: {bbox}, conf: {conf}")
        if bbox[0] < 0.: continue
        if bbox[1] < 0.: continue
        if bbox[2] > 1.: continue
        if bbox[3] > 1.: continue
        filtered_bboxes.append(bbox)
        filtered_confs.append(conf)
        """

    # Convert these from lists to numpy arrays.
    filtered_bboxes = np.array(filtered_bboxes)
    filtered_confs = np.array(filtered_confs)

    # Now, take the bounding box we are most confident in. If it's above 0.005 confidence, stretch and return it.
    # Otherwise, just return an empty list and 0 confidence.
    if filtered_bboxes.shape[0] != 0:
        sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
        filtered_bboxes = filtered_bboxes[sorted_idxs]
        filtered_confs = filtered_confs[sorted_idxs]
        bbox_to_keep = filtered_bboxes[0].ravel()
        conf_to_keep = float(np.asscalar(filtered_confs[0]))
        # are we enough confident?
        if conf_to_keep > .005:
            # Unpack the bbox values.
            xmin, ymin, xmax, ymax = bbox_to_keep

            # Make sure the bbox hasn't collapsed on itself.
            if abs(xmin-xmax)<0.005:
                if xmin > 0.006:
                    xmin=xmin - 0.005
                else:
                    xmax = xmax + 0.005
            if abs(ymin-ymax)<0.005:
                if ymin>0.006:
                    ymin=ymin - 0.005
                else:
                    ymax = ymax + 0.005

            # Whether we use constant (vs width-based) stretch.
            useConstant = 0

            # Set the constant stretch amount.
            stretch_const = 0.06

            # Set the fractional stretch factor.
            stretch_factor = 0.10

            if useConstant:
                stretch_constx = stretch_const
                stretch_consty = stretch_const
            else:
                stretch_constx = (xmax-xmin)*stretch_factor #  of the width
                stretch_consty = (ymax-ymin)*stretch_factor

            # Calculate the amount to stretch the x by.
            x_stretch = np.minimum(xmin, abs(1-xmax))
            x_stretch = np.minimum(x_stretch, stretch_constx)

            # Calculate the amount to stretch the y by.
            y_stretch = np.minimum(ymin, abs(1-ymax))
            y_stretch = np.minimum(y_stretch, stretch_consty)

            # Adjust the bounding box accordingly.
            xmin -= x_stretch
            xmax += x_stretch
            ymin -= y_stretch
            ymax += y_stretch
            return [xmin, ymin, xmax, ymax], conf_to_keep
        else:
            # No good proposals, return substantial nothing.
            return [], 0.
    else:
        # No proposals, return nothing.
        return [], 0.


def extract_resize_crop_bboxes(bboxes, IM_W, IM_H, image):
    """ Resizes the bbox, and crops the image accordingly. Returns the cropped image."""
    # Define the image input size. TODO: Make this an input to the function.
    POSE_IM_SIZE = 256
    # Prepare a placeholder for the images.
    prepped_images = np.zeros((0, POSE_IM_SIZE, POSE_IM_SIZE, 3), dtype=np.float32)
    # Scale normalized coordinates to image coordinates.
    scaled_bboxes = np.round(bboxes * np.array([IM_W, IM_H, IM_W, IM_H])).astype(int)

    # Extract the image using the bbox, then resize it to square (distorts aspect ratio).
    for i, bbox in enumerate(scaled_bboxes):
        # Unpack the bbox.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        # Crop the image.
        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2].astype(np.float32)

        # Resize the image to the pose input size.
        im = cv2.resize(bbox_image, (POSE_IM_SIZE, POSE_IM_SIZE), interpolation=cv2.INTER_NEAREST)
        """
        im = resize(bbox_image, (POSE_IM_SIZE, POSE_IM_SIZE, bbox_image.shape[2]))
        im = np.array(Image.fromarray(bbox_image).resize((POSE_IM_SIZE, POSE_IM_SIZE)))
        """

        # do per-image processing here, when the images are still 2D, to save time
        im = (im - 128) / 128
        if len(im.shape) < 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

       # Get a new 0th-dimension, to make the image(s) stackable.
        im = np.expand_dims(im, 0)

        # Concatenate the image to the stack.
        prepped_images = np.concatenate([prepped_images, im])

    # Now convert the image to a float and rescale between -1 and 1.
    """
    prepped_images = prepped_images.astype(np.uint8)
    prepped_images = prepped_images.astype(np.float32)
    prepped_images = np.subtract(prepped_images, 128.)
    prepped_images = np.divide(prepped_images, 128.)
    """
    return prepped_images


def post_proc_heatmaps(predicted_heatmaps, bboxes, IM_W, IM_H, POSE_IM_SIZE):
    """ Postprocesses the heatmaps generated by the SHG. Returns the keypoints and their associated scores in a list."""
    keypoints_res = []
    scores =[]
    # For each stack in the batch, extract out the bboxes from the heatmaps,
    # then, for each heatmap in the stack, extract out argmax point. This is the estimated keypoint.
    for b in range(len(predicted_heatmaps[0])):
        # Get the stack of heatmaps.
        heatmaps = predicted_heatmaps[0][b]

        # Clip the heatmaps.
        heatmaps = np.clip(heatmaps, 0., 1.)

        # Resize them to square.
        resized_heatmaps = cv2.resize(heatmaps, (POSE_IM_SIZE, POSE_IM_SIZE), interpolation=cv2.INTER_LINEAR)

        # Unpack the bboxes and rescale them from norm coordinates to image coordinates.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes[b]
        bbox_w = (bbox_x2 - bbox_x1) * IM_W
        bbox_h = (bbox_y2 - bbox_y1) * IM_H

        # Now resize the heatmaps to the original bbox size.
        rescaled_heatmaps = cv2.resize(resized_heatmaps, (int(np.round(bbox_w)), int(np.round(bbox_h))),
                                       interpolation=cv2.INTER_LINEAR)

        NUM_PARTS = len(rescaled_heatmaps[0][0])
        keypoints = np.zeros((NUM_PARTS, 3))
        # For each part-heatmap, extract out the keypoint, then place it in the original image's coordinates.
        for j in range(NUM_PARTS):
            # Get the current heatmap.
            hm = rescaled_heatmaps[:, :, j]
            score = float(np.max(hm))

            # Extract out the keypoint.
            x, y = np.array(np.unravel_index(np.argmax(hm), hm.shape)[::-1])

            # Place it in the original image's coordinates.
            imx = x + bbox_x1 * IM_W
            imy = y + bbox_y1 * IM_H

            # Store it.
            keypoints[j, :] = [imx, imy, score]

        # Store the x's, y's, and scores in lists.
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]
        ss = keypoints[:, 2]
        keypoints_res.append([xs.tolist(), ys.tolist()])
        scores.append(ss.tolist())
    return keypoints_res, scores


# The global poison pill that we pass between processes. TODO: Could probably have this encapsulated in a fxn?
POISON_PILL = "STOP"

def get_poison_pill():
    """ Just in case we need to access the poison pill from outside this module."""
    return POISON_PILL

def pre_det_inner(raw_data, median_frame, IM_TOP_H, IM_TOP_W):
    """ The per-frame work of preparing the image for detection """

    # TODO: Make these parameters inputs to the function.
    DET_IM_SIZE = 299
    POSE_IM_SIZE = 256
    # unpack the input data
    top_image, bbox = raw_data

    # Preprocess the image.
    top_input_image = pre_process_image(top_image, median_frame, IM_TOP_H, IM_TOP_W, DET_IM_SIZE)

    return [top_input_image, bbox], top_image


def pre_det(q_in, q_out_predet, q_out_raw, median_frame, IM_TOP_H, IM_TOP_W):
    """ Worker function that preprocesses raw images for input to the detection network.
    q_in: from frame feeding loop in the main function.

    q_out_predet: to det, the detection function.
    q_out_raw: to pre_hm, the pose estimation pre-processing function. """
    try:
        frame_num = 0
        while True:
            # Load in the raw image and (possibly dummy) bounding box
            raw_data = q_in.get()

            if raw_data == POISON_PILL:
                q_out_predet.put(POISON_PILL)
                q_out_raw.put(POISON_PILL)
                return

            predet_out_det, predet_out_pose = pre_det_inner(raw_data, median_frame, IM_TOP_H, IM_TOP_W)

            # Send the altered output image to the detection network, and the raw image to the pre-pose estimation fxn.
            q_out_predet.put(predet_out_det)
            q_out_raw.put(predet_out_pose)
            frame_num += 1
    except Exception as e:
        print("\nerror occurred during pre-processing for detection (function MARS_pose_machinery.pre_det)")
        print(e)
        raise(e)

def run_det_setup(view, opts):
    """ setup for run_det """
    if not opts['useExistingBBoxes']:
        if view == 'front':
            QUANT_DET_PATHS = opts['detector_front']
        else:
            QUANT_DET_PATHS = opts['detector_top']

        # Import the detection networks.
        detectors = []
        for i,p in enumerate(QUANT_DET_PATHS):
            detectors.append(ImportGraphDetection(p))

        return detectors

def run_det_inner(input_data, det_model, opts):
    """ run_det_inner: the real work of running detection, per video frame """
    input_image,bbox = input_data
    if opts['useExistingBBoxes']:
        det_out = bbox # move along to the next queue
    else:
        # Run the detection network.
        locations, confidences = det_model.run(input_image)

        # Package the output up.
        return [locations, confidences]

def run_det(q_in,q_out, view, opts):
    """ Worker function that houses and runs the bounding box detection network. If the user provided their own
    bounding boxes, we just pass them along to the next stage of the queue.
    q_in: from pre-det, the detection pre-processing function.

    q_out: to post-det, the detection post-processing function."""
    try:
        # Decide on which view to use.
        detectors = run_det_setup(view, opts)

        while True:
            # Get the input image.
            input_data = q_in.get()

            # Check if we got the poison pill --if so, shut down.
            if input_data == POISON_PILL:
                q_out.put(POISON_PILL)
                return
            det_out = []
            for d in detectors:
                det_out.append(run_det_inner(input_data, d, opts))
            # Send the output to the post-detection processing worker.
            q_out.put(det_out)
    except Exception as e:
        print("\nerror occurred during detection (function MARS_pose_machinery.run_det)")
        print(e)
        q_out.put(POISON_PILL)
        raise e


def post_det_setup():
    # Initialize the bounding boxes for each animal.
    max_mice = 99  # can be increased as desired, but damn what experiment are you running?
    det_prev_ok_loc = [np.array([1e-2, 1e-2, 2e-2, 2e-2])]*max_mice
    det_prev_ok_conf = [0.0001]*max_mice
    return det_prev_ok_loc, det_prev_ok_conf


def post_det_inner(det_input, det_prev_ok_loc, det_prev_ok_conf):
    ARBITRARY_CONFIDENCE_THRESHOLD = 0.005

    # Unpack the detection output.
    num_mice = len(det_input)
    det_bbox = [None]*num_mice
    det_conf = [None]*num_mice
    for count, mouse in enumerate(det_input):
        # Post-process the detection for each mouse
        locations, confidences = mouse

        det_bbox_to_pass, det_conf_to_pass = post_process_detection(locations, confidences)

        # If the confidence is high enough, use our new bounding box. Otherwise, use the old one.
        if det_conf_to_pass > ARBITRARY_CONFIDENCE_THRESHOLD:
            det_prev_ok_loc[count] = det_bbox_to_pass
            det_prev_ok_conf[count] = det_conf_to_pass
        else:
            det_bbox_to_pass = det_prev_ok_loc[count]
            det_conf_to_pass = det_prev_ok_conf[count]

        # Bundle everything together.
        det_bbox[count] = det_bbox_to_pass
        det_conf[count] = det_conf_to_pass

    return [det_bbox, det_conf]



def post_det(q_in, q_out):
    """ Worker function that processes the detection output, so that we know which portion of the image to crop for
        pose estimation.
    q_in: from det, the detection network.

    q_out: to pre_hm, the pose estimation pre-processing function."""
    try:
        det_prev_ok_loc, det_prev_ok_conf = post_det_setup()

        while True:
            # Load in the detection output.
            det_input = q_in.get()

            # Check if we got the poison pill --if so, shut down.
            if det_input == POISON_PILL:
                q_out.put(POISON_PILL)
                return

            det_output = post_det_inner(det_input, det_prev_ok_loc, det_prev_ok_conf)

            # Send it to pose estimation pre-processing.
            q_out.put(det_output)

    except Exception as e:
        print("\nerror occurred in detector post-processing (function MARS_pose_machinery.post_det)")
        print(e)
        raise(e)

def pre_hm_inner(det_out, raw_image, IM_W, IM_H):
    # Unpack the detection output.
    bboxes, confs = det_out

    # Convert the bounding boxes to an np.array
    bboxes = np.array(bboxes)

    # Prepare the images for pose estimation, by extracting the proper cropped region.
    prepped_images = extract_resize_crop_bboxes(bboxes, IM_W, IM_H, raw_image)

    return prepped_images, [bboxes, confs]


def pre_hm(q_in_det, q_in_image, q_out_img, q_out_bbox, IM_W, IM_H):
    """ Worker function that pre-processes an image for pose-estimation. Includes cropping and resizing it.
    q_in_det: from post_det, contains the post-processed bounding boxes.
    q_in_image: from pre_det, contains the raw image that needs to be cropped.

    q_out_img: to hm, the pose estimator.
    q_out_bbox: to post_hm, in order to reconstruct the keypoints' true locations."""
    try:
        while True:
            # Collect the detection output.
            det_out = q_in_det.get()

            # Check if we got the poison pill --if so, shut down.
            if det_out == POISON_PILL:
                q_out_img.put(POISON_PILL)
                q_out_bbox.put(POISON_PILL)
                return

            # Collect the raw image.
            raw_image = q_in_image.get()

            prepped_images, bboxes_confs = pre_hm_inner(det_out, raw_image, IM_W, IM_H)

            q_out_img.put(prepped_images)
            q_out_bbox.put(bboxes_confs)
    except Exception as e:
        print("\nerror occurred during pre-processing for pose estimation (function MARS_pose_machinery.pre_hm)")
        print( e)
        raise(e)


def run_hm_setup(view, opts):
    # Figure out which view we're using.
    if view == 'front':
        QUANT_POSE = str(Path(opts['pose_front'][0]))  # TODO: fix HM setup to support multiple pose models ################################################################
    else:
        QUANT_POSE = str(Path(opts['pose_top'][0]))

    # Import the pose model.
    return ImportGraphPose(QUANT_POSE)


def run_hm_inner(prepped_images, pose_model):
    # Run the pose estimation network.
    return pose_model.run(prepped_images)


def run_hm(q_in_img, q_out_hm, view, opts):
    """ Worker function that performs the pose-estimation.
    q_in_img: from pre_hm, contains the images cropped and resized for inference.

    q_out_hm: to post_hm, contains the heatmaps."""
    try:
        pose_model = run_hm_setup(view, opts)
        while True:
            # Collect the prepared images for inference.
            prepped_images = q_in_img.get()

            # Check if we got the poison pill --if so, shut down.
            if prepped_images == POISON_PILL:
                q_out_hm.put(POISON_PILL)
                return

            predicted_heatmaps = run_hm_inner(prepped_images, pose_model)

            # Send the heatmaps out for post-processing.
            q_out_hm.put(predicted_heatmaps)
    except Exception as e:
        print("\nerror occurred during pose estimation (function MARS_pose_machinery.run_hm)")
        print(e)
        raise(e)


def post_hm_setup(NUM_FRAMES):
    # Set a placeholder dictionary for our outputs.
    pose_frames = {'scores': [], 'keypoints': [],'bbox':[],'bscores':[]}
    # Set up the command-line progress bar.
    bar = progressbar.ProgressBar(widgets=['Pose ', progressbar.Percentage(), ' -- ',
                                            progressbar.FormatLabel('frame %(value)d'), '/',
                                            progressbar.FormatLabel('%(max)d'), ' [', progressbar.Timer(), '] ',
                                            progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=NUM_FRAMES)
    # Start the progress bar.
    bar.start()
    return pose_frames, bar


def post_hm_inner(predicted_heatmaps, bboxes_confs, IM_W, IM_H, POSE_IM_SIZE, NUM_FRAMES, POSE_BASENAME, pose_frames, bar, current_frame_num):
    # Post-process the heatmaps to get the keypoints out.
    keypoints_res, scores = post_proc_heatmaps(predicted_heatmaps, bboxes_confs[0], IM_W, IM_H, POSE_IM_SIZE)

    # Store the fresh keypoints, keypoint-scores, bounding boxes, and bounding box confidences.
    pose_frames['keypoints'].append(keypoints_res)
    pose_frames['scores'].append(scores)
    pose_frames['bbox'].append(bboxes_confs[0].tolist())
    pose_frames['bscores'].append(bboxes_confs[1])

    # Update our progress bar.
    bar.update(current_frame_num)

    #TODO: Should probably be an input?
    SAVE_EVERY_X_FRAMES = 2000

    # If we've reached a certain point, save a checkpoint for our pose output.
    if (current_frame_num % SAVE_EVERY_X_FRAMES == 0 or current_frame_num == NUM_FRAMES - 1) and current_frame_num > 1:
        with open(POSE_BASENAME + '.json', 'w') as fp:
            json.dump(pose_frames, fp)




def post_hm(q_in_hm, q_in_bbox, IM_W, IM_H, POSE_IM_SIZE, NUM_FRAMES, POSE_BASENAME):
    """ Worker function that processes the heatmaps for their keypoints, and saves them.
    q_in_hm: from hm, contains the unprocessed heatmaps.
    q_in_bbox: from prehm, contains the bounding boxes used to generate a given heatmap.
    """

    try:
        pose_frames, bar = post_hm_setup(NUM_FRAMES)

        # Initialize the current frame number.
        current_frame_num = 0
        while True:
            # Collect the predicted heatmaps.
            predicted_heatmaps = q_in_hm.get()

            # Collect the predicted bounding boxes.
            bboxes_confs = q_in_bbox.get()

            # Check if we got the poison pill --if so, shut down.
            if bboxes_confs == POISON_PILL:
                bar.finish()
                return pose_frames

            post_hm_inner(predicted_heatmaps, bboxes_confs, IM_W, IM_H, POSE_IM_SIZE, NUM_FRAMES, POSE_BASENAME, pose_frames, bar, current_frame_num)

            # Increment the frame_number.
            current_frame_num += 1
    except Exception as e:
        print("\nerror occurred during pose post-processing (function MARS_pose_machinery.post_hm)")
        print(e)
        raise(e)

def run_gpu(q_in_det, q_out_det, q_in_hm, q_out_hm, view, opts, queue_size):
    do_det = True
    do_hm = True
    BATCH_SIZE = min(queue_size, 16)
    print("\n\nIn run_gpu\n\n")
    try:
        # Decide on which view to use.
        det_black, det_white = run_det_setup(view, opts)
        pose_model = run_hm_setup(view, opts)
    except Exception as e:
        print("\nError occurred during loading of models")
        print(e)
        q_out_det.put(POISON_PILL)
        q_out_hm.put(POISON_PILL)
        raise e

    while do_det or do_hm:
        if do_det:
            for item in range(BATCH_SIZE):
                try:
                    # Get the input image.
                    if q_in_det.empty():
                        print("q_in_det is empty")
                        break
                    print('trying to get from q_in_det')
                    input_data = q_in_det.get(False)
                    print('got det')
                except Exception as e:
                    print(f'unexpected exception trying to get from q_in_det: {e}')
                    raise e

                # Check if we got the poison pill --if so, shut down.
                if input_data == POISON_PILL:
                    q_out.put(POISON_PILL)
                    do_det = False
                    break

                try:
                    det_b = run_det_inner(input_data, det_black, opts)
                    det_w = run_det_inner(input_data, det_white, opts)
                    # Send the output to the post-detection processing worker.
                    q_out_det.put([det_b, det_w])
                except Exception as e:
                    print("\nerror occurred during detection (function MARS_pose_machinery.run_det)")
                    print(e)
                    q_out.put(POISON_PILL)
                    raise e

        if do_hm:
            for item in range(BATCH_SIZE):
                try:
                    # Collect the prepared images for inference.
                    if q_in_hm.empty(): # only a hint
                        print("q_in_hm is empty")
                        break
                    print('trying to get from q_in_hm')
                    prepped_images = q_in_hm.get()
                    print('got hm')
                except Exception as e:
                    print(f"unexpected exception trying to get from q_in_hm: {e}")
                    raise e

                # Check if we got the poison pill --if so, shut down.
                if prepped_images == POISON_PILL:
                    q_out_hm.put(POISON_PILL)
                    do_hm = False
                    break

                try:
                    predicted_heatmaps = run_hm_inner(prepped_images, pose_model)

                    # Send the heatmaps out for post-processing.
                    q_out_hm.put(predicted_heatmaps)
                except Exception as e:
                    print("\nerror occurred during pose estimation (function MARS_pose_machinery.run_hm)")
                    print(e)
                    raise(e)
    return
