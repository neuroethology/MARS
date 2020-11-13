from __future__ import print_function
import os
import sys
import scipy.io as sp
from MARS_pose_machinery import *
import warnings
import multiprocessing as mp
import logging
import platform
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import time

sys.path.append('./')

import MARS_output_format as mof
from util.genericVideo import *
from MARS_detection_unpackers import *


def extract_pose_wrapper(video_fullpath, view, doOverwrite, progress_bar_signal='',
    verbose=0, output_suffix='', mars_opts={}, max_frames=999999):
    video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)
    output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                         output_suffix = output_suffix)

    extract_pose(video_fullpath= video_fullpath,
                 output_folder=output_folder,
                 output_suffix=output_suffix,
                 view = view,
                 doOverwrite=doOverwrite,
                 progress_bar_signal=progress_bar_signal,
                 mars_opts = mars_opts,
                 verbose = verbose,
                 max_frames = max_frames
                 )
    return


def get_macOS_version_info():
    if platform.system() != "Darwin":
        return (0, 0, 0)
    return tuple(map(int, platform.mac_ver()[0].split('.')))

def extract_pose(video_fullpath, output_folder, output_suffix, view,
                       doOverwrite, progress_bar_signal, mars_opts,
                       verbose=1, max_frames=999999):

    pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath,
                                    output_folder=output_folder,
                                    view=view,
                                    output_suffix=output_suffix)
    video_name = os.path.basename(video_fullpath)

    pose_mat_name = pose_basename  + '.mat'
    
    # Makes the output directory, if it doesn't exist.
    mof.getdir(output_folder)
    
    _,ext = os.path.splitext(video_fullpath)
    ext=ext[1:] # get rid of the dot.

    already_extracted_msg = (
        '1 - Pose already extracted. Change your settings to override, if you still want to extract the pose.')
    
    if not (ext in video_name):
        print("File type unsupported! Aborted.")
        return
    
    try:
        # coremltools on MACs doesn't interact well with multiprocessing, but access to the system GPU, even if it's not
        # from NVidia, more than makes up for sequential processing.  coremltools only works on MacOS Catalina and up.
        major, minor, _ = get_macOS_version_info()
        use_multiprocessing = not (major == 10 and minor >= 15)

        if verbose:
            print('1 - Extracting pose')
    
        if (not os.path.exists(pose_mat_name)) | (doOverwrite):

            if not (view == 'front') and not (view == 'top'):
                raise ValueError('Invalid view type, please specify top or front.')
                return

            # create the movie reader:
            reader = vidReader(video_fullpath)
            NUM_FRAMES = reader.NUM_FRAMES
            IM_H = reader.IM_H
            IM_W = reader.IM_W
            fps = reader.fps
            medianFrame = []
            if mars_opts['bgSubtract']:
                medianFrame = get_median_frame(vc,'cv2')


            NUM_FRAMES = min(NUM_FRAMES, max_frames)

            # unpack user-provided bounding boxes if they exist:
            bboxes = [None] * NUM_FRAMES
            if mars_opts['useExistingBBoxes']:
                print('   Unpacking user-provided bounding boxes...')
                bboxes = unpack_bbox_wrapper(mars_opts, video_fullpath, IM_W, IM_H, NUM_FRAMES)

            if verbose:
                print('   Processing video for detection and pose ...')
            DET_IM_SIZE = 299
            POSE_IM_SIZE = 256

            if use_multiprocessing:
                if verbose:
                    print("      Creating pool...")

                workers_to_use = 8
                pool = mp.Pool(workers_to_use)
                manager = mp.Manager()
                maxsize = 5
                
                if verbose:
                    print("      Pool created with %d workers. \n      Creating queues" % workers_to_use)
    
                # create managed queues
                q_start_to_predet = manager.Queue(maxsize)
                q_predet_to_det = manager.Queue(maxsize)
                q_predet_to_prehm = manager.Queue(maxsize)
                q_det_to_postdet = manager.Queue(maxsize)
                q_postdet_to_prehm = manager.Queue(maxsize)
                q_prehm_to_hm_IMG = manager.Queue(maxsize)
                q_prehm_to_posthm_BBOX = manager.Queue(maxsize)
                q_hm_to_posthm_HM = manager.Queue(maxsize)
                q_posthm_to_end = manager.Queue(maxsize)

                if verbose:
                    print("      Queues created. \n      Linking pools")

                try:

                    results_predet = pool.apply_async(pre_det,
                                                    (q_start_to_predet,
                                                    q_predet_to_det, q_predet_to_prehm,
                                                        medianFrame, IM_H, IM_W))
                    results_det = pool.apply_async(run_det,
                                                (q_predet_to_det,q_det_to_postdet,
                                                    view, mars_opts))
                    results_postdet = pool.apply_async(post_det,
                                                    (q_det_to_postdet, q_postdet_to_prehm))
                    results_prehm = pool.apply_async(pre_hm,
                                                    (q_postdet_to_prehm, q_predet_to_prehm,
                                                    q_prehm_to_hm_IMG, q_prehm_to_posthm_BBOX,
                                                    IM_W, IM_H))
                    results_hm = pool.apply_async(run_hm,
                                                (q_prehm_to_hm_IMG,q_hm_to_posthm_HM,
                                                view, mars_opts))
                    results_posthm = pool.apply_async(post_hm,
                                                        (q_hm_to_posthm_HM,q_prehm_to_posthm_BBOX,
                                                        IM_W, IM_H,POSE_IM_SIZE,NUM_FRAMES,pose_basename))
                except Exception as e:
                    print("Error starting Pools:")
                    print(e)
                    raise(e)

                if verbose:
                    print('      Pools linked.\n      Feeding data...')
                if progress_bar_signal:
                    # Update the progress bar with the number of total frames it will be processing.
                    progress_bar_signal.emit(0, NUM_FRAMES)

                for f in range(NUM_FRAMES):
                    img = reader.getFrame(f)
                    q_start_to_predet.put([img,bboxes[f]])

                # Push through the poison pill.
                q_start_to_predet.put(get_poison_pill())
        
                if verbose:
                    print("      Pools Started...")
                pool.close()
                pool.join()

                if verbose:
                    print("      Pools Finished. \n      Saving...")
                top_pose_frames = results_posthm.get()
    
            else:   # don't use multiprocessing, but process frames in batches
                time_steps = True


                if time_steps:
                    process_time = [0.] * 10
                    process_time_start = time.perf_counter()

                det_black, det_white = run_det_setup(view, mars_opts)
                det_prev_ok_loc, det_prev_ok_conf = post_det_setup()
                pose_model = run_hm_setup(view, mars_opts)
                top_pose_frames, bar = post_hm_setup(NUM_FRAMES)
                current_frame_num = 0

                if time_steps:
                    process_time[0] += time.perf_counter() - process_time_start

                BATCH_SIZE = 16
                in_q = [None] * BATCH_SIZE
                det_b_q = [None] * BATCH_SIZE
                det_w_q = [None] * BATCH_SIZE
                pose_image_q = [None] * BATCH_SIZE
                # """

                    # """
                for batch in range((NUM_FRAMES + BATCH_SIZE - 1) // BATCH_SIZE):
                    batch_start = batch * BATCH_SIZE
                    batch_end = min(batch_start + BATCH_SIZE, NUM_FRAMES)
                    
                    if time_steps:
                        process_time_start = time.perf_counter()
                    
                    for f in range(batch_start, batch_end):
                        if time_steps:
                            process_time_start_0 = time.perf_counter()

                        ix = f - batch_start
                        img = reader.getFrame(f)

                        if time_steps:
                            process_time_1a = time.perf_counter()
                            process_time[1] += process_time_1a - process_time_start_0

                        in_q[ix], pose_image_q[ix] = pre_det_inner([img, bboxes[f]], medianFrame, IM_H, IM_W)

                        if time_steps:
                            process_time_1 = time.perf_counter()
                            process_time[9] += process_time_1 - process_time_1a

                    for ix in range(batch_end - batch_start):
                        det_b_q[ix] = run_det_inner(in_q[ix], det_black, mars_opts)

                    if time_steps:
                        process_time_2 = time.perf_counter()
                        process_time[2] += process_time_2 - process_time_1

                    for ix in range(batch_end - batch_start):
                        det_w_q[ix] = run_det_inner(in_q[ix], det_white, mars_opts)

                    if time_steps:
                        process_time_3 = time.perf_counter()
                        process_time[3] += process_time_3 - process_time_2

                    for ix in range(batch_end - batch_start):
                        if time_steps:
                            process_time_start_1 = time.perf_counter()
                        
                        det_out = post_det_inner([det_b_q[ix], det_w_q[ix]], det_prev_ok_loc, det_prev_ok_conf)

                        if time_steps:
                            process_time_4 = time.perf_counter()
                            process_time[4] += process_time_4 - process_time_start_1
                        
                        prepped_images, bboxes_confs = pre_hm_inner(det_out, pose_image_q[ix], IM_W, IM_H)

                        if time_steps:
                            process_time_5 = time.perf_counter()
                            process_time[5] += process_time_5 - process_time_4
                        
                        predicted_heatmaps = run_hm_inner(prepped_images, pose_model)

                        if time_steps:
                            process_time_6 = time.perf_counter()
                            process_time[6] += process_time_6 - process_time_5
                        
                        post_hm_inner(predicted_heatmaps, bboxes_confs, IM_W, IM_H, POSE_IM_SIZE, NUM_FRAMES, pose_basename, top_pose_frames, bar, current_frame_num)

                        if time_steps:
                            process_time_7 = time.perf_counter()
                            process_time[7] += process_time_7 - process_time_6

                        # Increment the frame_number.
                        current_frame_num += 1

                    if time_steps:
                        process_time[8] += process_time_7 - process_time_start

                    if progress_bar_signal:
                        progress_bar_signal.emit(f, 0)
            
                if time_steps:
                    NS_PER_SECOND = 1000000000
                    print("Process Times")
                    print("-----------------------------")
                    print(f"Setup             : {process_time[0]} sec\n")
                    print(f"File Read         : {process_time[1] / NUM_FRAMES} sec / frame")
                    print(f"Pre Detection     : {process_time[9] / NUM_FRAMES} sec / frame")
                    print(f"Detection (black) : {process_time[2] / NUM_FRAMES} sec / frame")
                    print(f"Detection (white) : {process_time[3] / NUM_FRAMES} sec / frame")
                    print(f"Post Detection    : {process_time[4] / NUM_FRAMES} sec / frame")
                    print(f"Pre Heatmap       : {process_time[5] / NUM_FRAMES} sec / frame")
                    print(f"Heatmap (pose)    : {process_time[6] / NUM_FRAMES} sec / frame")
                    print(f"Post Heatmap      : {process_time[7] / NUM_FRAMES} sec / frame")
                    print(f"Total processing  : {process_time[8] / NUM_FRAMES} sec / frame")

            top_pose_frames['keypoints'] = np.array(top_pose_frames['keypoints'])
            top_pose_frames['scores'] = np.array(top_pose_frames['scores'])
    
            top_pose_frames['bbox'] = np.array(top_pose_frames['bbox'])
            top_pose_frames['bscores'] = np.array(top_pose_frames['bscores'])
    
            sp.savemat(pose_mat_name, top_pose_frames)

            if verbose:
                print("Saved.\nPose Extracted")
            reader.close()
            return
        else:
            if verbose:
                print(already_extracted_msg)
            return
    except Exception as e:
        print(e)
        raise(e)
    return