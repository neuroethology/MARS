from __future__ import print_function
import os
import sys
import scipy.io as sp
import numpy as np
import csv


def unpack_bbox_wrapper(opts, video_path, IM_W, IM_H, NUM_FRAMES):
    '''
    The functions in this file unpack user-provided bounding boxes and pass them along to the MARS pose estimator.
    If you would like to use your own bounding box data with MARS, add your own custom data-unpacking script
    to this file (see below), and *make sure it gets called* by this wrapper function, by adding an "elif" to the
    code of unpack_bbox_wrapper.

    **See unpack_bbox_Chen in MARS_detection_unpackers.py for an example of a custom bounding box unpacker.**
    -----------------------------------------------------------

    ## Inputs to your custom script:
        - the full path to the current video file being processed by MARS. You should name your bounding boxes in
          such a way that you can find them programmatically given an associated video! (Eg save them in the same
          folder, with a standardized naming format).
        - optional: the width and height of the current video in pixels, called IM_W and IM_H. MARS represents bounding
          boxes in fractional coordinates (values 0-1), so if your bounding box file is in pixels you'll need to change
          units.
        - optional: the number of frames being tracked, called NUM_FRAMES. Used for preallocating space/making sure you
          have bounding box data for each frame.

    ## Output of your custom script:
        - bounding boxes in the following format:

    For each mouse, on each frame, store locations and confidence scores (see below) in an array, eg:
    mouse_1 = [locations_mouse_1, confidences_mouse_1]
    mouse_2 = [locations_mouse_2, confidences_mouse_2]
    ...

    Locations should be 4 values giving the upper-left and lower-right corners of the bounding box on that frame,
     [xmin, ymin, xmax, ymax]
    as fractions of the image width and height (ie all values should be bounded between 0 and 1).
    The confidence score reflects how sure you are of the bounding box, and is a scalar ranging from 0 to 1. If
    you don't have confidence scores for your bounding boxes, just set this to 1 for all mice/frames.

    For each frame, package the values from each mouse into an array:
    detected_mice_frame1 = [mouse_1, mouse_2,...]

    Then package these values into an array with an entry for each frame, and return this array:
    detected_mice_all_frames = [detected_mice_frame1, detected_mice_frame2,... detected_mice_frameN]
    '''

    # Add an elif to this group to recognize your own 'bboxType' string and call your custom unpacking script:
    if not opts['bboxType']:
        raise ValueError('Please specify a bounding box type')
    elif opts['bboxType'] == 'Chen_detection':
        detected_mice_all_frames = unpack_bbox_Chen(video_path, IM_W, IM_H, NUM_FRAMES)
    else:
        raise ValueError('Unrecognized bboxType.')

    return detected_mice_all_frames


def unpack_bbox_Chen(video_path, IM_W, IM_H, NUM_FRAMES):
    ''' an example script that loads user-supplied bounding boxes from a csv file. View code for a walk-through of
    creating your own custom bounding box unpacker for integration with MARS.'''

    ''' 
    First, given the video path, we must figure out which bounding box file to load. In this case, bounding boxes are
    stored in the same folder as the video, in a csv file with _bboxes appended to the video name. You can use any
    system you like, but be sure to be consistent so that MARS can find the right file!
    '''
    fullpath,_ = os.path.splitext((video_path))
    bbox_file = fullpath+'_bboxes.csv'

    # exit if bbox_file doesn't exist where we expected it:
    if not os.path.isfile(bbox_file):
        raise ValueError('Couldn''t locate bounding box file at '+bbox_file)

    '''
    Now, unpack the bounding boxes by looping over frames, and for each frame looping over mice.
    As you can see by inspecting the csv, bounding boxes are saved in units of pixels, which we'll need to convert
    to fractional coordinates, and are saved as [x_min, y_min, width, height], which we'll need to convert to
    [x_min, y_min, x_max, y_max]. No confidence values are provided for bounding boxes, so we'll also need to decide
    what confidence values to return for each mouse/frame.
    '''
    detected_mice_all_frames = [None]*NUM_FRAMES
    with open(bbox_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        frame_number = 0
        for line_count,row in enumerate(csv_reader):
            if line_count == 0:
                # the first row of our csv contains headers, which we mostly ignore.
                # each mouse gets 4 columns of data, so divide by 4 to get number of mice tracked in this file
                num_mice = int(len(row)/4)
            else:
                all_mice_in_frame = [None]*num_mice
                for mouse in range(num_mice):
                    # read the 4 bounding-box values for each mouse
                    bbox_raw = [float(i) for i in row[mouse*4 : (mouse+1)*4]]

                    # reformat them to match the format MARS is expecting ([x_min, y_min, x_max, y_max])
                    locations = [bbox_raw[0]/IM_W,
                                 bbox_raw[1]/IM_H,
                                 (bbox_raw[0]+bbox_raw[2])/IM_W,
                                 (bbox_raw[1]+bbox_raw[3])/IM_H]
                    # make up a confidence score since it wasn't provided by the user. Confidence ranges from 0 to 1,
                    # with 1 being most confident; when confidence in a bounding box is low (<0.005), MARS will discard
                    # that bounding box and instead use the one from the previous frame.
                    # Here, we'll make confidence either 1 if a bounding box was provided, or 0 if all four bbox
                    # coordinates were 0 (which happens when the mouse is not visible.)
                    if all(coord == 0 for coord in locations):
                        confidence = 0
                    else:
                        confidence = 1

                    # add the current mouse to the array of tracked mice. We wrap this twice in an array because MARS
                    # allows detectors to pass multiple guesses of bounding boxes for each mouse- there's a post-
                    # processing function that will refine those guesses to a final choice. But since we're not making
                    # multiple guesses here, we just pass the one bounding box.
                    this_mouse = [[[locations]], [[confidence]]]
                    all_mice_in_frame[mouse] = this_mouse

                # add tracking data for this frame to the master array
                detected_mice_all_frames[frame_number] = all_mice_in_frame
                frame_number += 1

    return detected_mice_all_frames