import os
import xlwt


def get_supported_formats():
    return ['.seq', '.avi', '.mpg', '.mp4']


def get_names(video_name, pair_files=False):
    """Given a video name:
        If pair_files=False, it just returns the video name for top_name and mouse_name.
        If pair_file=True, returns the name for the front video, top video, and mouse name.
        To use pair_files=True, movie filenames must contain a camera position (_Top or _Front).
    """
    if not pair_files:  # if we don't need to match up cameras, don't impose any naming restrictions on videos
        if any(x in video_name for x in get_supported_formats()):
            return video_name, video_name, video_name
        else:
            return '', '', ''

    # this is legacy code to support various ways we distinguished Top vs Front-view video during development.
    if ('Top_J85.seq' in video_name):
        mouse_name = video_name[:-12]
    elif ('_t.seq' in video_name):
        mouse_name = video_name[:-6]
    elif ('_Top.' in video_name):
        mouse_name = os.path.splitext(video_name)[0]
        mouse_name = mouse_name.replace('_Top', '')

    elif ('Front_J85.seq' in video_name):
        mouse_name = video_name[:-14]
    elif ('_s.seq' in video_name):
        mouse_name = video_name[:-6]
    elif ('_Front.' in video_name):
        mouse_name = os.path.splitext(video_name)[0]
        mouse_name = mouse_name.replace('_Front', '')
    elif ('FroHi.seq' in video_name):
        mouse_name = video_name[:-10]

    else:
        return '', '', ''  # couldn't find any indication of camera position in the filename- skip!

    # Find the ending suffix for each video.
    if 'J85' in video_name:
        top_ending = '_Top_J85.seq'
        front_ending = '_Front_J85.seq'
    elif any(x in video_name for x in ['_s.seq', '_t.seq']):
        top_ending = '_t.seq'
        front_ending = '_s.seq'
    else:
        ext = os.path.splitext(video_name)[1]
        top_ending = '_Top' + ext
        front_ending = '_Front' + ext

    top_name = mouse_name + top_ending
    front_name = mouse_name + front_ending
    return front_name, top_name, mouse_name


def getdir(dirname):
    """If a directory doesn't exists, makes it."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return


def get_version_suffix():
    """Returns the suffix for this version of MARS."""
    return "v1_8"


def get_mouse_output_dir(dir_output_should_be_in, video_name, output_suffix=''):
    """Returns the output directory for the specific mouse we're using."""
    if not output_suffix:
        # Default suffix is just the version number.
        output_suffix = get_version_suffix()

    # Get the name of the mouse from the video.
    _, _, mouse_name = get_names(video_name)

    output_basename = 'output'

    output_name = '_'.join([output_basename, output_suffix])

    mouse_output_dir = os.path.join(dir_output_should_be_in, output_name, mouse_name)
    return mouse_output_dir


# TODO: Could wrap the no_ext's up into a single function, since they carry out basically the same stuff.
def get_pose_no_ext(video_fullpath, output_folder, view, output_suffix):
    """Gives the name of the pose file that should be in a given directory, without the extension on the end."""
    if not output_suffix:
        # Default suffix is just the version number.
        output_suffix = get_version_suffix()

    video_name = os.path.basename(video_fullpath)
    _, _, mouse_name = get_names(video_name)

    # # Generates the name of the output directory
    output_plus_mouse = os.path.join(output_folder, mouse_name)

    # Puts it in pose_(view)_(suffix) format
    pose_designator = '_'.join(['pose', view, output_suffix])

    pose_basename = output_plus_mouse + '_' + pose_designator
    return pose_basename


def get_most_recent(models, behavior):
    # Get all the models that predict the given behavior.
    models_with_this_behavior = filter(lambda x: os.path.splitext(x)[0].endswith('classifier_' + behavior), models)
    if models_with_this_behavior:
        # Pick the most recently trained model for this behavior.
        name_n_timestamp = dict([(x, os.stat(x).st_mtime) for x in models_with_this_behavior])

        model_name = max(name_n_timestamp, key=lambda k: name_n_timestamp.get(k))
    else:
        raise ValueError('I couldn''t find a classifier for ' + behavior + ' among models ' + ';'.join(models))

    return model_name


def get_feat_no_ext(opts, video_fullpath='', output_folder='', output_suffix=''):
    # (video_fullpath, output_folder, view, output_suffix=''):
    """Gives the name of the feature file that should be in a given directory, without the extension on the end."""
    if not output_suffix:
        output_suffix = get_version_suffix()

    # video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)
    video_front_name, video_top_name, mouse_name = get_names(video_name,
                                                             pair_files=opts['hasTopCamera'] and opts['hasFrontCamera'])
    # TODO: add support for feature extraction from top+front cameras
    view = 'top' if opts['hasTopCamera'] and opts['hasFrontCamera'] \
        else 'top' if opts['hasTopCamera'] \
        else 'front' if opts['hasFrontCamera'] \
        else 'none'

    # Generates the name of the output directory
    output_plus_mouse = os.path.join(output_folder, mouse_name)

    feat_basenames = {}
    classifier_path = opts['classifier_model']
    models = [filename for filename in os.listdir(classifier_path)]
    for behavior in opts['classify_behaviors']:  # figure out what kind of features we need for each classifier

        model_name = get_most_recent(models, behavior)
        clf = joblib.load(os.path.join(classifier_path, model_name))
        if 'project_config' in clf['params'].keys():
            feature_type = 'custom'  # TODO: specify the exact feature set to extract in the classifier(?) config file.
        elif 'pcf' in classifier_path:
            feature_type = 'raw_pcf'  # older mars classifiers don't have an associated project config saved with them.
        else:
            feature_type = 'raw'

        # Puts it in (type)_(view)_(suffix) format
        feat_designator = '_'.join([feature_type, 'feat', view, output_suffix])
        feat_basenames.update({behavior: output_plus_mouse + '_' + feat_designator})

    return feat_basenames


def get_clf_type(classifier_path):
    model_type = 'xgb' if 'xgb' in classifier_path else 'mlp'
    if 'topfront' in classifier_path:
        model_type += '_topfront'
    elif 'top_pcf' in classifier_path:
        model_type += '_top_pcf'
    else:
        model_type += '_top'
    model_type += '_wnd' if 'wnd' in classifier_path else ''
    return model_type


def get_classifier_savename(video_fullpath, output_folder, view='top', classifier_path='', output_suffix='',
                            model_type='xgb'):
    if not output_suffix:
        output_suffix = get_version_suffix()

    video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)
    video_front_name, video_top_name, mouse_name = get_names(video_name)

    # # Generates the name of the output directory
    output_plus_mouse = os.path.join(output_folder, mouse_name)

    # TODO: Parse this from input.
    classifier_basename = model_type
    # Puts it in pose_(view)_(suffix) format
    classifier_designator = '_'.join([classifier_basename, 'actions_pred', output_suffix])
    classifier_basename = output_plus_mouse + '_' + classifier_designator
    classifier_txtname = classifier_basename + '.txt'
    return classifier_txtname


def dump_bento_across_dir(root_path):
    ''' This function makes a bento file for a specific directory.'''
    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('Sheet1', cell_overwrite_ok=True)
    ws1.write(0, 0, os.path.abspath(root_path))  # A1
    ws1.write(0, 1, 'Ca framerate:')  # B1
    ws1.write(0, 2, 0)  # C1
    ws1.write(0, 3, 'Annot framerate:')  # D1
    ws1.write(0, 4, 30)  # E1
    ws1.write(0, 5, 'Multiple trials/Ca file:')  # F1
    ws1.write(0, 6, 0)  # G1
    ws1.write(0, 7, 'Multiple trails/annot file')  # H1
    ws1.write(0, 8, 0)  # I1
    ws1.write(0, 9, 'Includes behavior movies:')  # J1
    ws1.write(0, 10, 1)  # K1
    ws1.write(0, 11, 'Offset (in seconds; positive values = annot starts before Ca):')  # L1
    ws1.write(0, 12, 0)  # M1
    ws1.write(0, 13, 'Includes tracking data:')
    ws1.write(0, 14, 0)
    ws1.write(0, 15, 'Includes audio files:')
    ws1.write(0, 16, 0)

    ws1.write(1, 0, 'Mouse')  # A2
    mouse_col_num = 0
    session_col_num = 1
    trial_col_num = 2
    ws1.write(1, 1, 'Sessn')  # B2
    ws1.write(1, 2, 'Trial')  # C2
    ws1.write(1, 3, 'Stim')  # D2
    ws1.write(1, 4, 'Calcium imaging file')  # E2
    ws1.write(1, 5, 'Start Ca')  # F2
    ws1.write(1, 6, 'Stop Ca')  # G2
    ws1.write(1, 7, 'FR Ca')  # H2
    ws1.write(1, 8, 'Alignments')  # I2
    ws1.write(1, 9, 'Annotation file')  # J2
    annot_file_col_num = 9
    ws1.write(1, 10, 'Start Anno')  # K2
    ws1.write(1, 11, 'Stop Anno')  # L2
    ws1.write(1, 12, 'FR Anno')  # M2
    ws1.write(1, 13, 'Offset')  # N2
    ws1.write(1, 14, 'Behavior movie')  # O2
    behavior_movie_col_num = 14
    ws1.write(1, 15, 'Tracking')  # P2
    tracking_file_col_num = 15
    ws1.write(1, 16, 'Audio file')
    audio_file_col_num = 16
    ws1.write(1, 17, 'tSNE')

    # ws1.write(2, 0, 1)  # A2
    # ws1.write(2, 1, 1)  # B2
    # ws1.write(2, 2, 1)  # C2
    # ws1.write(2, 3, '')  # D2
    # ws1.write(2, 4, '')  # E2
    # ws1.write(2, 5, '')  # F2
    # ws1.write(2, 6, '')  # G2
    # ws1.write(2, 7, '')  # H2
    # ws1.write(2, 8, '')  # I2
    mouse_number = 0
    row_num = 2
    # Going through everything in this directory.
    audio_filenames = []
    add_audio_count = 0
    nonaudio_filenames = []
    for path, subdirs, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            fname = os.path.join(path, fname)
            if fname.endswith('.wav'):
                audio_filenames.append(fname)
            else:
                nonaudio_filenames.append(fname)

        audio_filenames = sorted(audio_filenames)
        nonaudio_filenames = sorted(nonaudio_filenames)

    for fname in nonaudio_filenames:
        try:
            cond1 = all(x not in fname for x in get_supported_formats())
            cond2 = 'skipped' in path

            if (cond1) | cond2:
                continue

            if any(x in fname for x in ['Top', '_t']):
                front_fname, top_fname, mouse_name = get_names(fname)
                fullpath_to_front = os.path.join(path, front_fname)
                fullpath_to_top = os.path.join(path, top_fname)
            else:
                # This is a seq file, but doesnt have "Top" or "Front" in it. Let's skip it.
                continue

            # Add their info to the bento file at the appropriate level.

            video_fullpath = fullpath_to_top

            output_suffix = ''
            video_path = os.path.dirname(video_fullpath)
            video_name = os.path.basename(video_fullpath)

            # Get the output folder for this specific mouse.
            output_folder = get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                                 output_suffix=output_suffix)
            _, _, mouse_name = get_names(video_name=video_name)

            pose_basename = get_pose_no_ext(video_fullpath=video_fullpath,
                                            output_folder=output_folder,
                                            view='top',
                                            output_suffix=output_suffix)

            top_pose_fullpath = pose_basename + '.json'

            same_path_ann = [os.path.join(root_path, f)
                             for f in os.listdir(root_path) if is_annotation_file(f, mouse_name)]

            ann = [os.path.join(output_folder, f)
                   for f in os.listdir(output_folder) if is_annotation_file(f, mouse_name)]

            ann = sorted(ann)
            ann = [get_normrel_path(f, root_path) for f in ann]

            pose_cond = os.path.exists(top_pose_fullpath)
            video_cond = os.path.exists(video_fullpath)

            should_write = (pose_cond and video_cond)

            if should_write:
                old_mouse_number = mouse_number
                mouse_number = get_mouse_number(video_fullpath)

                mouse_cond = (old_mouse_number == mouse_number)
                # TODO: Session condition
                sess_cond = (True)

                if mouse_cond and sess_cond:
                    trial_count += 1
                else:
                    trial_count = 1

                ws1.write(row_num, mouse_col_num, mouse_number)  # A2
                ws1.write(row_num, session_col_num, 1)  # B2
                ws1.write(row_num, trial_col_num, trial_count)  # C2

                ws1.write(row_num, annot_file_col_num, ';'.join(ann))  # J2
                ws1.write(row_num, 10, '')  # K2
                ws1.write(row_num, 11, '')  # L2
                ws1.write(row_num, 12, '')  # M2
                ws1.write(row_num, 13, '')  # N2

                track_file = get_normrel_path(top_pose_fullpath, root_path)

                ws1.write(row_num, behavior_movie_col_num, get_normrel_path(fullpath_to_top, root_path))  # O2
                ws1.write(row_num, tracking_file_col_num, track_file)  # P2
                row_num += 1
        except Exception as e:
            print(e)
            error_msg = 'ERROR: ' + fname + ' has failed. ' + str(e)

            continue
            # End of try-except block
            # End of particular fname
            # End of the particular root_path

    last_row = row_num
    row_num = 2
    for audio_file_count, audio_file in enumerate(audio_filenames):
        # Write the files in order.
        ws1.write(row_num + audio_file_count, audio_file_col_num + 2, get_normrel_path(audio_file, root_path))

    bento_name = 'bento_root_path' + get_version_suffix() + '.xls'
    wb.save(os.path.join(root_path, bento_name))
    return


def get_normrel_path(path, start=''):
    return '/' + (os.path.relpath(path, start))


def is_annotation_file(filename, mouse_name):
    cond1 = filename.startswith(mouse_name)
    cond2 = filename.endswith('.txt')
    cond3 = os.path.exists(filename)
    return (cond1 and cond2)
