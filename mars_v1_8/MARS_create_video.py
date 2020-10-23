import os,sys
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import json
from matplotlib import cm as cm
import matplotlib.colors as col
import matplotlib.colorbar as cbar
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import progressbar
sys.path.append('./')
from util.seqIo import seqIo_reader,parse_ann_dual
import MARS_output_format as mof
import multiprocessing as mp
import cv2
import pdb

def create_video_results_wrapper(top_video_fullpath,classifier_path,
                                 progress_bar_signal,
                                 view='top',
                                 doOverwrite=0,output_suffix=''):

    try:
        video_fullpath = top_video_fullpath
        video_path = os.path.dirname(video_fullpath)
        video_name = os.path.basename(video_fullpath)

        model_type = mof.get_clf_type(classifier_path=classifier_path)


        # Get the output folder for this specific mouse.
        output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                                 output_suffix=output_suffix)

        # Get the pose and feature files' names.
        pose_basename = mof.get_pose_no_ext(video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view='top',
                                            output_suffix=output_suffix)
        top_pose_fullpath = pose_basename + '.json'

        # Get the name of the text file we're going to save to.
        classifier_savename = mof.get_classifier_savename(video_fullpath=top_video_fullpath,
                                                          output_folder=output_folder,
                                                          view=view,
                                                          classifier_path=classifier_path,
                                                          output_suffix=output_suffix,model_type=model_type)

        # print(classifier_savename)
        predictions_exists=os.path.exists(classifier_savename)
        top_pose_exist = os.path.exists(top_pose_fullpath)
        video_savename = classifier_savename[:-3]+'mp4'
        video_exists=os.path.exists(video_savename)
        # print(video_savename)
        if not top_pose_exist:
            raise ValueError("No pose has been extracted for this video!")
        if not predictions_exists:
            raise ValueError("No behavior classified for this video!")
        if (not video_exists) | doOverwrite:
            pool = mp.Pool(1)
            manager = mp.Manager()
            queue_for_progress_bar = manager.Queue(20)
            result = pool.apply_async(create_mp4_prediction, (video_fullpath,top_pose_fullpath,classifier_savename,video_savename, queue_for_progress_bar))
            still_frames_left = True
            while still_frames_left:
                progress_bar_input = queue_for_progress_bar.get()
                if progress_bar_input:
                    still_frames_left = True
                    progress_bar_signal.emit(progress_bar_input[0], progress_bar_input[1])
                else:
                    still_frames_left = False
                    break

            pool.close()
            pool.join()
            result.get()
        else:
            print("4 - Video already exists")
            return

    except Exception as e:
        print(e)
        raise (e)
    return


def create_mp4_prediction(top_video_fullpath,
                          top_pose_fullpath,
                          pred_file_fullpath,
                          video_savename,
                          queue_for_number_processed):
    try:
        #top video and pose info
        video_name = os.path.basename(top_video_fullpath)
        with open(top_pose_fullpath, 'r') as fp:
            frames_pose = json.load(fp)
        num_pt = len(frames_pose['scores'][0][0])
        keypoints = frames_pose['keypoints']
        scores = frames_pose['scores']

        ext = top_video_fullpath[-3:]

        if ext == 'seq':
            srTop = seqIo_reader(top_video_fullpath)
            IM_TOP_W = srTop.header['width']
            IM_TOP_H = srTop.header['height']
            fps = srTop.header['fps']
        elif any(x not in ext for x in ['avi', 'mpg']):
            vc = cv2.VideoCapture(top_video_fullpath)
            if vc.isOpened():
                rval = True
            else:
                rval = False
                print('video not readable')
            fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
            if np.isnan(fps): fps = 30.
            IM_TOP_W = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            IM_TOP_H = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        num_frames = len(keypoints)

        actions_pred = parse_ann_dual(pred_file_fullpath)
        ####################################################################
        plt.ioff()
        fig = plt.figure(figsize=(float(IM_TOP_W-150) / 100.0, float(IM_TOP_H + 100) / 100.0), dpi=100, frameon=False)
        fig.patch.set_visible(False)
        gs1 = gridspec.GridSpec(9, 12)
        gs1.update(left=0, right=1, hspace=0.5,bottom=0.05, top=1,wspace=0.)
        gs1.tight_layout(fig)
        ax = fig.add_subplot(gs1[:-2, :])
        # ax = plt.subplot2grid((5, 5), (0, 0), colspan=5, rowspan=4)
        # ax.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim([0, IM_TOP_W])
        ax.set_ylim([IM_TOP_H, 0])
        ax.set_axis_off()
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        im = ax.imshow(np.ones((IM_TOP_H, IM_TOP_W)), cmap='gray', interpolation='nearest')
        im.set_clim([0, 1])

        # prediction behaviors
        # ax2 = plt.subplot2grid((5, 5), (4, 1), colspan=3, rowspan=1)
        ax2  = fig.add_subplot(gs1[-2, 1:-2])
        ax2.set_xlim([0, 300])
        ax2.set_ylim([20, 0])
        # ax2.set_axis_off()
        ax2.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)

        # prediction interactions
        ax3  = fig.add_subplot(gs1[-1, 1:-2])
        ax3.set_xlim([0, 300])
        ax3.set_ylim([20, 0])
        # x2.set_axis_off()
        ax3.axes.get_yaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)

        num_pt = len(keypoints[0][0][0])

        # mouse 1 and 2 body part dots
        pts = []
        colors = ['y' for i in range(num_pt)] + ['b' for i in range(num_pt)]
        for i in range(num_pt * 2):
            pt = mpatches.Ellipse((0, 0), 2, 2, 0, color='none', ec=colors[i], fc =colors[i] )
            ax.add_patch(pt)
            pts.append(pt)

        #mouse 1 lines
        l11 =  Line2D([100,200], [100,200], linewidth=2, color = 'r');  ax.add_line(l11)
        l12 =  Line2D([100,200], [100,200], linewidth=2, color = '#ffa07a');  ax.add_line(l12)
        l13 =  Line2D([100,200], [100,200], linewidth=2, color = 'yellow');  ax.add_line(l13)
        l14 =  Line2D([100,200], [100,200], linewidth=2, color = 'orange');  ax.add_line(l14)
        l15 =  Line2D([100,200], [100,200], linewidth=2, color = 'orange');  ax.add_line(l15)
        l16 =  Line2D([100,200], [100,200], linewidth=2, color = '#ff1493');  ax.add_line(l16)
        l17 =  Line2D([100,200], [100,200], linewidth=2, color = '#ff1493');  ax.add_line(l17)

        #mouse 3 lines
        l21 =  Line2D([100,200], [100,200], linewidth=2, color = '#000080');  ax.add_line(l21)
        l22 =  Line2D([100,200], [100,200], linewidth=2, color = '#6495ed');  ax.add_line(l22)
        l23 =  Line2D([100,200], [100,200], linewidth=2, color = 'b');  ax.add_line(l23)
        l24 =  Line2D([100,200], [100,200], linewidth=2, color = 'chartreuse');  ax.add_line(l24)
        l25 =  Line2D([100,200], [100,200], linewidth=2, color = 'chartreuse');  ax.add_line(l25)
        l26 =  Line2D([100,200], [100,200], linewidth=2, color = 'cyan');  ax.add_line(l26)
        l27 =  Line2D([100,200], [100,200], linewidth=2, color = 'cyan');  ax.add_line(l27)

        # add predicted labels to subplot
        label2id = {'other': 0,
                'cable_fix': 0,
                'intruder_introduction': 0,
                'corner': 0,
                'ignore': 0,
                'groom': 0,
                'groom_genital': 0,
                'grooming': 0,
                'tailrattle': 0,
                'tail_rattle': 0,
                'tailrattling': 0,
                'intruder_attacks': 0,
                'approach': 0,

                'closeinvestigate': 1,
                'closeinvestigation': 1,
                'investigation': 1,
                'sniff_genitals': 1,
                'sniff-genital': 1,
                'sniff-face': 1,
                'sniff-body': 1,
                'sniffurogenital': 1,
                'sniffgenitals': 1,
                'agg_investigation': 1,
                'sniff_face': 1,
                'anogen-investigation': 1,
                'head-investigation': 1,
                'sniff_body': 1,
                'body-investigation': 1,
                'socialgrooming': 1,

                'mount': 2,
                'aggressivemount': 2,
                'mount_attempt': 2,
                'intromission': 2,

                'attack': 3,

                'interaction':4

                    }

        if 'interaction' in actions_pred['behs_frame']:
            pred_lab = actions_pred['behs_frame2']
            pred_lab_int = actions_pred['behs_frame']
        else:
            pred_lab = actions_pred['behs_frame']
            pred_lab_int = actions_pred['behs_frame2']

        labels_pred=[]
        for i in pred_lab:
            labels_pred.append(label2id[i]) if i in label2id else labels_pred.append(0)
        labels_pred.insert(0,0)

        labels_pred_int=[]
        for i in pred_lab_int:
            labels_pred_int.append(label2id[i]) if i in label2id else labels_pred_int.append(0)
        labels_pred_int.insert(0,0)

        # define the colormap
        # cmap = cm.jet
        # extract all colors from the .jet map
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        # cmaplist[-1] = (.5,.5,.5,1.0)
        # create the new map
        # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # cmap = cm.ScalarMappable(col.Normalize(0, len(class_names)), cm.jet)
        # cmap = cmap.to_rgba(range(len(class_names)))
        # cmap = mpl.colors.ListedColormap(cmap)

        class_names = ['other', 'closeinvestigate', 'mount', 'attack','interaction']
        colors=['white','blue','green','red','grey']
        cmap = ListedColormap(colors)
        cmap.set_over('0.25')
        cmap.set_under('0.75')
        # define the bins and normalize
        bounds = np.linspace(0,5,6)
        norm = col.BoundaryNorm(bounds, cmap.N)

        # behs pred
        pd_mat = np.tile(np.array(labels_pred[:300]),(20,1))
        im2 = ax2.imshow(pd_mat,cmap=cmap,norm=norm)
        pd_ind = mpatches.Rectangle((0,0), 1.5, 20, angle=0.0, color='none', ec='black',linewidth=1)
        ax2.add_patch(pd_ind)
        pd_text = ax2.text(148, -2, "0")
        pd_tag = ax2.text(-30,12, "BEHS")

        # interaction pred
        pd_mat_int = np.tile(np.array(labels_pred_int[:300]),(20,1))
        im3 = ax3.imshow(pd_mat_int,cmap=cmap,norm=norm)
        # gt_line = Line2D([IM_TOP_W/2,IM_TOP_W/2],[0,50],color='r', linewidth=3); ax2.add_line(gt_line)
        pd_ind_int = mpatches.Rectangle((0,0), 1.5, 20, angle=0.0, color='none', ec='black',linewidth=1)
        ax3.add_patch(pd_ind_int)
        pd_tag_int = ax3.text(-30, 12, "INTER")

        # colorbar
        ticks = class_names
        # create a second axes for the colorbar
        ax4= fig.add_axes([0.85, 0.03, 0.015, 0.2])
        # ax3.set_axis_off()
        cb = cbar.ColorbarBase(ax4, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds+.5, boundaries=bounds)
        cb.ax.tick_params(labelsize=8)
        cb.ax.set_yticklabels(ticks)

        bar = progressbar.ProgressBar(widgets=
                                      [progressbar.FormatLabel('frame %(value)d'), '/' , progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
                                       progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=len(keypoints))
        bar.start()
        queue_for_number_processed.put([0,num_frames-1])

        def animate(f):
            queue_for_number_processed.put([f,0])
            # get current frame's mouse data
            x1 = np.array(keypoints[f][0][0])
            y1 = np.array(keypoints[f][0][1])
            x2 = np.array(keypoints[f][1][0])
            y2 = np.array(keypoints[f][1][1])
            s1 = np.array(scores[f][0])
            s2 = np.array(scores[f][1])

            if ext == 'seq':
                frame = srTop.getFrame(f)[0]
            else:
                vc.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f)
                _, frame = vc.read()
                frame = frame.astype(np.float32)
            im.set_data(frame / 256.)

            for i in range(7):
                ms1 = min(np.sqrt(s1[i]) * 15.0, 15.0)
                pts[i].center = (x1[i], y1[i])
                pts[i].width = ms1
                pts[i].height = ms1
                ms2 = min(np.sqrt(s2[i]) * 15.0, 15.0)
                pts[i + 7].center = (x2[i], y2[i])
                pts[i + 7].width = ms2
                pts[i + 7].height = ms2

            l11.set_data([x1[1], x1[2]], [y1[1], y1[2]]);    l11.set_linewidth(1.5 if np.min(s1[1:3]) < 0.5 else 3.5)
            l12.set_data([x1[0], x1[1]], [y1[0], y1[1]]);    l12.set_linewidth(1.5 if np.min([s1[0],s1[2]]) < 0.5 else 3.5)
            l13.set_data([x1[0], x1[2]], [y1[0], y1[2]]);    l13.set_linewidth(1.5 if np.min([s1[3],s1[4]]) < 0.5 else 3.5)
            l14.set_data([x1[3], x1[4]], [y1[3], y1[4]]);    l14.set_linewidth(1.5 if np.min([s1[4],s1[6]]) < 0.5 else 3.5)
            l15.set_data([x1[4], x1[6]], [y1[4], y1[6]]);    l15.set_linewidth(1.5 if np.min([s1[0],s1[2]]) < 0.5 else 3.5)
            l16.set_data([x1[3], x1[5]], [y1[3], y1[5]]);    l16.set_linewidth(1.5 if np.min([s1[3],s1[5]]) < 0.5 else 3.5)
            l17.set_data([x1[5], x1[6]], [y1[5], y1[6]]);    l17.set_linewidth(1.5 if np.min([s1[5],s1[6]]) < 0.5 else 3.5)

            l21.set_data([x2[1], x2[2]], [y2[1], y2[2]]);    l21.set_linewidth(1.5 if np.min(s2[1:3]) < 0.5 else 3.5)
            l22.set_data([x2[0], x2[1]], [y2[0], y2[1]]);    l22.set_linewidth(1.5 if np.min([s2[0],s2[2]]) < 0.5 else 3.5)
            l23.set_data([x2[0], x2[2]], [y2[0], y2[2]]);    l23.set_linewidth(1.5 if np.min([s2[3],s2[4]]) < 0.5 else 3.5)
            l24.set_data([x2[3], x2[4]], [y2[3], y2[4]]);    l24.set_linewidth(1.5 if np.min([s2[4],s2[6]]) < 0.5 else 3.5)
            l25.set_data([x2[4], x2[6]], [y2[4], y2[6]]);    l25.set_linewidth(1.5 if np.min([s2[0],s2[2]]) < 0.5 else 3.5)
            l26.set_data([x2[3], x2[5]], [y2[3], y2[5]]);    l26.set_linewidth(1.5 if np.min([s2[3],s2[5]]) < 0.5 else 3.5)
            l27.set_data([x2[5], x2[6]], [y2[5], y2[6]]);    l27.set_linewidth(1.5 if np.min([s2[5],s2[6]]) < 0.5 else 3.5)

            pd_text.set_text(str(f))

            if f >= 300:
                pd_mat = np.tile(np.array(labels_pred[f-150:f+151]), (20, 1))
                im2.set_data(pd_mat)
                pd_ind.set_xy((150,0))
                pd_mat_int = np.tile(np.array(labels_pred_int[f - 150:f + 151]), (20, 1))
                im3.set_data(pd_mat_int)
                pd_ind_int.set_xy((150, 0))
            else:
                pd_ind.set_xy((f,0))
                pd_ind_int.set_xy((f,0))
            bar.update(f)
            return
        # print("animating")
        ani = FuncAnimation(fig, animate, repeat=False, blit=False, frames= num_frames)

        mywriter = FFMpegWriter(fps=fps)
        ani.save(video_savename, writer=mywriter, dpi=200)
        queue_for_number_processed.put([])
        bar.finish()
        plt.close()
        if ext == 'seq':        srTop.close()
        else: vc.release()
        return
    except Exception as e:
        print(e)
        raise(e)