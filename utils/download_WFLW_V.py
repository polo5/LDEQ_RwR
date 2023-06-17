import concurrent.futures
import argparse
import pytube
import os
import time
import cv2
import pandas as pd


def download_and_crop_youtube_video(video_ID, start_frame, end_frame, print_index):

    tmp_folder_path = os.path.join(args.output_folder, 'temp')
    video_folder_path = os.path.join(args.output_folder, 'videos')
    video_path_tmp = os.path.join(tmp_folder_path, f'{video_ID}.mp4')
    video_path = os.path.join(video_folder_path, f'{video_ID}.mp4')
    video_url = f"https://www.youtube.com/watch?v={video_ID}"

    if os.path.isfile(video_path):
        print(f'skipping video {print_index}/1000 since already processed')

    else:
        print(f'processing video {print_index}/1000')

        ## Downloading
        # print(f'Downloading video {video_ID}')
        video = pytube.YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
        stream = video.streams.get_highest_resolution()
        stream.download(output_path=os.path.join(args.output_folder, 'temp'), filename=f'{video_ID}.mp4', skip_existing=False, max_retries=5)
        time.sleep(0.1)

        ## Cropping
        # print(f'Cropping video {video_ID}')
        video_capture = cv2.VideoCapture(video_path_tmp)
        im_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, (im_width, im_height))

        frame_idx = -1

        while True:
            _, frame = video_capture.read()
            frame_idx += 1
            if frame_idx>=start_frame and frame_idx<=end_frame:
                video_writer.write(frame)
            elif frame_idx>end_frame:
                video_capture.release()
                # cv2.destroyAllWindows() #early versions of opencv trigger error
                break

        os.remove(video_path_tmp)

def main(args):

    df = pd.read_csv('./datasets/WFLW_V/download_ranges.txt', sep=' ', header=None)
    tmp_folder_path = os.path.join(args.output_folder, 'temp')
    video_folder_path = os.path.join(args.output_folder, 'videos')
    if not os.path.isdir(tmp_folder_path): os.mkdir(tmp_folder_path)
    if not os.path.isdir(video_folder_path): os.mkdir(video_folder_path)

    # Extract column 2 as a list
    video_IDs = df[0].astype(str).tolist()
    start_frames = df[1].astype(int).tolist()
    end_frames = df[2].astype(int).tolist()
    indexes = [i for i in range(1,len(video_IDs)+1)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_processes) as executor:
        for _ in executor.map(download_and_crop_youtube_video, video_IDs, start_frames, end_frames, indexes):
            pass

    os.remove(tmp_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEQ Inference')
    parser.add_argument('--output_folder', type=str, default='path/to/WFLW')
    parser.add_argument('--n_processes', type=int, default=32)

    args = parser.parse_args()

    print('\nStarting...')

    main(args)
