
import cv2
import numpy as np
import torch
import math
import itertools

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale #*200 -- for floating point range? We divide by 200 in crop_img anyways.
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_transform_from_bbox(bbox, extra_scale=1.2, target_im_size=256):
    """
    bbox must have x1, y1, x2, y2 as first 4 entries
    """
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32) #round up, +0.5 to put pixel zero in middle of first pixel

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) #max of height or width
    scale = scale * extra_scale

    transform_matrix = get_affine_transform(center=center, scale=scale, rot=0.0, output_size=(target_im_size, target_im_size))

    return transform_matrix


def apply_affine_transform_to_kpts(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ np.linalg.inv(trans[:,0:2].T)
    return pt

def apply_affine_transform_to_kpts_torch(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ torch.linalg.inv(trans[:,0:2].T)
    return pt

def draw_bbox(bbox, image, bgr=(255,0,0), thickness=5):
    # print(bbox[:4])
    sx, sy, ex, ey = [int(i) for i in bbox[:4]]
    start_point = (sx, sy)
    end_point = (ex, ey)
    image = cv2.rectangle(image, start_point, end_point, bgr, thickness)

    return image

def draw_landmark(landmark, image, bgr=(0,255,0), linewidth=2):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), linewidth, bgr, -1)
    return image

# def draw_arrows(start_points)


def get_triangle_vertices_from_center(point=(0,0), size=2):
    x0, y0 = point
    # vertices = np.array([ #pointing down in image coordinate frame
    #     [x0, y0+1.5*size],
    #     [x0-size, y0-size],
    #     [x0+size, y0-size]
    # ])
    vertices = np.array([ #pointing up
        [x0, y0-1.5*size],
        [x0-size, y0+size],
        [x0+size, y0+size]
    ])
    return vertices #floating point


def draw_landmark_as_triangle(landmark, image, bgr=(0,255,0), linewidth=2):
    for (x, y) in landmark:
        vertices = get_triangle_vertices_from_center((x,y), size=linewidth)
        vertices = (vertices+0.5).astype(np.int32)
        cv2.drawContours(image, [vertices], 0, bgr, -1)
    return image

def hsv_to_rgb(h, s, v):
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)


def translate_english_to_other(english_string_list, target_language='fr'):

    import googletrans
    from googletrans import Translator
    assert target_language in googletrans.LANGUAGES

    translator = Translator()
    output_strings = []
    for string in english_string_list:
        try:#depending on googletrans version this can fail
            translation = translator.translate(string, src='en', dest=target_language).text
            output_strings.append(translation)
        except:
            print(f'Failed to translate english string {string} to target language {target_language}. Check googletrans version if this error is too frequent')
            continue

    return output_strings


def cv2_images_to_grid(list_of_images, margins=(2,2), n_images=(5,5), scale_to_max_dim=True, max_dim=(1080,720)):

    # n_images_width = n_images_height = math.ceil(np.sqrt(len(list_of_images)))
    n_images_width, n_images_height = n_images
    assert n_images_width*n_images_height >= len(list_of_images)

    img_h, img_w, img_c = list_of_images[0].shape
    m_x, m_y = margins

    new_height = img_h * n_images_height + m_y * (n_images_height - 1)
    new_width = img_w * n_images_width + m_x * (n_images_width - 1)

    imgmatrix = np.zeros((new_height, new_width , img_c), np.uint8)
    imgmatrix.fill(255)
    # print(imgmatrix.shape)

    # positions = itertools.product(range(n_images_width), range(n_images_height))
    positions = itertools.product(range(n_images_height), range(n_images_width))
    # for (x_i, y_i), img in zip(positions, list_of_images):
    for (y_i, x_i), img in zip(positions, list_of_images):
        # for y_i in range(n_images_height):
        #     for x_i in range(n_images_width):
        #
        #
        print(x_i, y_i)
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y + img_h, x:x + img_w, :] = img

    ## resize final image
    if scale_to_max_dim:
        r_w, r_h = new_width/max_dim[0], new_height/max_dim[1]
        r = max(r_w, r_h)
        imgmatrix = cv2.resize(imgmatrix, (int(new_width/r), int(new_height/r)), interpolation = cv2.INTER_AREA)

    return imgmatrix



if __name__=='__main__':

    ########
    # print(hsv_to_rgb(359,1,1))
    # N = 60
    # colors = [hsv_to_rgb(x*1.0/N, 1, 1) for x in range(N)]
    # print(colors)

    ########
    import googletrans
    print(googletrans.LANGUAGES)
    english_strings = ['How to play the violon', 'makeup tutorial for girls']
    print(translate_english_to_other(english_strings, target_language='nl'))
    # print(translate_english_to_other(english_strings, target_language='af'))
