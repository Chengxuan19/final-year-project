import numpy as np
import skimage.measure
from utils.denoising_utils import *
from utils.common_utils import *


def find_x_square_index(input_img_np):
    mean = np.mean(input_img_np)
    sum_t = 0
    for i in range(128):
        for j in range(128):
            sum_t += (input_img_np[0][i][j]-mean)**2
    return sum_t/16384      



def find_x_y_index(input_img_np_one,input_img_np_two):
    mean_one = np.mean(input_img_np_one)
    mean_two = np.mean(input_img_np_two)
    sum_t = 0
    for i in range(128):
        for j in range(128):
            sum_t += (input_img_np_one[0][i][j]-mean_one)*(input_img_np_two[0][i][j]-mean_two)
    return sum_t/16384


def find_Q_zero(input_img_np_one,input_img_np_two):
    mean_one = np.mean(input_img_np_one)
    mean_two = np.mean(input_img_np_two)
    theta_xy = find_x_y_index(input_img_np_one,input_img_np_two)
    theta_x_square = find_x_square_index(input_img_np_one)
    theta_y_square = find_x_square_index(input_img_np_two)
    result = (4*theta_xy*mean_one*mean_two)/(((mean_one**2)+(mean_two**2))*(theta_x_square+theta_y_square))
    return result

def find_fusion_index(input_img_one,input_img_two,fusion_result):
    entropy_one = skimage.measure.shannon_entropy(input_img_one)
    entropy_two = skimage.measure.shannon_entropy(input_img_two)
    lambda_one = entropy_one/(entropy_one+entropy_two)
    lambda_two = 1-lambda_one

    img_original_np = pil_to_np(fusion_result)
    img_up_np = pil_to_np(input_img_one)
    img_down_np = pil_to_np(input_img_two)
    
    q_zero_one = find_Q_zero(img_up_np,img_original_np)
    q_zero_two = find_Q_zero(img_down_np,img_original_np)

    result = lambda_one * q_zero_one + lambda_two * q_zero_two

    return result    