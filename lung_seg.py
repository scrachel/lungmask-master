from lungmask import mask
import SimpleITK as sitk
# import radiomics
import numpy as np
import os
import torch

# =============================================================================
# #spyder import failed.
# import sys
# base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(base_path)
# from lungmask import mask
# =============================================================================

def Resegment(input_image, roi_array, roi_value):
    '''
    Parameters
    ----------
    input_image : sitk.Image, sitk image of the data file.
    roi_array : np.array, sitk array of the roi file.
    roi_value: list of int, ROI of which value to extract.
                1 for the left lung while 2 for the right one.

    Returns
    -------
    roi_reseg : sitk.Image, data image after the resegment.
    '''
    
    im_arr = sitk.GetArrayFromImage(input_image)

    # Original method
    # ma_arr = (roi_array >= 1)  # boolean array
    # ma_arr = (roi_array == roi_value)  # boolean array

    # value list method
    ma_arr = np.zeros_like(im_arr, dtype=int)
    for roi_v in roi_value:
        ma_arr += (roi_array == roi_v)  # 判断语句 非直接叠加
    ma_arr = (ma_arr == 1)

    newMask_arr = np.zeros_like(im_arr, dtype=int)
    newMask_arr = newMask_arr - 1024
    newMask_arr[ma_arr] = im_arr[ma_arr]
  
    newMask = sitk.GetImageFromArray(newMask_arr)
    newMask.CopyInformation(input_image)
    
    return newMask


if __name__ == '__main__':
    # # TEST MODEL 1
    # # image_dir = r'D:\Work\Data\rjyy_yangwenjie\lung_in_ex\rjl\20210421113631.000'
    # # input_image = sitk.ReadImage(image_dir + r'\8-inspi.nii.gz')
    #
    # image_dir = input('Input the directory:')
    # file_name = input('Input the file name')
    # input_image = sitk.ReadImage(os.path.join(image_dir, file_name))
    # output_image = os.path.join(image_dir, file_name)[: -7] + '_roi.nii.gz'
    #
    # # model = mask.get_model('unet', 'LTRCLobes')
    # # segmentation = mask.apply(input_image, model)
    # # model is LTRCLobes
    #
    # segmentation = mask.apply(input_image)  # np.array
    # # default model is U-net(R231)
    #
    # # segmentation = mask.apply_fused(input_image)
    # # LTRCLobes_R231
    #
    # # Output an roi map of the original image
    # # output_mask = Resegment(input_image, segmentation, [1])
    # # sitk.WriteImage(output_mask, image_dir + r'\8_in_roi.nii.gz')
    #
    # # lung lobe segment
    # # value_list = [[1], [2], [3], [4, 5]]  # 4/5肺叶分割不良 合并计算
    # # out_name_list = ['_1', '_2', '_3', '_45']
    # #
    # # for i in range(len(value_list)):
    # #     output_mask = Resegment(input_image, segmentation, value_list[i])
    # #     sitk.WriteImage(output_mask, image_dir + '\\Ex_roi'+out_name_list[i]+'.nii.gz')
    #
    # # Output the segmentation roi
    # segmentation_image = sitk.GetImageFromArray(segmentation)
    # segmentation_image.CopyInformation(input_image)
    # sitk.WriteImage(segmentation_image, output_image)
    # # sitk.WriteImage(segmentation_image, os.path.join(image_dir, 'lung_seg_roi_fast.nii.gz'))

    # TEST MODEL 2
    import pandas as pd
    input_csv = input('Input the csv file:\n')
    # D:\Work\Data\Ruijin\LungSeg\Ruijin_filename.csv
    df = pd.read_csv(input_csv)
    # input_image = sitk.ReadImage(df['nii_name'][0])
    # output_image = df['nii_name'][0][: -7]+'_roi.nii.gz'
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    for x in df['nii_name']:
        input_image = sitk.ReadImage(x)
        output_image = x[: -7]+'_roi.nii.gz'
        segmentation = mask.apply(input_image)  # np.array
        segmentation_image = sitk.GetImageFromArray(segmentation)
        segmentation_image.CopyInformation(input_image)
        sitk.WriteImage(segmentation_image, output_image)