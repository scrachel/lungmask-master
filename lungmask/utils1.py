import os

data_extensions = [
    '.nii.gz',
]
roi_extensions = [
    'roi.nii.gz',
]


def is_image_file(filename, mode='data'):
    if mode == 'roi':
        return any(filename.endswith(extension) for extension in roi_extensions)
    elif mode == 'data':
        if not 'roi' in filename and \
                any(filename.endswith(extension) for extension in data_extensions):
            return True
        else:
            return False
    else:
        raise ValueError('Undefined mode %s while reading data' % mode)


def make_dataset(dir, max_dataset_size=float("inf"), mode='data'):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        # return of os.walk: root dir, folders, files
        for fname in fnames:
            if is_image_file(fname, mode):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

if __name__ == '__main__':
    dir = r'D:\Work\Data\TestData4Unet\newdata'
    img = make_dataset(dir, mode='roi')
