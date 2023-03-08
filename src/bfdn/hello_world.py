from metaflow import FlowSpec, step
from metaflow import Parameter
#from submodules.bias_free_denoising.data.preprocess_bsd400 import data_augmentation, Im2Patch
import numpy as np

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
class LinearFlow(FlowSpec):
    data_path = Parameter(
        "data-path",
        help="",
        type=str,
        default=None,
    )
    scales = Parameter(
        "scales",
        help="",
        default="1,0.9,0.8,0.7",
        type=str,
    )
    patch_size = Parameter(
        "patch-size",
        help="",
        default=50,
        type=int,
    )
    stride = Parameter(
        "stride",
        help="",
        default=10,
        type=int,
    )
    aug_times = Parameter(
        "aug-times",
        help="",
        default=2,
        type=int,
    )

    @step
    def start(self):
        from glob import glob
        import os
        train_files =[("train", x) for x in sorted(glob(os.path.join(self.data_path, "Train400", "*.png")))][:5]
        test_files = [("test", x) for x in sorted(glob(os.path.join(self.data_path, "Set12", "*.png")))][:5]

        self.scale_floats = [float(sc) for sc in self.scales.split(",")]

        self.files = train_files + test_files
        self.next(self.process_images,foreach="files")

    @step
    def process_images(self):
        import cv2
        import numpy as np
        subset, img = self.input
        img = cv2.imread(img)
        if subset == "test":
            self.processed_images = [ np.expand_dims(img[:, :, 0], 0) / 255.0]
        else:
            h, w, c = img.shape
            self.processed_images = list()

            for k,v in enumerate(self.scale_floats):
                Img = cv2.resize(img, (int(h * v), int(w * v)), interpolation=cv2.INTER_CUBIC)
                Img = np.expand_dims(Img[:, :, 0].copy(), 0) / 255.0
                patches = Im2Patch(Img, win=self.patch_size, stride=self.stride)
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    self.processed_images.append(data)
                    for m in range(self.aug_times - 1):
                        data_aug = data_augmentation(data, np.random.randint(1, 8))
                        self.processed_images.append(data_aug)


        self.next(self.join)

    @step
    def join(self,inputs):
        self.training_set  = []
        self.test_set = []
        for fanout in inputs:
            if fanout.input[0] == "train":
                self.training_set.extend(fanout.processed_images)
            else:
                self.test_set.extend(fanout.processed_images)

        self.next(self.end)

    @step
    def end(self):
        print('the data artifact is still: %s' % self.my_var)

if __name__ == '__main__':
    LinearFlow()