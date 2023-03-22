from metaflow import FlowSpec, step
from metaflow import Parameter
#from submodules.bias_free_denoising.data.preprocess_bsd400 import data_augmentation, Im2Patch
import numpy as np
import os


__here__ = os.path.dirname(__file__)

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

def _process_image(subset,img_path,scale_floats, patch_size, stride,aug_times):
    from PIL import Image
    import numpy as np
    raw_img = Image.open(img_path)
    # print(f"image shape: {img.size}")
    if subset == "test":
        return [ np.expand_dims(np.array(raw_img), 0) / 255.0]

    h, w = raw_img.size
    processed_images = list()
    raw_img.show()
    for k,v in enumerate(scale_floats):
        img = raw_img.resize((int(h * v),int(w * v)),resample=3)
        img = np.array(img)

        img = np.expand_dims(img[:, :].copy(), 0) / 255.0
        patches = Im2Patch(img, win=patch_size, stride=stride)
        for n in range(patches.shape[3]):
            data = patches[:, :, :, n].copy()
            processed_images.append(data)
            for m in range(aug_times - 1):
                data_aug = data_augmentation(data, np.random.randint(1, 8))
                processed_images.append(data_aug)
    return processed_images

class LinearFlow(FlowSpec):
    data_path = Parameter(
        "data-path",
        help="",
        type=str,
        default=os.path.abspath(os.path.join(__here__, "../../submodules/bias_free_denoising/data/")),
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
    debug = Parameter(
        "debug",
        help="",
        default=False,
        type=bool,
    )
    @step
    def start(self):
        from glob import glob
        import os
        print(f" data path: {self.data_path}")
        train_files =[("train", x) for x in sorted(glob(os.path.join(self.data_path, "Train400", "*.png")))]
        test_files = [("test", x) for x in sorted(glob(os.path.join(self.data_path, "Test/Set12", "*.png")))]

        if self.debug:
            train_files = train_files[:5]
            test_files = test_files[:5]

        self.scale_floats = [float(sc) for sc in self.scales.split(",")]

        self.files = np.array_split(train_files + test_files,50)
        self.next(self.process_images,foreach="files")

    @step
    def process_images(self):
        self.processed_images = {"train": list(), "test": list()}
        for k,v in self.input:
            self.processed_images[k].extend(_process_image(img_path=v,
                                                           scale_floats=self.scale_floats,
                                                           patch_size=self.patch_size,
                                                           stride=self.stride,
                                                           aug_times=self.aug_times,
                                                           subset=k
                                                           ))

        self.next(self.join)

    @step
    def join(self,inputs):
        self.training_set  = []
        self.test_set = []
        for fanout in inputs:
            self.training_set.extend(fanout.processed_images["train"])
            self.test_set.extend(fanout.processed_images["test"])

        self.next(self.end)

    @step
    def end(self):
        pass
if __name__ == '__main__':
    LinearFlow()