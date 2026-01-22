import logging
import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, write_domain_img_file2txt, split_domain_txt2txt
from utils.datautils.core50data import CORE50


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    train_data = None
    train_targets = None
    test_data = None
    test_targets = None


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)


class iCore50(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_split_ratio = 0.7
    train_session_seq = ['s11', 's4', 's2', 's9', 's1', 's6', 's5', 's8']
    test_session_seq = ['s3', 's7', 's10']
    test_split_file_prefix = 'TEST-set'

    def __init__(self, args, split_train4test=False, split_test4test=False):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order

        self.split_train4test = split_train4test
        self.split_test4test = split_test4test

    def download_data(self):
        train_session_id = self.args.get('task_name', list(range(8)))

        self.train_session_seq = [self.train_session_seq[i] for i in train_session_id]
        logging.info("Training sequence of domains: {}".format(self.train_session_seq))

        datagen = CORE50(root=self.args["data_path"], scenario="ni", train_seq=train_session_id)

        dataset_list = []
        img_list = []
        label_list = []

        if not self.split_train4test:
            for i, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch  # img_list_: RGB image data
                print("Loading {}: {} ".format(self.train_session_seq[i], len(img_list_)))
                label_list_ += i * 50
                img_list_ = img_list_.astype(np.uint8)
                # dataset_list.append([imglist, labellist])
                img_list.append(img_list_)
                label_list.append(label_list_)
            train_x = np.concatenate(img_list)
            train_y = np.concatenate(label_list)
            self.train_data = train_x
            self.train_targets = train_y

            test_x, test_y = datagen.get_test_set()
            test_x = test_x.astype(np.uint8)
            self.test_data = test_x
            self.test_targets = test_y
        else:
            self.write2txt()
            train_idx_lst, test_idx_lst = self.read2array()
            train_img_lst = []
            train_label_arr = np.array([], dtype=np.int32)
            test_img_lst = []
            test_label_arr = np.array([], dtype=np.int32)

            for i, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch  # img_list_: RGB image data
                print("Loading {}: {} ".format(self.train_session_seq[i], len(img_list_)))
                label_list_ += i * 50
                img_list_ = img_list_.astype(np.uint8)

                train_img_lst.append(img_list_[train_idx_lst[i]])
                train_label_arr = np.append(train_label_arr, label_list_[train_idx_lst[i]])
                test_img_lst.append(img_list_[test_idx_lst[i]])
                test_label_arr = np.append(test_label_arr, label_list_[test_idx_lst[i]])

            if self.split_test4test:
                img_arr, label_arr = datagen.get_test_set()
                img_arr = img_arr.astype(np.uint8)
                train_img_lst.append(img_arr[train_idx_lst[-1]])
                train_label_arr = np.append(train_label_arr, label_arr[train_idx_lst[-1]])
                test_img_lst.append(img_arr[test_idx_lst[-1]])
                test_label_arr = np.append(test_label_arr, label_arr[test_idx_lst[-1]])

            self.train_data = np.concatenate(train_img_lst)
            self.train_targets = train_label_arr

            self.test_data = np.concatenate(test_img_lst)
            self.test_targets = test_label_arr

    def write2txt(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        if self.split_test4test and os.path.exists(os.path.join(self.args["data_path"],
                                                                self.test_split_file_prefix + '_test_idx.txt')):
            return

        if not os.path.exists(os.path.join(self.args["data_path"], self.train_session_seq[-1]+'_test_idx.txt')):
            print("Writing train-test split index to txt file ...")

            for idx, train_batch in enumerate(datagen):
                img_list_, label_list_ = train_batch
                label_list_ += idx * 50

                train_idx_ = np.random.choice(len(img_list_), int(len(img_list_) * self.train_split_ratio), replace=False)
                test_idx_ = np.setdiff1d(np.arange(len(img_list_)), train_idx_)
                with open(os.path.join(self.args["data_path"], self.train_session_seq[idx]+'_train_idx.txt'), 'w') as f:
                    for i in train_idx_:
                        f.write(str(i) + '\n')
                with open(os.path.join(self.args["data_path"], self.train_session_seq[idx]+'_test_idx.txt'), 'w') as f:
                    for i in test_idx_:
                        f.write(str(i) + '\n')

        if self.split_test4test:
            test_x, test_y = datagen.get_test_set()
            train_idx_ = np.random.choice(len(test_x), int(len(test_x) * self.train_split_ratio), replace=False)
            test_idx_ = np.setdiff1d(np.arange(len(test_x)), train_idx_)
            with open(os.path.join(self.args["data_path"], self.test_split_file_prefix + '_train_idx.txt'), 'w') as f:
                for i in train_idx_:
                    f.write(str(i) + '\n')
            with open(os.path.join(self.args["data_path"], self.test_split_file_prefix + '_test_idx.txt'), 'w') as f:
                for i in test_idx_:
                    f.write(str(i) + '\n')

    def read2array(self):
        def _read2array(seq, train_idx:list, test_idx:list):
            for idx in seq:
                with open(os.path.join(self.args["data_path"], idx+'_train_idx.txt'), 'r') as f:
                    train_idx.append([np.int32(i) for i in f.readlines()])
                with open(os.path.join(self.args["data_path"], idx+'_test_idx.txt'), 'r') as f:
                    test_idx.append([np.int32(i) for i in f.readlines()])
            return train_idx, test_idx

        train_idx, test_idx = [], []
        train_idx, test_idx = _read2array(self.train_session_seq, train_idx, test_idx)

        if self.split_test4test:
            train_idx, test_idx = _read2array(self.test_session_seq, train_idx, test_idx)

        return train_idx, test_idx


class iDomainNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = (np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.nb_sessions = args["total_sessions"]
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        def _read_data(image_list_paths) -> (np.ndarray, np.ndarray):
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                with open(image_list_path) as f:
                    image_list = f.readlines()
                # 重写 target class := original value + taskid * args["increment"]
                for entry in image_list:
                    img_label = int(entry.split()[1])
                    if self.class_incremental:
                        if img_label < taskid * self.cl_n_inc or img_label >= (taskid + 1) * self.cl_n_inc:
                            continue
                    elif img_label > self.cl_n_inc:
                        raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                    else:  # correct the label for DIL tasks
                        img_label = img_label + taskid * self.cl_n_inc
                    imgs.append((entry.split()[0], img_label))

            img_x, img_y = [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])

            return np.array(img_x), np.array(img_y)

        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)


class iOfficeHome(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = (np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['Art', "Clipart", "Product", "Real_World"]
        logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = []

        for d in self.domain_names:
            write_domain_img_file2txt(self.image_list_root, d)
            split_domain_txt2txt(self.image_list_root, d, train_ratio=0.7, seed=self.args['seed'])

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]

        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            # imgs: (relative_path, label)
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
        test_x, test_y = [], []
        for item in imgs:
            test_x.append(os.path.join(self.image_list_root, item[0]))
            test_y.append(item[1])
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
