class AdaCoresetMethod(object):
    def __init__(self, dst_train, network, args, fraction=0.5, random_seed=None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.model = network
        # for all training
        # self.num_classes = len(dst_train.classes)
        # for tl
        self.num_classes = args.num_classes
        # when using random split
        # self.num_classes = len(dst_train.dataset.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return

