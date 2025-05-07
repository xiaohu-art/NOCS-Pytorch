import torch.nn as nn
from .utils import SamePad2d, pyramid_roi_align

class Nocs_head_bins_wt_unshared(nn.Module):
    '''
    NOCS Class for binning. Weights are seperate, meaning x,y,z predictions will not share weights
    '''

    def __init__(self,depth, pool_size,image_shape, num_classes, num_bins, net_name):
        super(Nocs_head_bins_wt_unshared, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_bins=num_bins
        self.net_name=net_name
        self.depth=depth

        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, self.num_bins * self.num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        # x (input) : (batch_size, num_rois, 4)
        # x (output) : (num_rois, num_classes, num_bins, mask_height, mask_width)
        # x_feature : (num_rois, num_classes*num_bins, mask_height, mask_width)

        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x_feature = self.deconv(x)
        x = self.relu(x_feature)
        x = self.conv5(x)

        x=x.view(x.shape[0], -1,self.num_bins, x.shape[2], x.shape[3])
        x = self.softmax(x)

        return x,x_feature
    
class NocsHeadBinsShared(nn.Module):
    """
    One head that predicts per-pixel bins for the three NOCS coordinates.
    The conv blocks (conv1-conv4 + deconv) are shared; only the **very last
    1×1 projection** is duplicated so each coordinate still has its own logits.
    This is the usual ‘shared trunk + task-specific classifier’ pattern.
    """

    def __init__(self, depth, pool_size, image_shape,
                 num_classes, num_bins):
        super().__init__()
        self.pool_size   = pool_size
        self.image_shape = image_shape
        self.num_bins    = num_bins
        self.num_classes = num_classes
        self.depth       = depth

        self.pad = SamePad2d(kernel_size=3, stride=1)

        # ── shared trunk ──────────────────────────────────────
        self.conv1 = nn.Conv2d(depth, 256, 3, 1)
        self.bn1   = nn.BatchNorm2d(256, eps=1e-3)
        self.conv2 = nn.Conv2d(256, 256, 3, 1)
        self.bn2   = nn.BatchNorm2d(256, eps=1e-3)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.bn3   = nn.BatchNorm2d(256, eps=1e-3)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.bn4   = nn.BatchNorm2d(256, eps=1e-3)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, 2)

        # ── *three* task-specific 1×1 heads ───────────────────
        out_ch = num_bins * num_classes          # per coordinate
        self.head_x = nn.Conv2d(256, out_ch, 1)  # W= out_ch
        self.head_y = nn.Conv2d(256, out_ch, 1)
        self.head_z = nn.Conv2d(256, out_ch, 1)

        self.relu    = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)         # over the bins

    def _trunk(self, x):
        """Shared trunk forward pass."""
        x = self.relu(self.bn1(self.conv1(self.pad(x))))
        x = self.relu(self.bn2(self.conv2(self.pad(x))))
        x = self.relu(self.bn3(self.conv3(self.pad(x))))
        x = self.relu(self.bn4(self.conv4(self.pad(x))))
        x = self.relu(self.deconv(x))
        return x                                 # (N,256,H,W)

    def _project_and_reshape(self, t, head):
        """Apply 1×1 head, then reshape to (N, C, B, H, W)."""
        t = head(t)                              # (N, C*B, H, W)
        N, CB, H, W = t.shape
        t = t.view(N, self.num_classes, self.num_bins, H, W)
        return self.softmax(t)                  # softmax over bins

    def forward(self, pyramid_feats, rois):
        """
        Returns three tensors:
            bins_x/y/z  : (R, C, B, h, w)
            feat_trunk  : (R, 256, h, w)   (for potential auxiliary losses)
        """
        # ROI-align once ⇒ shared trunk ⇒ split
        t = pyramid_roi_align([rois] + pyramid_feats,
                              self.pool_size, self.image_shape)
        t = self._trunk(t)

        bins_x = self._project_and_reshape(t, self.head_x)
        bins_y = self._project_and_reshape(t, self.head_y)
        bins_z = self._project_and_reshape(t, self.head_z)
        return (bins_x, bins_y, bins_z), t

    
class CoordBinValues(nn.Module):
    '''
    Module to convert NOCS bins to values in range [0,1]
    '''

    def __init__(self, coord_num_bins):
        super(CoordBinValues, self).__init__()
        self.coord_num_bins = coord_num_bins

    def forward(self, mrcnn_coord_bin):

        mrcnn_coord_bin_value = mrcnn_coord_bin.argmax(dim = 2) / self.coord_num_bins

        return mrcnn_coord_bin_value


