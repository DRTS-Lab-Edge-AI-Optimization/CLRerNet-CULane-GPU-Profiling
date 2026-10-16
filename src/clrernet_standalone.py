# ===== CLRerNet (DLA34) – standalone, inference-only PyTorch re-implementation =====
# Mirrors hirotomusiker/CLRerNet (libs/models/...) module-for-module so the official
# checkpoint loads with strict=True, but without mmcv / mmdet / the CUDA NMS extension.
# Every block is wrapped in SECTION(...) so we can time / count FLOPs / bytes per part.
import math
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


# ----------------------------------------------------------------------------- profiling hook
class SectionRecorder:
    """Hierarchical named sections around blocks of the forward pass.
    Disabled (default) -> zero cost, so the same model is used for eval / export.
    Enabled -> CUDA events per section (perf_counter on CPU); optionally also
    torch.profiler.record_function labels and NVTX ranges (for Nsight Systems)."""

    def __init__(self):
        self.enabled = False
        self.use_record_function = False
        self.use_nvtx = False
        self.stack = []
        self.pending = []
        self.times = defaultdict(list)      # "backbone/level3" -> [ms, ...]
        self.order = {}                     # first-seen order (= forward order)

    @property
    def current(self):
        return "/".join(self.stack)

    @contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        self.stack.append(name)
        full = self.current
        self.order.setdefault(full, len(self.order))
        use_cuda = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() is False
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
        else:
            s, e = time.perf_counter(), None
        if self.use_nvtx and use_cuda:
            torch.cuda.nvtx.range_push(full)
        ctx = torch.profiler.record_function(full) if self.use_record_function else nullcontext()
        try:
            with ctx:
                yield
        finally:
            if self.use_nvtx and use_cuda:
                torch.cuda.nvtx.range_pop()
            if e is not None:
                e.record()
                self.pending.append((full, s, e))
            else:
                self.times[full].append((time.perf_counter() - s) * 1e3)
            self.stack.pop()

    def collect(self):
        if self.pending:
            torch.cuda.synchronize()
            for full, s, e in self.pending:
                self.times[full].append(s.elapsed_time(e))
            self.pending.clear()

    def reset(self):
        self.pending.clear()
        self.times.clear()
        self.stack.clear()


SECTION = SectionRecorder()


# ----------------------------------------------------------------------------- DLA-34 backbone
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.bn(self.conv(torch.cat(x, 1)))
        if self.residual:
            x = x + children[0]
        return self.relu(x)


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False,
                 root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        self.levels = levels
        if levels == 1 and in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, block=BasicBlock, out_indices=(2, 3, 4, 5), residual_root=False):
        super().__init__()
        self.channels = channels
        self.out_indices = out_indices
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        with SECTION("base_layer"):
            x = self.base_layer(x)
        for i in range(6):
            with SECTION(f"level{i}"):
                x = getattr(self, f"level{i}")(x)
            if i in self.out_indices:
                y.append(x)
        return y


class DLANet(nn.Module):
    def __init__(self, levels=(1, 1, 1, 2, 2, 1), in_channels=(16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.model = DLA(levels=list(levels), channels=list(in_channels), block=BasicBlock)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------------- mmcv ConvModule (minimal)
class ConvModule(nn.Module):
    """conv -> (bn) -> (relu); parameter names match mmcv.cnn.ConvModule."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias="auto", norm=False, act=True):
        super().__init__()
        if bias == "auto":
            bias = not norm
        self.with_norm, self.with_act = norm, act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if act:
            self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.activate(x)
        return x


# ----------------------------------------------------------------------------- FPN neck
class CLRerNetFPN(nn.Module):
    def __init__(self, in_channels=(128, 256, 512), out_channels=64):
        super().__init__()
        self.in_channels = list(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for c in in_channels:
            self.lateral_convs.append(ConvModule(c, out_channels, 1, norm=False, act=False))
            self.fpn_convs.append(ConvModule(out_channels, out_channels, 3, padding=1, norm=False, act=False))

    def forward(self, inputs):
        inputs = list(inputs)[-len(self.in_channels):]          # drop level2 output (stride 4)
        with SECTION("lateral_1x1"):
            laterals = [l(inputs[i]) for i, l in enumerate(self.lateral_convs)]
        with SECTION("top_down"):
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:],
                                                                  mode="nearest")
        with SECTION("fpn_3x3"):
            outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return tuple(outs)


# ----------------------------------------------------------------------------- head pieces
class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, self.size).flatten(2)


class AnchorVecFeatureMapAttention(nn.Module):
    def __init__(self, n_query, dim):
        super().__init__()
        self.dim = dim
        self.resize = FeatureResize()
        self.f_key = ConvModule(dim, dim, 1, norm=True, act=True)
        self.f_query = nn.Sequential(nn.Conv1d(n_query, n_query, 1, groups=n_query), nn.ReLU())
        self.f_value = nn.Conv2d(dim, dim, 1)
        self.W = nn.Conv1d(n_query, n_query, 1, groups=n_query)

    def forward(self, roi, fmap):
        query = self.f_query(roi)                                 # [B, Np, C]
        key = self.resize(self.f_key(fmap))                       # [B, C, 250]
        value = self.resize(self.f_value(fmap)).permute(0, 2, 1)  # [B, 250, C]
        sim_map = F.softmax((self.dim ** -0.5) * torch.matmul(query, key), dim=-1)
        context = torch.matmul(sim_map, value)                    # [B, Np, C]
        return self.W(context)


class ROIGather(nn.Module):
    def __init__(self, in_channels=64, num_priors=192, sample_points=36, fc_hidden_dim=64,
                 refine_layers=3, mid_channels=48):
        super().__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.attention = AnchorVecFeatureMapAttention(num_priors, in_channels)
        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(ConvModule(in_channels, mid_channels, (9, 1), padding=(4, 0), bias=False, norm=True))
            self.catconv.append(ConvModule(mid_channels * (i + 1), in_channels, (9, 1), padding=(4, 0),
                                           bias=False, norm=True))
        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def forward(self, roi_features, fmap, layer_index):
        bs = fmap.size(0)
        with SECTION("roi_conv_fc"):
            feats = [self.convs[i](f) for i, f in enumerate(roi_features)]
            roi = self.catconv[layer_index](torch.cat(feats, dim=1))
            roi = roi.reshape(bs * self.num_priors, -1)
            roi = F.relu(self.fc_norm(self.fc(roi))).reshape(bs, self.num_priors, -1)
        with SECTION("cross_attention"):
            roi = roi + self.attention(roi, fmap)                 # dropout = identity at inference
        return roi


class CLRerNetAnchorGenerator(nn.Module):
    def __init__(self, num_priors=192, num_points=72):
        super().__init__()
        self.num_priors = num_priors
        self.num_offsets = num_points
        self.prior_embeddings = nn.Embedding(num_priors, 3)     # learned (y0, x0, theta), loaded from ckpt

    def generate_anchors(self, anchor_params, prior_ys, sample_x_indices, img_w, img_h):
        """anchor_params [..., 3] -> anchor xs [..., 72] and sampled xs [..., 36] (normalised)."""
        x0 = anchor_params[..., 1:2] * (img_w - 1)
        y0 = anchor_params[..., 0:1]
        theta = anchor_params[..., 2:3] * math.pi
        tan_lane = torch.tan(theta + 1e-5)
        anchor_xs = (x0 - (prior_ys - y0) * img_h / tan_lane) / (img_w - 1)
        sampled_xs = torch.index_select(anchor_xs, -1, sample_x_indices)
        return anchor_xs, sampled_xs


class CLRerHead(nn.Module):
    def __init__(self, img_w=800, img_h=320, num_priors=192, num_points=72, prior_feat_channels=64,
                 fc_hidden_dim=64, num_fc=2, refine_layers=3, sample_points=36):
        super().__init__()
        self.anchor_generator = CLRerNetAnchorGenerator(num_priors, num_points)
        self.img_w, self.img_h = img_w, img_h
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.prior_feat_channels = prior_feat_channels
        self.register_buffer("sample_x_indices",
                             (torch.linspace(0, 1, steps=sample_points, dtype=torch.float32) * self.n_strips).long())
        self.register_buffer("prior_feat_ys", torch.flip((self.sample_x_indices.float() / self.n_strips), dims=[-1]))
        self.register_buffer("prior_ys", torch.linspace(1, 0, steps=num_points, dtype=torch.float32))
        reg_modules, cls_modules = [], []
        for _ in range(num_fc):
            reg_modules += [nn.Linear(fc_hidden_dim, fc_hidden_dim), nn.ReLU(inplace=True)]
            cls_modules += [nn.Linear(fc_hidden_dim, fc_hidden_dim), nn.ReLU(inplace=True)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(fc_hidden_dim, num_points + 4)
        self.cls_layers = nn.Linear(fc_hidden_dim, 2)
        self.attention = ROIGather(prior_feat_channels, num_priors, sample_points, fc_hidden_dim, refine_layers)

    def pool_prior_features(self, batch_features, prior_xs):
        """grid_sample features along each prior: [B,C,H,W] x [B,Np,Ns] -> [B*Np, C, Ns, 1]."""
        b = batch_features.shape[0]
        prior_xs = prior_xs.reshape(b, self.num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.reshape(1, 1, -1, 1).expand(b, self.num_priors, -1, 1)
        grid = torch.cat((prior_xs * 2.0 - 1.0, prior_ys * 2.0 - 1.0), dim=-1)
        feature = F.grid_sample(batch_features, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=True).permute(0, 2, 1, 3)
        return feature.reshape(b * self.num_priors, self.prior_feat_channels, self.sample_points, 1)

    def forward(self, x, return_all_stages=False):
        b = x[0].shape[0]
        feature_pyramid = list(x[len(x) - self.refine_layers:])[::-1]   # [10x25, 20x50, 40x100]
        ag = self.anchor_generator
        with SECTION("anchor_init"):
            emb = ag.prior_embeddings.weight
            _, sampled_xs = ag.generate_anchors(emb, self.prior_ys, self.sample_x_indices, self.img_w, self.img_h)
            anchor_params = emb.unsqueeze(0).expand(b, -1, -1)               # [B, Np, 3]
            priors_on_featmap = sampled_xs.unsqueeze(0).expand(b, -1, -1)    # [B, Np, Ns]
        preds, pooled_stages = [], []
        for stage in range(self.refine_layers):
            with SECTION(f"stage{stage}"):
                with SECTION("roi_pool(grid_sample)"):
                    pooled_stages.append(self.pool_prior_features(feature_pyramid[stage], priors_on_featmap))
                fc_features = self.attention(pooled_stages, feature_pyramid[stage], stage)
                fc_features = fc_features.reshape(b * self.num_priors, self.fc_hidden_dim)
                with SECTION("cls_reg_fc"):
                    cls_f, reg_f = fc_features, fc_features
                    for m in self.cls_modules:
                        cls_f = m(cls_f)
                    for m in self.reg_modules:
                        reg_f = m(reg_f)
                    cls_logits = self.cls_layers(cls_f).reshape(b, -1, 2)
                    reg = self.reg_layers(reg_f).reshape(b, -1, self.n_offsets + 4)
                with SECTION("anchor_decode"):
                    anchor_params = anchor_params + reg[:, :, :3]
                    anchor_xs, sampled = ag.generate_anchors(anchor_params, self.prior_ys, self.sample_x_indices,
                                                             self.img_w, self.img_h)
                    xs = anchor_xs + reg[..., 4:]
                    preds.append((cls_logits, anchor_params, reg[:, :, 3:4], xs))
                    priors_on_featmap = sampled
        return preds if return_all_stages else preds[-1]


class CLRerNet(nn.Module):
    """Input: BGR float tensor in [0,1], shape [B,3,320,800] (rows 270:590 of the 1640x590 frame).
    Output (final refine stage): cls_logits [B,192,2], anchor_params [B,192,3], lengths [B,192,1], xs [B,192,72]."""

    def __init__(self):
        super().__init__()
        self.backbone = DLANet()
        self.neck = CLRerNetFPN()
        self.bbox_head = CLRerHead()

    def forward(self, img):
        with SECTION("backbone"):
            feats = self.backbone(img)
        with SECTION("neck"):
            feats = self.neck(feats)
        with SECTION("head"):
            return self.bbox_head(feats)


def load_clrernet(ckpt_path, device="cpu"):
    """Build the model and load the official checkpoint with strict=True."""
    model = CLRerNet()
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("bbox_head.seg_decoder")}   # training-only aux head
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()