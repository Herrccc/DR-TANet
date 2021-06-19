import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Temporal_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 groups=1, bias=False, refinement=False):
        super(Temporal_Attention, self).__init__()
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.refinement = refinement

        print('Attention Layer-kernel size:{0},stride:{1},padding:{2},groups:{3}...'.format(self.kernel_size,self.stride,self.padding,self.groups))
        if self.refinement:
            print("Attention with refinement...")

        assert self.outc % self.groups == 0, 'out_channels should be divided by groups.'

        self.w_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


        #relative positional encoding...
        self.rel_h = nn.Parameter(torch.randn(self.outc // 2, 1, 1, self.kernel_size, 1), requires_grad = True)
        self.rel_w = nn.Parameter(torch.randn(self.outc // 2, 1, 1, 1, self.kernel_size), requires_grad = True)
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


        init.kaiming_normal_(self.w_q.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_v.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, feature_map):

        fm_t0, fm_t1 = torch.split(feature_map, feature_map.size()[1]//2, 1)
        assert fm_t0.size() == fm_t1.size(), 'The size of feature maps of image t0 and t1 should be same.'

        batch, _, h, w = fm_t0.size()


        padded_fm_t0 = F.pad(fm_t0, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.w_q(fm_t1)
        k_out = self.w_k(padded_fm_t0)
        v_out = self.w_v(padded_fm_t0)

        if self.refinement:

            padding = self.kernel_size
            padded_fm_col = F.pad(fm_t0, [0, 0, padding, padding])
            padded_fm_row = F.pad(fm_t0, [padding, padding, 0, 0])
            k_out_col = self.w_k(padded_fm_col)
            k_out_row = self.w_k(padded_fm_row)
            v_out_col = self.w_v(padded_fm_col)
            v_out_row = self.w_v(padded_fm_row)

            k_out_col = k_out_col.unfold(2, self.kernel_size * 2 + 1, self.stride)
            k_out_row = k_out_row.unfold(3, self.kernel_size * 2 + 1, self.stride)
            v_out_col = v_out_col.unfold(2, self.kernel_size * 2 + 1, self.stride)
            v_out_row = v_out_row.unfold(3, self.kernel_size * 2 + 1, self.stride)


        q_out_base = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1).repeat(1, 1, 1, 1, 1, self.kernel_size*self.kernel_size)
        q_out_ref = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1).repeat(1, 1, 1, 1, 1, self.kernel_size * 2 + 1)
        
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.outc // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

        inter_out = (q_out_base * k_out).sum(dim=2)

        out = F.softmax(inter_out, dim=-1)
        out = torch.einsum('bnhwk,bnchwk -> bnchw', out, v_out).contiguous().view(batch, -1, h, w)

        if self.refinement:

            k_out_row = k_out_row.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            k_out_col = k_out_col.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_row = v_out_row.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_col = v_out_col.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

            out_row = F.softmax((q_out_ref * k_out_row).sum(dim=2),dim=-1)
            out_col = F.softmax((q_out_ref * k_out_col).sum(dim=2),dim=-1)
            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_row, v_out_row).contiguous().view(batch, -1, h, w)
            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_col, v_out_col).contiguous().view(batch, -1, h, w)

        return out







