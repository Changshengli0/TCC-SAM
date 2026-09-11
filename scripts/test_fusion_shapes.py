import torch
from model.ow_fusion_cat_attr import FusionCatAttr


def main():
    B = 2
    L = 8
    d_text = 1024
    C = 256
    K = 6
    H = 32
    W = 32

    H_fused = torch.randn(B, L, d_text)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    cls_vec = torch.randn(B, d_text)
    F_img = torch.randn(B, C, H, W)

    fusion = FusionCatAttr(d_text=d_text, C=C, K=K, gamma_cls=0.2, gate_type="residual", wordtype_head="mlp")
    route_tokens, debug = fusion(H_fused, attention_mask, cls_vec, F_img)

    assert route_tokens.shape == (B, K, C), route_tokens.shape
    print("route_tokens", route_tokens.shape)
    print("G_prob", debug["G_prob"].shape)
    print("F_A", debug["F_A"].shape)


if __name__ == "__main__":
    main()
