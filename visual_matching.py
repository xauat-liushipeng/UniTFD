import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    """
    计算两个张量的余弦相似度，a和b形状均为(N, D)或(1, D)
    返回形状 (N,)
    """
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    sim = torch.mm(a_norm, b_norm.t()).squeeze()
    return sim

def match_features(exemplar_feat, query_feats, metric='cosine'):
    """
    exemplar_feat: [1, D]
    query_feats: [N, D]
    返回每个query_feat和exemplar_feat的相似度分数
    """
    if metric == 'cosine':
        scores = cosine_similarity(query_feats, exemplar_feat)  # [N]
    elif metric == 'l2':
        dist = torch.cdist(query_feats, exemplar_feat, p=2).squeeze()
        scores = -dist  # 距离越小，相似度越高
    else:
        raise ValueError(f"Unsupported metric {metric}")
    return scores

if __name__ == '__main__':
    import torch
    e = torch.randn(1, 512)
    q = torch.randn(5, 512)
    s = match_features(e, q)
    print(s)
