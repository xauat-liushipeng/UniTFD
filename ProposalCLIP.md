**1.初始提案**

输入一张图像$I$，用EdgeBoxes提取初始提案$\{O_m\}_{m=1}^M$，$O_m \in R^4$表示坐标，对应初始得分为$SL_m$

**2.计算基于CLIP的目标得分**

CLIP提取特征为$V_M$，与每一类文本特征$\{T_c\}_{c=1}^C$计算相似度，跟一个softmax

计算CLIP相似熵$E_m = -\sum_{c=1}^C Sim_{m,c} \times \text{log} (Sim_{m,c})$

提出40%高熵值提案，剩下的为$\{O_t\}_{t=1}^T$

计算基于CLIP的目标得分 $S_t = - \frac{T}{C} \frac{E_t}{\sqrt{\sum_{t=1}^T E_t^2}} + \lambda_{sim} \max \limits_{c=1,...,C} Sim_{t,c} + \lambda_{sl}SL_t$，$\lambda_{sim}=0.06,\lambda_{sl}=1$

**3.基于graph的提案筛选**

无向图 $\mathcal{G}=<\mathcal{N}, \mathcal{\varepsilon}>$，节点$\mathcal{N}$是$\{O_t\}_{t=1}^T$，边$\mathcal{\varepsilon}$由空间与语义相似度计算

空间相似性用IoU，$IoU_{i,j}=\frac{O_i \cap O_j}{O_i \cup O_j}$

语义相似性$PSim_{i,j} = \frac{V_i \cdot V_j}{||V_i|| \cdot ||V_j||}$

$O_i$和$O_j$之间的边$\mathcal{\varepsilon}$计算：$e_{i,j} = U(IoU_{i,j} - Thr_{IoU}) \times U(PSim_{i,j} - Thr_{PSim})$，$Thr_{IoU}=0.5$，$Thr_{PSim} = 0.9$，$U(\cdot)$是单位阶跃函数

极大联通子图$\{ \mathcal{H}_k \}_{k'=1}^{K'}$，删除只有一个节点（提案）的子图，合并剩下子图$\mathcal{H}_{k'}$，最后得到K个合并的提案$\{\tilde{O}_k \}_{k=1}^K$

用**第二步骤**计算相似熵$\{ \mathcal{E}_k \}_{k'=1}^{K'}$和目标得分$\{ \mathcal{S}_k \}_{k'=1}^{K'}$，如果某个提案的熵值大于最大熵，即$\tilde{E}_k > \max \limits_{c=1,...,C} E_t $，则删除。

**4.提案回归**