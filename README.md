# Your Transformer May Not be as Powerful as You Expect

This repository is the official implementation of “[Your Transformer May Not be as Powerful as You Expect](https://arxiv.org/pdf/2205.13401.pdf)”, based on the official implementation of [Fairseq](https://github.com/facebookresearch/fairseq) and [Graphormer](https://github.com/microsoft/Graphormer) in [PyTorch](https://github.com/pytorch/pytorch).

> Your Transformer May Not be as Powerful as You Expect
>
> *Shengjie Luo\*, Shanda Li\*, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, Di He

## 🔥 News
- **2022.12.01**: Initial Commit. Code will be released soon!

## Overview

![poster](docs/poster.png)

Relative Positional Encoding (RPE), which encodes the relative distance between any pair of tokens, is one of the most successful modifications to the original Transformer. As far as we know, theoretical understanding of the RPE-based Transformers is largely unexplored. In this work, we mathematically analyze the power of RPE-based Transformers regarding whether the model is capable of approximating any continuous sequence-to-sequence functions. One may naturally assume the answer is in the affirmative---RPE-based Transformers are universal function approximators. However, we present a negative result by showing there exist continuous sequence-to-sequence functions that RPE-based Transformers cannot approximate no matter how deep and wide the neural network is. One key reason lies in that most RPEs are placed in the softmax attention that always generates a right stochastic matrix. This restricts the network from capturing positional information in the RPEs and limits its capacity. To overcome the problem and make the model more powerful, we first present sufficient conditions for RPE-based Transformers to achieve universal function approximation. With the theoretical guidance, we develop a novel attention module, called Universal RPE-based (URPE) Attention, which satisfies the conditions. Therefore, the corresponding URPE-based Transformers become universal function approximators. Extensive experiments covering typical architectures and tasks demonstrate that our model is parameter-efficient and can achieve superior performance to strong baselines in a wide range of applications. 

## Citation

If you find this work useful, please kindly cite following papers:

```latex
@inproceedings{
luo2022your,
title={Your Transformer May Not be as Powerful as You Expect},
author={Shengjie Luo and Shanda Li and Shuxin Zheng and Tie-Yan Liu and Liwei Wang and Di He},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=NQFFNdsOGD}
}
```

## Contact

Shengjie Luo (luosj@stu.pku.edu.cn)

Sincerely appreciate your suggestions on our work!

## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/lsj2408/Transformer-M/blob/main/LICENSE) for additional details.
