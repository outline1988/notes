### DOA估计的CRB分析

对于单信源和单快拍的窄带DOA估计模型来说，若已知信源波形，则模型变为
$$
\mathbf{y}(t) = \mathbf{a}(\phi_{k})s_{k}(t) + \mathbf{w}(t)
$$
直观上，CRB应该反应出接收数据对于参数的敏感程度。所以，单源单快拍的窄带DOA估计模型对于相位$\phi_{k}$的FI主要由下式决定
$$
\mathbf{d}_{k}^H \mathbf{d}_{k} |s_{k}(t)|^2
$$
其中，$\mathbf{d}_{k}^H \mathbf{d}_{k}$可以认为是导向矢量对于相位差$\phi_{k}$的敏感程度。这个敏感程度会被信号强度$s_{k}(t)$进一步放大，信号越强，接收数据越敏感。而对于单源的多个快拍，多个独立的单快拍FI的叠加，形成
$$
\mathbf{d}_{k}^H \mathbf{d}_{k} \sum\limits_{t=1}^{N} |s_{k}(t)|^2
$$
而对于多源多块拍的窄带DOA模型
$$
\begin{aligned}
\mathbf{y}(t) &= \mathbf{A}\mathbf{s}(t) + \mathbf{w}(t) \\
&= \sum\limits_{k=1}^{K} \mathbf{a}(\phi_{k}) s_{k}(t) +  \mathbf{w}(t)
\end{aligned}
$$
FIM由下式决定
$$
(\mathbf{D}^H \mathbf{D}) \odot \hat{\mathbf{P}}^T
$$
其某一元素$J_{k, p}$由下式决定
$$
\mathbf{d}_{k}^H \mathbf{d}_{p} \sum\limits_{t=1}^{N} s_{k}^*(t)s_{p}(t)^2
$$
其中
$$
[\mathbf{d}_{k}]_{m} = \left[ \frac{\partial \mathbf{a}(\phi_{k})}{\partial \phi_{k}} \right]_{m} = \mathrm{j}(m - 1) \exp[ \mathrm{j} \phi_{k}(m - 1) ]
$$
$$
\mathbf{d}_{k}^H \mathbf{d}_{p} = \sum\limits_{m = 1}^M (m - 1)^2 \exp[\mathrm{j} (\phi_{p} - \phi_{k})(m - 1)]
$$

$\mathbf{d}_{k}^H \mathbf{d}_{p}$可以理解为，将序列$\{ m \}_{m = 0}^{M - 1}$投影到某个傅里叶基$\exp[\mathrm{j} (\phi_{p} - \phi_{k})(m - 1)]$中，所以在不同$\Delta \phi$的情况下（不同角度差），投影的情况为（横坐标为$\pi$）

![image-20251001183731839](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20251001183731839.png)


由式可知，FIM的对角线元素仍然保持不变，而最终的CRB则受到非对角线元素的影响。而非对角线元素主要由两个因素决定：一是DOA真值的角度差，角度差越小，信源离得越近，则对角线元素中的$\mathbf{d}_{k}^H \mathbf{d}_{p}$就越大；其二是信源波形的影响，$\sum\limits_{t=1}^{N} s_{k}^*(t)s_{p}(t)$是信源协方差矩阵的某一元素$\hat{\mathbf{P}}_{{k, p}}$，所以信源之间相关性越大，该项就越大。 











