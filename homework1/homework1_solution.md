Name: 鲁旋 \
Student ID: 220810118 \
Email: 220810118@stu.hit.edu.cn

## Exercise 1

($p_i \geq 0$ 由目标函数的定义域与最优解的指数族结构隐含保障，故可不显式写出。)

### 1. 拉格朗日对偶函数推导
原始问题（采用最大化形式）
$$ \max_{p} ; H(p) = -\sum_{i=1}^K p_i \log p_i. $$

约束： 
$$ \sum_{i=1}^K p_i = 1,\quad \sum_{i=1}^K p_i f_i = F,\quad (p_i \ge 0). $$

拉格朗日函数
引入乘子 $α$（对应归一化）与 $β$（对应期望约束）：

$$\mathcal{L}(p,α,β) = -\sum_{i=1}^K p_i \log p_i +
α\left(\sum_{i=1}^K p_i - 1\right) +
β\left(\sum_{i=1}^K p_i f_i - F\right).$$

对每个 $p_i$ 求偏导并令零：
$$
\begin{align*}
&\frac{\partial \mathcal{L}}{\partial p_i} = -( \log p_i + 1 ) + α + β f_i = 0 \\ 
&\Rightarrow 
\log p_i = α + β f_i - 1 \\  
&\Rightarrow
p_i = e^{α-1} e^{β f_i}.
\end{align*}
$$
记配分函数（规范化因子）
$$
Z(β) = \sum_{j=1}^K e^{β f_j}.
$$
利用约束 $\sum p_i = 1$:
$$
1 = e^{α-1} Z(β) \Rightarrow e^{α-1} = \frac{1}{Z(β)}.
$$
于是最优（对给定 $β$）的分布：
$$
p_i^*(\beta) = \frac{e^{β f_i}}{Z(β)}.
$$
代回得到对偶函数.
首先计算熵项在最优 $p^*$ 上:
$$
\begin{align*}
-\sum p_i^* \log p_i^* &= -\sum p_i^* \big(β f_i - \log Z(β)\big) \\
&= -β \sum p_i^* f_i + \log Z(β)\sum p_i^* \\
&= -β F + \log Z(β).
\end{align*}
$$
在最优 $p^*$ 上，两条等式约束正好满足，因此含 $α$ 与 $β$ 的线性罚项为零（$α(∑p_i^* -1)=0, β(∑p_i^* f_i - F)=0）$。故对偶函数：
$$
g(β) = \log Z(β) - β F = \log\Big(\sum_{i=1}^K e^{β f_i}\Big) - β F.
$$
$α$ 已被消去（其值可写为 $α = 1 - \log Z(β)$）。

对偶问题
$$
 \min_{β \in \mathbb{R}} g(β) = \min_{β} \Big[ \log\Big(\sum_{i=1}^K e^{β f_i}\Big) - β F \Big].
$$
一阶最优条件:
$$
\begin{align*}
&g'(β) = \frac{\sum f_i e^{β f_i}}{\sum e^{β f_i}} - F = \mathbb{E}{p_β}[f] - F = 0 \\
&\Rightarrow \sum_{i=1}^K p_i^*(β) f_i = F.
\end{align*}
$$
从而 $β^*$ 的解正好“匹配”原始的期望约束。

强对偶与可行性
- 原问题：在（概率单纯形 ∩ 仿射子空间）上最大化严格凹函数 → 凸优化（等价最小化凹函数的负值）。
- 若 $F$ 位于集合 $\{f_i\}$ 的凸包内部，则存在严格正的可行分布（Slater 条件），故强对偶成立（对偶间隙为 0）。
- 边界情形（$F$ 在凸包边界）可能使部分 $p_i^*=0$，仍可取极限维持对偶最优性。

最终结果汇总
- 最优分布（Primal Optimal）： $ p_i^* = \frac{e^{β^* f_i}}{\sum_{j} e^{β^* f_j}}, β^* \text{ 由 } \sum_i p_i^* f_i = F \text{ 确定}. $
- 对偶函数（Dual Function）： $ g(β) = \log\Big(\sum_{i=1}^K e^{β f_i}\Big) - β F. $
- 对偶问题（Dual Problem）： $ \min_{β \in \mathbb{R}} g(β). $

### 2. 证明最大熵解属于指数族并给出形式
原始最大熵问题
$$ \max_{p}; H(p)= -\sum_{i=1}^K p_i \log p_i \quad \text{s.t.}\quad \sum_{i=1}^K p_i = 1, \sum_{i=1}^K p_i f_i = F, p_i\ge 0. $$

构造拉格朗日函数并求极值
为了得到题目中所给的 $\exp(-\lambda f_i)$ 形式，我们对期望约束采用乘子放在形式 $\lambda(F - \sum p_i f_i)$（而不是 $\lambda(\sum p_i f_i - F)$：

$$ \mathcal{L}(p,\alpha,\lambda) = -\sum_{i=1}^K p_i \log p_i + \alpha\Big(\sum_{i=1}^K p_i - 1\Big) + \lambda\Big(F - \sum_{i=1}^K p_i f_i\Big). $$

对每个 $p_i$ 求偏导并令零： 
$$ 
\frac{\partial \mathcal{L}}{\partial p_i} = -( \log p_i + 1 ) + \alpha - \lambda f_i = 0 \Rightarrow \log p_i = \alpha - 1 - \lambda f_i \Rightarrow p_i = e^{\alpha-1} e^{-\lambda f_i}. 
$$

令 $Z_\lambda = \sum_{k=1}^K e^{-\lambda f_k}$, 利用归一化约束 $\sum_i p_i =1$ 得 $e^{\alpha-1} = \frac{1}{Z_\lambda}$, 从而 $\boxed{p_i = \frac{1}{Z_\lambda}\exp(-\lambda f_i)}$.
这正是题目所给形式。

指数族表示.
指数族一般形如 $p_i = h(i)\exp\big(\eta T(i) - A(\eta)\big)$, 其中 $A(\eta)=\log \sum_i h(i)\exp(\eta T(i))$
在这里可取：
- 基函数：$h(i)=1$,
- 充分统计量：$T(i)= -f_i$（或等价地把自然参数取为 $\theta=-\lambda$ 并令 $T(i)=f_i$）,
- 自然参数：$\eta = \lambda$（或 $\theta=-\lambda$）,
- 对数配分函数：$A(\eta)=\log Z_\lambda$.

因此最大熵分布就是一个一维（单参数）指数族.

参数 $\lambda$ 的确定
期望约束： $ \sum_{i=1}^K p_i f_i = F. $

用上面的分布： $ E_{p_\lambda}[f] = \frac{\sum_i f_i e^{-\lambda f_i}}{\sum_i e^{-\lambda f_i}} = F. $

又因为 $ Z_\lambda = \sum_i e^{-\lambda f_i},\quad \log Z_\lambda' = \frac{Z_\lambda'}{Z_\lambda} = \frac{\sum_i (-f_i) e^{-\lambda f_i}}{Z_\lambda} = - E_{p_\lambda}[f], $ 所以 $ E_{p_\lambda}[f] = -\frac{d}{d\lambda}\log Z_\lambda, $ 约束等价于 $ -\frac{d}{d\lambda}\log Z_\lambda = F. $

存在与唯一性
设 $f_{\min} = \min_i f_i$, $f_{\max} = \max_i f_i$。
可证明函数 $\lambda \mapsto E_{p_\lambda}[f]$ 单调下降：

$$ \frac{d}{d\lambda}E_{p_\lambda}[f] = -\operatorname{Var}_{p_\lambda}(f) \le 0.$$

极限：

- $(\lambda \to +\infty)$: 权重集中在 $f$ 最小的索引，$(E[f]\to f_{\min})$。
- $(\lambda \to -\infty)$: 权重集中在 $f$ 最大的索引，$(E[f]\to f_{\max})$。

因此当且仅当 $F \in [f_{\min}, f_{\max}]$（更精细地：若要严格正分布则需 $F$ 在开区间）时，存在唯一 $\lambda$ 使得约束成立。


## Exercise 2

### LASSO 回归与压缩感知的异同

#### 共同点

- **数学形式**：两者的优化目标完全一致，都是
  $$
  \min_{w} \frac{1}{2} \|y - Bw\|_2^2 + \lambda \|w\|_1
  $$
  其中 $w$ 可以是 $\beta$（LASSO）或 $x$（压缩感知），$B$ 可以是 $X$ 或 $A$。
- **稀疏性**：都利用 $\ell_1$ 正则化鼓励解的稀疏性。
- **应用场景**：都可用于变量选择、信号恢复等。

---

#### 区别

| 方面         | LASSO 回归                                   | 压缩感知（Compressed Sensing）                |
| ------------ | ------------------------------------------- | --------------------------------------------- |
| **目标**     | 统计建模、变量选择、回归预测                 | 稀疏信号恢复、信息重建                        |
| **$B$ 矩阵** | $X$ 是特征矩阵，通常 $n$ 样本 $\times$ $p$ 特征 | $A$ 是测量矩阵，通常 $m \ll n$，$x$ 稀疏      |
| **$w$ 物理意义** | $\beta$ 是回归系数，反映特征对输出的影响      | $x$ 是原始信号，目标是从少量观测恢复稀疏信号   |
| **数据假设** | $X$ 列相关性低，$n \geq p$ 或 $n < p$ 均可    | $A$ 满足 RIP 或随机性，$x$ 必须稀疏            |
| **应用领域** | 统计学习、机器学习、基因数据、经济建模等      | 信号处理、图像重建、雷达、MRI、压缩采样等      |
| **解释性**   | 关注变量选择和模型解释性                      | 关注信号重建精度，变量本身无解释性             |

---

### 总结

- **LASSO** 关注于“解释”与“预测”，主要用于统计建模和特征选择。
- **压缩感知** 关注于“重建”与“采样”，主要用于从少量观测中恢复原始稀疏信号。
- 虽然数学形式一致，但背景假设、变量含义和应用目标不同。


## Exercise 3

### Compressed Sensing in Python 教程学习总结

#### 1. 理论与实践的桥梁

- **压缩感知（Compressed Sensing, CS）** 理论指出：只要信号在某个基下是稀疏的，即使观测远少于信号长度，也能准确重建原信号。
- 关键思想：利用 $\ell_1$ 正则化（如 LASSO/Basis Pursuit）将不可解的欠定线性系统转化为可解的稀疏优化问题。

---

#### 2. Python 实现与实验

- 教程通过 Python 代码演示了如何生成稀疏信号、构造观测矩阵、采样观测、并用 $\ell_1$ 优化重建信号。
- 主要步骤包括：
  - **信号生成**：随机生成稀疏向量 $x_0$。
  - **观测矩阵**：通常用高斯随机矩阵 $A$。
  - **采样**：$y = Ax_0$。
  - **重建**：用 `scipy.optimize` 或 `sklearn.linear_model.Lasso` 等工具求解
    $$
    \min_x \frac{1}{2}\|y - Ax\|_2^2 + \lambda \|x\|_1
    $$
- 代码展示了不同稀疏度、观测数、噪声水平下的重建效果。

---

#### 3. 主要收获

- **理论与数值实验一致**：只要观测数 $m$ 足够，且信号足够稀疏，$\ell_1$ 优化能准确恢复原信号。
- **观测矩阵的选择**：高斯随机矩阵通常满足 RIP 条件，适合压缩感知。
- **噪声影响**：有噪声时，重建精度下降，但 $\ell_1$ 正则化仍能抑制噪声影响，恢复主要稀疏分量。
- **参数选择**：$\lambda$ 的选取影响稀疏性与拟合度，需结合实际调优。
- **可视化**：通过画图直观展示了原信号、观测、重建信号的对比，帮助理解算法效果。

---

#### 4. 实践建议

- 实际应用中，需关注观测矩阵的设计、信号稀疏性假设是否成立、噪声水平等。
- Python 生态（如 `scikit-learn`, `cvxpy`, `scipy.optimize`）为压缩感知实验和应用提供了丰富工具。

---

#### 5. 总结

- 教程很好地将压缩感知的理论与 Python 实践结合，帮助理解 $\ell_1$ 优化在信号恢复中的作用。
- 通过动手实验，更直观地体会了“少量观测+稀疏性假设=高质量重建”的核心思想。

