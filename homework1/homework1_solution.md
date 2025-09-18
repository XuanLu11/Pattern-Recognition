# Compulsory

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