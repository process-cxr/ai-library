---
title: Tokenizer
created: 2026-02-22
published: 2026-02-22
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - pretraining
  - tokenizer
---

Tokenizer 是把原始文本映射为离散 token 序列的组件。对 LLM 来说，它不是前处理小工具，而是模型输入空间、输出空间、训练 token 统计、embedding 矩阵和评测口径的一部分。

一旦模型开始预训练，tokenizer 基本就成为模型契约：改变 tokenizer 意味着输入 token ID、embedding、LM head、数据长度分布和 loss 口径都发生变化。除非做专门的 tokenizer adaptation，否则不能把它当作可随意替换的模块。

## Tokenizer 解决什么问题

语言模型需要在离散符号上建模。如果直接使用字符，序列会很长，长程依赖和训练成本上升；如果直接使用词，词表会极大，且无法处理未登录词、代码符号、拼写变化和多语言。Subword tokenizer 在两者之间折中：

- 高频词或片段可以作为单个 token；
- 低频词可以拆成多个 subword；
- 未见过的词仍可由更小单位表示；
- 词表大小保持在可控范围；
- 序列长度比字符级明显更短。

现代 LLM 常见 tokenizer 包括 BPE、byte-level BPE、SentencePiece BPE、SentencePiece unigram 等。

## BPE

Byte Pair Encoding 最初来自压缩算法，在 NLP 中被用于学习 subword vocabulary。基本流程是：

1. 从字符或 byte 级初始符号开始；
2. 统计相邻符号 pair 的频率；
3. 合并最高频 pair，形成新 token；
4. 重复直到达到目标 vocab size。

BPE 的特点是简单、高效、可解释，适合大规模工程使用。它倾向于把高频片段合成较长 token，把低频词拆成较小片段。GPT 系列常使用 byte-level BPE 变体，使任意 Unicode 文本都能被 byte 序列覆盖，避免 unknown token。

风险在于：BPE 合并由频率驱动，不一定符合语言学边界。对形态丰富语言、低资源语言、数学符号和代码标识符，它可能产生不理想切分。

## SentencePiece

SentencePiece 是一种把文本视为 raw Unicode sequence 的 tokenizer 工具，常见模式包括 unigram language model 和 BPE。它的工程价值在于：

- 不依赖预先分词，适合没有空格分词的语言；
- 能统一处理多语言文本；
- 用特殊空格符号显式编码 whitespace；
- 支持 subword regularization 等训练技巧；
- 便于把 tokenizer 训练和文本规范化纳入同一 pipeline。

SentencePiece unigram 会从候选 subword vocabulary 中学习一个概率模型，并通过最大化语料 likelihood 选择词表。相比 BPE 的贪心合并，它更像是在候选切分空间中选择较优子词集合。

## Vocabulary Size

词表大小直接影响模型和数据：

- vocab 越大，平均序列越短，训练 token 数可能减少；
- vocab 越大，embedding 和 LM head 参数越多；
- vocab 越大，低频 token 学习更困难；
- vocab 越小，序列更长，context 被消耗更多；
- vocab 过小会让代码、数学、多语言文本被切得很碎；
- vocab 过大可能浪费容量在稀有片段上。

对于 decoder-only LM，输入 embedding 和输出 LM head 的维度通常与 vocab size 相关。如果 hidden size 为 $d_{\mathrm{model}}$，vocab size 为 $V$，未共享或共享权重的 embedding / output projection 都会带来约 $V \cdot d_{\mathrm{model}}$ 级别参数。因此 tokenizer 会影响参数量口径，也会影响 [[training/scaling/model-data-compute|Model Data and Compute]] 中的 $D$。

## Compression Ratio 与 Fertility

评估 tokenizer 时常看两个概念：

- **Compression ratio**：原始文本长度与 token 数的比例，表示 tokenizer 是否高效压缩文本；
- **Fertility**：一个词、字符或语义单位平均被拆成多少 token，尤其常用于多语言比较。

例如，同一个英文句子可能被切成较少 token，而中文、泰文、阿拉伯语或低资源语言可能被切成更多 token。对代码来说，长变量名、缩进、符号和字符串也会影响 token 数。

这会带来三个后果：

1. 同样的 context length 能容纳的真实文本量不同；
2. 同样的语料在不同 tokenizer 下会得到不同 $D$；
3. 不同 tokenizer 的 PPL 不可直接比较。

因此，报告训练 token 数、loss 或 PPL 时必须说明 tokenizer。

## 多语言、代码与数学

Tokenizer 会塑造模型对特殊领域的表示效率。

对多语言：

- 需要避免低资源语言被拆得过碎；
- 需要检查 Unicode normalization 是否破坏文本；
- 需要考虑没有空格边界的语言；
- 需要统计各语言 token/character、token/word 和 validation loss。

对代码：

- whitespace、indentation 和 newline 很重要；
- camelCase、snake_case、路径、包名、API 名称需要合理切分；
- byte-level fallback 能覆盖任意符号，但可能让罕见符号序列变长；
- 代码 tokenizer 不佳会浪费 context，并削弱结构模式学习。

对数学：

- LaTeX 命令、变量、上下标、公式环境和特殊符号需要稳定表示；
- 如果公式被切得过碎，模型学习长公式结构会更困难；
- 训练数据中的 OCR 噪声会进一步恶化 tokenization。

## Special Tokens 与模型协议

Tokenizer 还定义特殊 token，例如：

- beginning-of-sequence；
- end-of-sequence；
- padding；
- unknown token，若有；
- separator；
- system/user/assistant role token；
- tool call 或 function call 标记；
- fill-in-the-middle 标记；
- document boundary 标记。

在 base pretraining 阶段，special tokens 的使用要谨慎。它们会被模型学习成强格式信号。后续 SFT、RLHF、tool use 和 chat template 往往依赖这些 token 或其等价字符串。如果预训练阶段和后训练阶段协议不一致，可能导致格式迁移困难。

## Tokenizer 与 PPL 可比性

Perplexity 是 token-level 指标：

$$
\mathrm{PPL} = \exp(L)
$$

但 token 是 tokenizer 定义的。两个模型如果 tokenizer 不同，即使在同一文本上计算 PPL，也不应直接比较。一个 tokenizer 把文本切得更细，token-level loss 和 PPL 的含义就改变了。

更稳妥的比较方式包括：

- 使用同 tokenizer 的模型比较 token-level PPL；
- 报告 bits-per-byte 或 byte-normalized loss；
- 在同一 detokenized 文本任务上比较 downstream metrics；
- 对多语言分别报告 fertility 和 normalized loss。

这点对 scaling law 很重要，因为 $D$ 是 tokenizer 后的 token 数。改变 tokenizer 等于改变横轴。

## 训练 Tokenizer 的实践检查

训练 tokenizer 前应明确：

- 目标语言和领域；
- corpus 是否经过去重、清洗和 normalization；
- vocab size；
- byte fallback 或 unknown token 策略；
- special tokens；
- 对代码、数学、表格、URL、emoji、非拉丁文字的处理；
- 是否保留大小写、空格、换行和格式；
- tokenizer 训练数据是否代表最终 pretraining data mix。

训练后应检查：

- 各语言和领域的 token/character；
- 高频 token 和异常 token；
- 长尾 token 利用率；
- 乱码、控制字符和 HTML 残留；
- 代码缩进与换行；
- LaTeX 和公式；
- prompt/chat template 的可逆性；
- encode-decode 是否无损或满足预期。

## 常见失败模式

- **只用英文网页训练 tokenizer**：多语言和代码会被严重碎片化。
- **vocab 过小**：sequence length 被浪费，长上下文有效容量下降。
- **vocab 过大**：embedding/LM head 成本上升，稀有 token 学习不足。
- **normalization 不一致**：训练、评测和部署时同一文本被编码不同。
- **special token 规划不足**：后续 chat、tool、FIM 或多模态协议难以扩展。
- **跨 tokenizer 比较 PPL**：容易得到误导性结论。
- **忽略 tokenizer 对数据规模的影响**：同一 corpus 的 token count 会因 tokenizer 改变，从而影响 compute estimate。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[fundamentals/information-theory/perplexity|Perplexity]]
- [[sources/papers/2016-neural-machine-translation-of-rare-words-with-subword-units|Neural Machine Translation of Rare Words with Subword Units]]
- [[sources/papers/2018-sentencepiece|SentencePiece]]
