# Position Encoding

Position encoding becomes more interesting once the model is no longer reading only text. In a pure language model, position is a one-dimensional notion: each token has a place in a sequence. In a vision-language or omni model, that assumption breaks down. Image tokens live on a 2D grid, video tokens live on a spatiotemporal grid, and audio tokens carry their own notion of time. The design question is no longer only how to encode order, but how to encode structure.

This page organizes three related ideas in that progression:

- **RoPE** for one-dimensional token sequences
- **M-RoPE** for multimodal or spatiotemporal coordinates
- **TM-RoPE** for tighter time alignment across modalities, especially audio and video

The formalism below keeps the essential equations, but the emphasis is on what each method is trying to preserve.

## RoPE

Rotary Position Embedding encodes position by rotating the query and key vectors in paired two-dimensional subspaces. Let

$$
x \in \mathbb{R}^d
$$

and split its dimensions into pairs:

$$
x = (x_0, x_1, x_2, x_3, \ldots).
$$

For position \(p\), the \(i\)-th two-dimensional block is rotated by an angle

$$
\theta_i(p) = p \, \omega_i,
$$

where \(\omega_i\) is the frequency assigned to that block. The rotated block is

$$
\begin{pmatrix}
x'_{2i} \\
x'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos \theta_i(p) & -\sin \theta_i(p) \\
\sin \theta_i(p) & \cos \theta_i(p)
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}.
$$

The standard frequency schedule is geometric:

$$
\omega_i = 10000^{-2i/d}.
$$

The appeal of RoPE is that it makes attention depend naturally on **relative position**. If \(q\) and \(k\) are rotated at positions \(p\) and \(p'\), then their dot product depends on \(p-p'\) rather than only on the two absolute indices. This is why RoPE is more than an additive absolute position vector: it changes the geometry of the query-key interaction itself.

For text, this is a very natural fit. Tokens form a one-dimensional sequence, and the model mainly needs to know which token is earlier or later and by how much. The limitation appears when the input is no longer intrinsically 1D.

## M-RoPE

In multimodal models such as Qwen2.5-VL, one token may correspond not only to a sequence index, but also to a location on a visual grid or to a frame in a video. A patch token is better described by coordinates such as

$$
(t, h, w),
$$

where \(t\) is time, \(h\) is height, and \(w\) is width. M-RoPE extends RoPE to that setting.

One useful abstract way to view M-RoPE is to split the hidden dimensions into subspaces assigned to different axes:

$$
d = d_t + d_h + d_w.
$$

Then the model applies separate rotary encodings to different slices of the hidden state:

$$
\mathrm{M\mbox{-}RoPE}(x; t,h,w)
=
\big[
\mathrm{RoPE}_t(x_t; t),
\mathrm{RoPE}_h(x_h; h),
\mathrm{RoPE}_w(x_w; w)
\big].
$$

This expression should be read as a conceptual decomposition rather than the exact implementation detail of every model. The key idea is that different coordinate axes receive different rotary phases, so the attention score can depend on

$$
(t-t', h-h', w-w').
$$

That is the modeling jump from RoPE to M-RoPE: the position signal is no longer a scalar, but a structured coordinate.

In the Qwen2.5-VL family, the official documentation describes MRoPE together with dynamic resolution for images, dynamic FPS sampling for video, and absolute-time handling for temporal positions. In practical terms, this means the model does not treat a video as only a flat sequence of patches. It keeps track of temporal and spatial layout explicitly, and the temporal identifiers can be aligned to real sampling intervals rather than only frame count. This is one of the reasons Qwen2.5-VL behaves much more like a video model than a text model with images appended to it.

The main limitation of the split-subspace view is that the available frequency spectrum is also split. If each axis gets only part of the hidden dimensions, then each axis also gets only part of the usable frequencies. For short images this is usually acceptable. For long videos, especially when temporal understanding matters more than static spatial detail, this can become a real modeling constraint.

## TM-RoPE

TM-RoPE is best understood as the next step after M-RoPE, motivated by omni settings in which the model must reason jointly over video and audio. At that point, time is no longer just one coordinate among several. It becomes the axis along which different modalities must be synchronized.

The Qwen2.5-Omni project describes **TMRoPE** as a mechanism introduced to synchronize video and audio timestamps when the two modalities are interleaved in one sequence. That description is important because it tells us what problem TM-RoPE is solving: not merely "more dimensions," but **cross-modal temporal alignment**. In other words, an audio token and a video token that originate from the same physical time should remain close in the temporal position space even if their tokenization patterns are very different.

At an abstract level, TM-RoPE can be viewed as moving away from a strictly partitioned encoding of

$$
(t,h,w)
$$

and toward a denser, shared phase construction. A convenient conceptual form is

$$
\theta_i(t,h,w)
=
\alpha_i t + \beta_i h + \gamma_i w,
$$

where \(\alpha_i\), \(\beta_i\), and \(\gamma_i\) distribute each frequency block across multiple axes rather than assigning that block to only one axis. The corresponding rotation becomes

$$
R\big(\theta_i(t,h,w)\big),
$$

applied to the \(i\)-th two-dimensional block.

This formula should again be read as a conceptual summary rather than as a claim that every implementation literally uses this exact parameterization. The point is the change in modeling assumption:

- in **M-RoPE**, different coordinate axes are largely encoded in different subspaces
- in **TM-RoPE**, temporal and spatial coordinates can contribute jointly to the same rotary phase

That shift matters for two reasons. First, the temporal dimension is no longer restricted to a small, dedicated slice of the spectrum. Second, time alignment across modalities becomes easier to express, because audio and video can be mapped into a shared temporal reference instead of only being assigned separate local position ids.

## A Unified View

All three methods can be placed in one common template:

$$
\theta_i = \sum_k \lambda_{i,k} \, p_k,
$$

where \(p_k\) are position components and \(\lambda_{i,k}\) determine how each frequency block uses them.

For standard RoPE, there is only one position component:

$$
\theta_i = \omega_i p.
$$

For a split-axis view of M-RoPE, each block primarily serves one coordinate:

$$
\theta_i =
\begin{cases}
\omega_i t, & i \in \mathcal{I}_t \\
\omega_i h, & i \in \mathcal{I}_h \\
\omega_i w, & i \in \mathcal{I}_w
\end{cases}
$$

For a coupled view such as TM-RoPE, one frequency block can mix several coordinates:

$$
\theta_i = \lambda_{i,t} t + \lambda_{i,h} h + \lambda_{i,w} w.
$$

This is the cleanest way to compare them:

- **RoPE**: one-dimensional phase encoding
- **M-RoPE**: multi-dimensional phase encoding by axis decomposition
- **TM-RoPE**: multi-dimensional phase encoding with stronger cross-axis and cross-modal coupling

## Comparison

| Method | Position space | Frequency use | Relative structure preserved | Best mental model |
| --- | --- | --- | --- | --- |
| RoPE | 1D sequence index | Full spectrum for one axis | Relative token order | Sequence rotation |
| M-RoPE | Multi-axis coordinates such as \((t,h,w)\) | Spectrum partitioned across axes | Relative time and space | Axis-wise rotary extension |
| TM-RoPE | Multi-axis and cross-modal time-aligned coordinates | More shared or coupled use of spectrum | Relative time, space, and cross-modal temporal alignment | Unified spatiotemporal phase encoding |

## Practical Interpretation

RoPE is enough when the model only needs to know where a token sits in a text sequence. M-RoPE becomes necessary once the model sees images and videos as genuine structured inputs rather than as flattened lists of tokens. TM-RoPE becomes attractive when the system must align modalities that evolve on the same physical timeline, especially audio and video.

From that perspective, the progression is not merely a story of "better position encoding." It is a story of how the notion of position itself changes:

- from order in a sequence,
- to coordinates in space-time,
- to synchronized coordinates shared across modalities.

## Notes

- The RoPE equations above are standard.
- The M-RoPE and TM-RoPE formulas in this page are **conceptual abstractions** used to explain the design space. Official Qwen documentation describes the role of MRoPE and TMRoPE clearly, but does not always expose a single canonical closed-form equation for every implementation detail.
- For Qwen-specific references, the most relevant public sources are:
  - [Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)
  - [Qwen2.5-VL blog](https://qwenlm.github.io/blog/qwen2.5-vl/)
  - [Qwen2.5-Omni repository](https://github.com/QwenLM/Qwen2.5-Omni)
