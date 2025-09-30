vocab_size : 50257
context_length : 1024
num_layers : 48
d_model : 1600
num_heads : 25
d_ff : 6400

vocab_size  50257
context_length  1024
num_layers  48
d_model 1600
num_heads   25
d_ff    6400


Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?
Deliverable: A one-to-two sentence response.

embedding 
    w vocab_size x d_model 
transformer block 
    rms_norm_0 d_model
    attn 4 x d_model x d_model
    rms_norm_1 d_model
    fnn 3 x d_model x d_ff
rms_norm_0 g d_model
linear w d_model 

(vocab_size * d_model) + num_layers * (d_model + 4 * d_model * d_model + d_model + 3 * d_model * d_ff) + d_model + d_model

(vocab_size * d_model) + num_layers * d_model (1 + 4 * d_model + 1 + 3 * d_ff) + 2 * d_model

d_model * vocab_size + num_layers * d_model (2 + 4 * d_model + 3 * d_ff) + 2 * d_model

parameters_count = d_model * (vocab_size + num_layers * (2 + 4 * d_model + 3 * d_ff) + 2)

parameters_count =  1600  * (50257 + 48  * (2 + 4 * 1600 + 3 * 6400) + 2) =2 046 648 000
bytes = parameters_count * 4 = 8 186 592 000
7.624 GB

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
required.

embedding layer:
    0, only indexing operation 
    
transformer block: x num_layers
    rms_norm_0:
        ignore
    attn: without rope
        [batch, seq_len, d_model] x [d_model, d_model] 4 QKV + O proj = 2 * 4 * batch * seq_len * d_model^2  20 971 520 000
        [batch, seq_len, d_model] x [batch, d_model, seq_len] QK^T  = 2 * batch * d_model * seq_len^2 3 355 443 200
        [batch, seq_len, d_model] x [batch, d_model, seq_len] (QK^T) * V  = 2 * batch * d_model * seq_len^2 3 355 443 200
    rms_norm_1:
        ignore
    ffn:
        [batch, seq_len, d_model] x 3 [d_model x d_ff] = 3 * 2 * batch * seq_len * d_model * d_ff 62 914 560 000
rms_norm_0 
    ignore
linear 
    [batch, seq_len, d_model] x [d_model, vocab_size]  = 2 * batch * seq_len * d_model * vocab_size 164 682 137 600

num_layers * ((2 * 4 * batch * seq_len * d_model^2) + (4 * batch * d_model * seq_len^2) + (3 * 2 * batch * seq_len * d_model * d_ff))
+  ( 2 * batch * seq_len * d_model * vocab_size)

num_layers * ((8 * batch * seq_len * d_model^2) + (4 * batch * d_model * seq_len^2) + (6 * batch * seq_len * d_model * d_ff))
+  ( 2 * batch * seq_len * d_model * vocab_size)

2 * num_layers * batch * seq_len * d_model (4 * d_model + 2*seq_len + 3 * d_ff)
+  ( 2 * batch * seq_len * d_model * vocab_size)


Total FLOPs = 4 513 336 524 800

(c) Based on your analysis above, which parts of the model require the most FLOPs?
Deliverable: A one-to-two sentence response.
last output linear projection


(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?

Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.

#GPT-2 small
vocab_size = 50257
context_length = 1024
num_layers = 12
d_model = 768
num_heads = 12
d_ff = d_model * 4

#GPT-2 medium
vocab_size = 50257
context_length = 1024
num_layers = 24
d_model = 1024
num_heads = 16
d_ff = d_model * 4

#GPT-2 large
vocab_size = 50257
context_length = 1024
num_layers = 36
d_model = 1280
num_heads = 20
d_ff = d_model * 4


#GPT-2 small
embedding layer:
    0, only indexing operation 
    
transformer block: x num_layers
    rms_norm_0:
        ignore
    attn: without rope
        [batch, seq_len, d_model] x [d_model, d_model] 4 QKV + O proj = 2 * 4 * batch * seq_len * d_model**2  4 831 838 208
        [batch, seq_len, d_model] x [batch, d_model, seq_len] QK^T  = 2 * batch * d_model * seq_len**2  1 610 612 736
        [batch, seq_len, d_model] x [batch, d_model, seq_len] (QK^T) * V  = 2 * batch * d_model * seq_len**2  1 610 612 736
    rms_norm_1:
        ignore
    ffn:
        [batch, seq_len, d_model] x 3 [d_model x d_ff] = 3 * 2 * batch * seq_len * d_model * d_ff 14 495 514 624
rms_norm_0 
    ignore
linear 
    [batch, seq_len, d_model] x [d_model, vocab_size]  = 2 * batch * seq_len * d_model * vocab_size 79 047 426 048

#GPT-2 medium
embedding layer:
    0, only indexing operation 
    
transformer block: x num_layers
    rms_norm_0:
        ignore
    attn: without rope
        [batch, seq_len, d_model] x [d_model, d_model] 4 QKV + O proj = 2 * 4 * batch * seq_len * d_model**2  8 589 934 592
        [batch, seq_len, d_model] x [batch, d_model, seq_len] QK^T  = 2 * batch * d_model * seq_len**2  2 147 483 648
        [batch, seq_len, d_model] x [batch, d_model, seq_len] (QK^T) * V  = 2 * batch * d_model * seq_len**2  2 147 483 648
    rms_norm_1:
        ignore
    ffn:
        [batch, seq_len, d_model] x 3 [d_model x d_ff] = 3 * 2 * batch * seq_len * d_model * d_ff 25 769 803 776
rms_norm_0 
    ignore
linear 
    [batch, seq_len, d_model] x [d_model, vocab_size]  = 2 * batch * seq_len * d_model * vocab_size 105 396 568 064

#GPT-2 large
embedding layer:
    0, only indexing operation 
    
transformer block: x num_layers
    rms_norm_0:
        ignore
    attn: without rope
        [batch, seq_len, d_model] x [d_model, d_model] 4 QKV + O proj = 2 * 4 * batch * seq_len * d_model**2 13 421 772 800
        [batch, seq_len, d_model] x [batch, d_model, seq_len] QK^T  = 2 * batch * d_model * seq_len**2 2 684 354 560
        [batch, seq_len, d_model] x [batch, d_model, seq_len] (QK^T) * V  = 2 * batch * d_model * seq_len**2 2 684 354 560
    rms_norm_1:
        ignore
    ffn:
        [batch, seq_len, d_model] x 3 [d_model x d_ff] = 3 * 2 * batch * seq_len * d_model * d_ff 40 265 318 400
rms_norm_0 
    ignore
linear 
    [batch, seq_len, d_model] x [d_model, vocab_size]  = 2 * batch * seq_len * d_model * vocab_size 131 745 710 080



(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?
Deliverable: A one-to-two sentence response.

