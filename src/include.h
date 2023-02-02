#pragma once


#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>


#include "ggml.h"
#include "utils.h"



#ifdef GPT_2_1558M
#define MODEL_FILE "models/gpt-2-1558M/ggml-model.bin"
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 1600;
    int32_t n_head  = 25;
    int32_t n_layer = 48;
    int32_t f16     = 1;
};
#elif GPT_2_774M
#define MODEL_FILE "models/gpt-2-774M/ggml-model.bin"
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 1280;
    int32_t n_head  = 20;
    int32_t n_layer = 36;
    int32_t f16     = 1;
};
#elif GPT_2_345M
#define MODEL_FILE "models/gpt-2-345M/ggml-model.bin"
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 1024;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t f16     = 1;
};
#else
#define MODEL_FILE "models/gpt-2-117M/ggml-model.bin"
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t f16     = 1;
};
#endif

struct gpt2_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w_trans; // transposed for efficiency
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding
    struct ggml_tensor * wpe; //    token embedding

    std::vector<gpt2_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab);
bool gpt2_eval(
        const gpt2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token
);

bool generate (char *output, gpt_vocab vocab, gpt2_model model, gpt_params params);
