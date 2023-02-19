#include "include.h"




extern char *generate (gpt_vocab vocab, gpt2_model model, gpt_params params);

int main(int argc, char **argv) {
    gpt_params params;
    params.model = MODEL_FILE;

    gpt_params_parse(argc, argv, params); 
   
    gpt_vocab vocab;
    gpt2_model model;

    generate (vocab, model, params);

    ggml_free(model.ctx);
}
