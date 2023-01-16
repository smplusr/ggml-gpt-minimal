#include "include.h"




extern bool generate (char *output, gpt_vocab vocab, gpt2_model model, gpt_params params);

int main(int argc, char **argv) {
	char output[4096];

    gpt_params params;
    params.model = MODEL_FILE;

    gpt_params_parse(argc, argv, params); 
   

    gpt_vocab vocab;
    gpt2_model model;


    

	generate (output, vocab, model, params);
	
	printf ("%s\n", output);



    ggml_free(model.ctx);


}

